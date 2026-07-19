"""
grn_balladeer.training.cross_validation
===========================================
Module 9 — "Implement train_fold / run_cross_validation" (Week 6).
Full subject-level k-fold CV loop, EEG-only or dual-branch, built on top
of everything validated in Weeks 1-5: stratified_subject_kfold (label+
sex+age_bin), train_epoch/train_epoch_dual_branch, and eval.baselines'
extended EvalResult (balanced_accuracy/sensitivity/specificity, not just
accuracy - see that module's docstring for why that matters given the
real ~64%/36% class imbalance).

DESIGN CHOICE: run_cross_validation takes a PRE-BUILT dataset_by_subject
dict (subject_id -> list of (X_i, L_norm_i)), not raw file paths. Building
a subject's graph dataset (load->filter->ICA->CQT->connectivity) is
expensive and does NOT change across folds - only which subjects land in
train vs val changes. Build each subject's dataset ONCE (via
data.build_dataset.build_subject_dataset) before calling this, not once
per fold. On the full 138-subject Colab run this matters a lot; on this
session's 4-subject smoke test it would not have mattered either way,
but the design should already reflect the scale it's meant for.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from grn_balladeer.data.labels import stratified_subject_kfold
from grn_balladeer.model.grn_encoder import GRNEncoder, build_resonance_head
from grn_balladeer.model.classification_head import ClassificationHead, global_pool, split_real_imag
from grn_balladeer.model.aux_branch_encoder import AuxBranchEncoder
from grn_balladeer.model.cross_attention_fusion import CrossAttentionFusion
from grn_balladeer.training.train_epoch import train_epoch
from grn_balladeer.training.train_epoch_dual_branch import train_epoch_dual_branch
from grn_balladeer.training.omega_diagnostics import check_omega_collapse
from grn_balladeer.eval.baselines import evaluate, EvalResult
from grn_balladeer.preprocessing.mne_loading import CGX_CHANNELS


def _flatten_subjects(
    subject_ids: List[str],
    dataset_by_subject: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]],
    labels_by_subject: Dict[str, int],
    device: torch.device,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor, List[str]]:
    """Expands a list of subject ids into a flat (epoch-level) batch,
    labels tensor, and parallel subject-id-per-epoch list - the format
    train_epoch/train_epoch_dual_branch actually consume.

    Moves each (X_i, L_norm_i) pair to `device` here, once, rather than
    inside the per-epoch training loop -- downstream code (model/
    losses) already threads `device=some_tensor.device` through instead
    of hardcoding 'cpu', so getting the INPUT tensors onto the right
    device once is enough to get the whole forward/backward pass
    running there. Confirmed by grep across model/losses/training: no
    module ever called .cuda() or .to(device) anywhere before this fix
    -- meaning every previous run (including the first full 114-subject
    CV run this session) silently ran on CPU even with a GPU attached
    (0.0/15.0 GB GPU RAM used during that run is the direct evidence).
    """
    batch, labels, ids = [], [], []
    for sid in subject_ids:
        for X_i, L_norm_i in dataset_by_subject[sid]:
            batch.append((X_i.to(device), L_norm_i.to(device)))
            labels.append(labels_by_subject[sid])
            ids.append(sid)
    return batch, torch.tensor(labels, dtype=torch.long, device=device), ids


def train_fold(
    train_subject_ids: List[str],
    val_subject_ids: List[str],
    dataset_by_subject: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]],
    labels_by_subject: Dict[str, int],
    aux_vectors_by_subject: Optional[Dict[str, np.ndarray]] = None,
    ch_names: List[str] = CGX_CHANNELS,
    n_epochs: int = 30,
    lr: float = 1e-3,
    seed: int = 42,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    lambda3: float = 1.0,
    embedding_dim: int = 8,
    device: Optional[torch.device] = None,
) -> dict:
    """Trains ONE fold from scratch (fresh model, fresh optimizer -
    folds must not share weights, that would leak information across
    the CV) and evaluates on the held-out val subjects.

    aux_vectors_by_subject: if None, runs EEG-only (train_epoch). If
    given (subject_id -> 12-dim vector, from model.aux_branch_encoder.
    build_aux_vector), runs dual-branch (train_epoch_dual_branch) - the
    same function handles both modes, matching Week 6's "EEG-only vs
    dual-branch" comparison requirement without duplicating the loop.

    device: defaults to CUDA if available, else CPU. Every model
    (encoder, resonance_head, head, and aux_encoder/fusion in
    dual-branch mode) is moved there, and _flatten_subjects moves the
    input tensors there too -- both sides must match or torch raises.

    Returns: {'eval_result': EvalResult, 'history': [...], 'val_subject_ids':
    [...], 'final_omega_collapse': OmegaCollapseReport}
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dual_branch = aux_vectors_by_subject is not None

    train_batch, train_labels, train_ids = _flatten_subjects(train_subject_ids, dataset_by_subject, labels_by_subject, device)
    val_batch, val_labels, val_ids = _flatten_subjects(val_subject_ids, dataset_by_subject, labels_by_subject, device)

    in_channels = train_batch[0][0].shape[1]
    torch.manual_seed(seed)
    encoder = GRNEncoder(in_channels=in_channels, hidden_channels=[16, embedding_dim], K=3).to(device)
    resonance_head = build_resonance_head(embedding_dim=embedding_dim).to(device)
    head = ClassificationHead(in_features=2 * embedding_dim, n_classes=2).to(device)

    if dual_branch:
        aux_encoder = AuxBranchEncoder().to(device)
        fusion = CrossAttentionFusion().to(device)
        params = (list(encoder.parameters()) + list(resonance_head.parameters()) +
                  list(aux_encoder.parameters()) + list(fusion.parameters()) + list(head.parameters()))
    else:
        params = list(encoder.parameters()) + list(resonance_head.parameters()) + list(head.parameters())

    optimizer = torch.optim.Adam(params, lr=lr)

    history = []
    for _ in range(n_epochs):
        if dual_branch:
            stats = train_epoch_dual_branch(
                encoder, resonance_head, aux_encoder, fusion, head,
                train_batch, train_labels, train_ids, aux_vectors_by_subject, ch_names, optimizer,
                lambda1=lambda1, lambda2=lambda2, lambda3=lambda3,
            )
        else:
            stats = train_epoch(
                encoder, head, resonance_head, train_batch, train_labels, ch_names, optimizer,
                lambda1=lambda1, lambda2=lambda2,
            )
        history.append(stats)

    # --- evaluation on held-out subjects ---
    encoder.eval(); resonance_head.eval(); head.eval()
    if dual_branch:
        aux_encoder.eval(); fusion.eval()

    with torch.no_grad():
        logits_list = []
        for i in range(len(val_batch)):
            X_i, L_norm_i = val_batch[i]
            h_i = encoder(X_i, L_norm_i)
            z_eeg_i = global_pool(split_real_imag(h_i))
            if dual_branch:
                aux_vec = aux_vectors_by_subject[val_ids[i]]
                z_aux_i = aux_encoder(torch.tensor(aux_vec, dtype=torch.float32, device=device).unsqueeze(0))
                z_i, _, _ = fusion(z_eeg_i.unsqueeze(0), z_aux_i)
            else:
                z_i = z_eeg_i.unsqueeze(0)
            logits_list.append(head(z_i))
        val_logits = torch.cat(logits_list, dim=0)
        val_probs = torch.softmax(val_logits, dim=-1)[:, 1].cpu().numpy()  # P(class=1=ADHD)
        val_preds = val_logits.argmax(dim=-1).cpu().numpy()

    result = evaluate(val_labels.cpu().numpy(), val_preds, val_probs)
    collapse = check_omega_collapse(history[-1]["last_omega"])

    return {
        "eval_result": result,
        "history": history,
        "val_subject_ids": val_subject_ids,
        "final_omega_collapse": collapse,
    }


def run_cross_validation(
    label_df: pd.DataFrame,
    dataset_by_subject: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]],
    aux_vectors_by_subject: Optional[Dict[str, np.ndarray]] = None,
    k: int = 5,
    seed: int = 42,
    n_epochs: int = 30,
    device: Optional[torch.device] = None,
    **train_fold_kwargs,
) -> pd.DataFrame:
    """Full k-fold CV: stratified_subject_kfold(label_df, k, seed) ->
    train_fold per fold -> aggregate. Only subjects present in BOTH
    label_df and dataset_by_subject are usable - others are silently
    dropped from the fold (not an error, since dataset_by_subject will
    legitimately not cover every subject in label_df until the full
    Colab run).

    device: defaults to CUDA if available, else CPU -- resolved and
    PRINTED here explicitly (not left to silently default somewhere
    downstream), because that silence is exactly what let the whole
    project run on CPU-only for 6 weeks despite a GPU being attached.
    Forwarded to train_fold via train_fold_kwargs.

    Returns a DataFrame, one row per fold (columns: fold, n_train,
    n_val, accuracy, balanced_accuracy, f1, f1_class0, f1_class1,
    sensitivity, specificity, auc, omega_collapsed), plus a final
    'MEAN' row (excludes the boolean omega_collapsed column) and 'STD'
    row - report BOTH, never just the mean, given how few folds are
    feasible with the current subject count.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run_cross_validation] device utilisé pour tous les folds : {device}")
    train_fold_kwargs["device"] = device

    labels_by_subject = dict(zip(label_df["user_id"], label_df["label"]))
    available_subjects = set(dataset_by_subject.keys())

    folds = stratified_subject_kfold(label_df, k=k, seed=seed)

    rows = []
    for fold_idx, fold in enumerate(folds):
        train_ids = [s for s in fold["train_ids"] if s in available_subjects]
        val_ids = [s for s in fold["val_ids"] if s in available_subjects]
        if not train_ids or not val_ids:
            continue  # this fold has no usable subjects yet - expected on partial data

        result = train_fold(
            train_ids, val_ids, dataset_by_subject, labels_by_subject,
            aux_vectors_by_subject=aux_vectors_by_subject, n_epochs=n_epochs, seed=seed,
            **train_fold_kwargs,
        )
        r: EvalResult = result["eval_result"]
        rows.append({
            "fold": fold_idx, "n_train": len(train_ids), "n_val": len(val_ids),
            "accuracy": r.accuracy, "balanced_accuracy": r.balanced_accuracy,
            "f1": r.f1, "f1_class0": r.f1_class0, "f1_class1": r.f1_class1,
            "sensitivity": r.sensitivity, "specificity": r.specificity, "auc": r.auc,
            "omega_collapsed": result["final_omega_collapse"].is_collapsed,
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise ValueError(
            "run_cross_validation: no fold had usable subjects on both sides - "
            "check that dataset_by_subject covers enough of label_df."
        )

    numeric_cols = [c for c in df.columns if c not in ("fold", "omega_collapsed")]
    mean_row = df[numeric_cols].mean()
    mean_row["fold"] = "MEAN"
    std_row = df[numeric_cols].std()
    std_row["fold"] = "STD"
    return pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
