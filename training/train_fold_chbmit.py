"""
grn_balladeer.training.train_fold_chbmit
============================================
Single-fold training/evaluation for CHB-MIT, same structure as
training.train_fold_biciv2a (labels vary per window within a subject,
not per subject -- see that module's docstring for the underlying
rationale, identical here). Kept as its own file rather than merged
with the biciv2a version because default channel list, default
symbolic cluster, and the loss module imported
(train_epoch_batched_chbmit) all differ.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from grn_balladeer.data.build_dataset_chbmit import CHBMIT_CHANNELS
from grn_balladeer.losses.epilepsy_channels import EPILEPSY_CHANNELS
from grn_balladeer.model.grn_encoder import GRNEncoder, build_resonance_head
from grn_balladeer.model.classification_head import ClassificationHead, global_pool, split_real_imag
from grn_balladeer.training.train_epoch_batched_chbmit import train_epoch_batched_chbmit
from grn_balladeer.training.omega_diagnostics import check_omega_collapse
from grn_balladeer.eval.baselines import evaluate

DatasetBySubject = Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]]
LabelsBySubject = Dict[str, np.ndarray]  # subject_id -> (n_windows,) int array, 1=ictal/0=interictal


def _flatten_subjects_chbmit(
    subject_ids: List[str],
    dataset_by_subject: DatasetBySubject,
    labels_by_subject: LabelsBySubject,
    device: torch.device,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor, List[str]]:
    """Per-window equivalent of train_fold's _flatten_subjects -- same
    logic as train_fold_biciv2a's _flatten_subjects_biciv2a, duplicated
    rather than shared because the two live in domain-specific modules
    by convention (see that module's docstring)."""
    batch, labels, ids = [], [], []
    for sid in subject_ids:
        subject_graphs = dataset_by_subject[sid]
        subject_labels = labels_by_subject[sid]
        if len(subject_graphs) != len(subject_labels):
            raise ValueError(
                f"_flatten_subjects_chbmit: subject {sid} has {len(subject_graphs)} graphs "
                f"but {len(subject_labels)} labels -- must match 1:1."
            )
        for (X_i, L_norm_i), label in zip(subject_graphs, subject_labels):
            batch.append((X_i.to(device), L_norm_i.to(device)))
            labels.append(int(label))
            ids.append(sid)
    return batch, torch.tensor(labels, dtype=torch.long, device=device), ids


def train_fold_chbmit(
    train_subject_ids: List[str],
    val_subject_ids: List[str],
    dataset_by_subject: DatasetBySubject,
    labels_by_subject: LabelsBySubject,
    ch_names: List[str] = CHBMIT_CHANNELS,
    symbolic_channels: List[str] = EPILEPSY_CHANNELS,
    n_epochs: int = 30,
    lr: float = 1e-3,
    seed: int = 42,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    embedding_dim: int = 8,
    device: Optional[torch.device] = None,
    eval_every: int = 20,
    batch_size: Optional[int] = 64,
    weight_decay: float = 1e-4,
    dropout: float = 0.3,
) -> dict:
    """Trains ONE LOSO fold from scratch and evaluates on the held-out
    subject. EEG-only, no dual-branch (no auxiliary modality for
    CHB-MIT). lambda2=0.0 runs the L_symb-disabled ablation condition
    (see losses.epilepsy_channels for why this domain tests both
    conditions rather than committing to one).

    Returns: {'eval_result', 'history', 'val_trajectory',
    'val_subject_ids', 'final_omega_collapse', 'encoder',
    'resonance_head', 'head'}.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_batch, train_labels, train_ids = _flatten_subjects_chbmit(
        train_subject_ids, dataset_by_subject, labels_by_subject, device
    )
    val_batch, val_labels, val_ids = _flatten_subjects_chbmit(
        val_subject_ids, dataset_by_subject, labels_by_subject, device
    )

    in_channels = train_batch[0][0].shape[1]
    torch.manual_seed(seed)
    encoder = GRNEncoder(in_channels=in_channels, hidden_channels=[16, embedding_dim], K=3).to(device)
    resonance_head = build_resonance_head(embedding_dim=embedding_dim).to(device)
    head = ClassificationHead(in_features=2 * embedding_dim, n_classes=2, dropout=dropout).to(device)

    params = list(encoder.parameters()) + list(resonance_head.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    class_counts = torch.bincount(train_labels, minlength=2).float()
    class_weights = (class_counts.sum() / (2.0 * class_counts)).to(device)

    train_X_batch = torch.stack([X_i for X_i, _ in train_batch])
    train_L_batch = torch.stack([L_i for _, L_i in train_batch])
    val_X_batch = torch.stack([X_i for X_i, _ in val_batch])
    val_L_batch = torch.stack([L_i for _, L_i in val_batch])

    def _evaluate_val_batched():
        encoder.eval(); resonance_head.eval(); head.eval()
        with torch.no_grad():
            h_val = encoder(val_X_batch, val_L_batch)
            z_val = global_pool(split_real_imag(h_val), method="mean")
            logits_val = head(z_val)
            probs_val = torch.softmax(logits_val, dim=-1)[:, 1].cpu().numpy()
            preds_val = logits_val.argmax(dim=-1).cpu().numpy()
        encoder.train(); resonance_head.train(); head.train()
        return evaluate(val_labels.cpu().numpy(), preds_val, probs_val)

    history = []
    val_trajectory = []
    n_train_samples = train_X_batch.shape[0]

    for epoch_idx in range(n_epochs):
        if batch_size is None or batch_size >= n_train_samples:
            stats = train_epoch_batched_chbmit(
                encoder, head, resonance_head, train_X_batch, train_L_batch, train_labels, ch_names, optimizer,
                symbolic_channels=symbolic_channels, lambda1=lambda1, lambda2=lambda2, class_weights=class_weights,
            )
        else:
            perm = torch.randperm(n_train_samples, device=device)
            last_stats = None
            for start in range(0, n_train_samples, batch_size):
                idx = perm[start:start + batch_size]
                last_stats = train_epoch_batched_chbmit(
                    encoder, head, resonance_head, train_X_batch[idx], train_L_batch[idx], train_labels[idx],
                    ch_names, optimizer, symbolic_channels=symbolic_channels,
                    lambda1=lambda1, lambda2=lambda2, class_weights=class_weights,
                )
            stats = last_stats
        history.append(stats)

        if eval_every > 0 and (epoch_idx % eval_every == 0 or epoch_idx == n_epochs - 1):
            val_trajectory.append((epoch_idx, _evaluate_val_batched()))

    result = _evaluate_val_batched()
    collapse = check_omega_collapse(history[-1]["last_omega"])

    encoder.eval(); resonance_head.eval(); head.eval()

    return {
        "eval_result": result,
        "history": history,
        "val_trajectory": val_trajectory,
        "val_subject_ids": val_subject_ids,
        "final_omega_collapse": collapse,
        "encoder": encoder,
        "resonance_head": resonance_head,
        "head": head,
    }
