"""
grn_balladeer.training.cross_validation_biciv2a
===================================================
LOSO (leave-one-subject-out) orchestration for BCI IV 2a: 9 folds, one
subject held out each time -> training.train_fold_biciv2a.
train_fold_biciv2a per fold -> aggregate. The per-fold training logic
itself now lives in training.train_fold_biciv2a (split out on request,
see that module's docstring for the full rationale) -- this file only
handles the fold loop and result aggregation, mirroring
training.cross_validation.run_cross_validation's structure and output
shape (a DataFrame with per-fold rows plus MEAN/STD rows), substituting
data.labels_biciv2a.leave_one_subject_out for stratified_subject_kfold
since there is no demographic metadata to stratify on here.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from grn_balladeer.data.labels_biciv2a import leave_one_subject_out, SUBJECT_IDS
from grn_balladeer.training.train_fold_biciv2a import train_fold_biciv2a
from grn_balladeer.eval.baselines import EvalResult

DatasetBySubject = Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]]
LabelsBySubject = Dict[str, np.ndarray]  # subject_id -> (n_trials,) int array, per-trial


def run_loso_biciv2a(
    dataset_by_subject: DatasetBySubject,
    labels_by_subject: LabelsBySubject,
    subject_ids: List[str] = SUBJECT_IDS,
    seed: int = 42,
    n_epochs: int = 30,
    device: Optional[torch.device] = None,
    **train_fold_kwargs,
):
    """Full LOSO loop (9 folds, one held-out subject each) ->
    train_fold_biciv2a per fold -> aggregate.

    device: defaults to CUDA if available, else CPU -- resolved and
    printed here explicitly, same reasoning as
    training.cross_validation.run_cross_validation (a GPU silently
    going unused cost the ADHD project 6 weeks before it was noticed;
    not repeating that here).
    """
    import pandas as pd

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run_loso_biciv2a] device used for all folds: {device}")
    train_fold_kwargs["device"] = device

    available_subjects = set(dataset_by_subject.keys())
    folds = leave_one_subject_out([sid for sid in subject_ids if sid in available_subjects])

    rows = []
    for fold_idx, fold in enumerate(folds):
        train_ids = fold["train_ids"]
        val_ids = fold["val_ids"]
        if not train_ids or not val_ids:
            continue  # this fold has no usable subjects yet - expected on partial data

        result = train_fold_biciv2a(
            train_ids, val_ids, dataset_by_subject, labels_by_subject,
            n_epochs=n_epochs, seed=seed, **train_fold_kwargs,
        )
        r: EvalResult = result["eval_result"]
        rows.append({
            "fold": fold_idx, "held_out_subject": val_ids[0],
            "n_train_subjects": len(train_ids),
            "n_val_trials": len(dataset_by_subject[val_ids[0]]),
            "accuracy": r.accuracy, "balanced_accuracy": r.balanced_accuracy,
            "f1": r.f1, "f1_class0": r.f1_class0, "f1_class1": r.f1_class1,
            "sensitivity": r.sensitivity, "specificity": r.specificity, "auc": r.auc,
            "omega_collapsed": result["final_omega_collapse"].is_collapsed,
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise ValueError("run_loso_biciv2a: no fold had usable subjects.")

    numeric_cols = [c for c in df.columns if c not in ("fold", "held_out_subject", "omega_collapsed")]
    mean_row = df[numeric_cols].mean()
    mean_row["fold"] = "MEAN"
    std_row = df[numeric_cols].std()
    std_row["fold"] = "STD"
    return pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
