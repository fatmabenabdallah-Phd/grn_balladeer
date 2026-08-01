"""
grn_balladeer.eval.run_loso_rf_biciv2a
==========================================
Random Forest band-power baseline, LOSO, on BCI IV 2a -- the same
classical-baseline standard applied to BALLADEER and Nasrabadi (see
eval.baselines), now on the third domain of the architectural-
generality study. Exists to contextualize GRN's LOSO AUC (~0.51,
chance level, with omega_collapsed=True on every fold -- see this
session's diagnostic) against a non-deep-learning reference on the
SAME trials, not a separately-sourced number from another paper.

Uses data.build_dataset_biciv2a.extract_raw_trials_biciv2a (same cue
extraction, same artifact exclusion, same window as
build_subject_dataset_biciv2a) so the RF baseline and GRN are compared
on identical trials -- a divergent trial count here would make the
comparison meaningless without anyone noticing.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from grn_balladeer.data.build_dataset_biciv2a import extract_raw_trials_biciv2a, BICIV2A_CHANNELS, SFREQ_HZ
from grn_balladeer.data.labels_biciv2a import leave_one_subject_out, SUBJECT_IDS
from grn_balladeer.eval.baselines import extract_band_power_features, train_rf_baseline, evaluate


def build_raw_trials_by_subject(
    npz_dir: str, subject_ids: List[str] = SUBJECT_IDS
) -> Dict[str, tuple]:
    """Loads raw trials + labels for every subject once (expensive-ish
    I/O + windowing step), so run_loso_rf_biciv2a does not redo it once
    per fold -- same rationale as train_fold_biciv2a's own
    dataset_by_subject pre-build convention.

    Returns {subject_id: (raw_trials, labels)}.
    """
    out = {}
    for sid in subject_ids:
        trials, labels = extract_raw_trials_biciv2a(f"{npz_dir}/{sid}.npz")
        out[sid] = (trials, labels)
    return out


def run_loso_rf_biciv2a(
    raw_trials_by_subject: Dict[str, tuple],
    subject_ids: List[str] = SUBJECT_IDS,
    sfreq: float = SFREQ_HZ,
) -> pd.DataFrame:
    """Full LOSO loop (9 folds) for the Random Forest band-power
    baseline. Same output shape (DataFrame, per-fold rows + MEAN/STD)
    as run_loso_biciv2a, for direct side-by-side comparison.
    """
    available_subjects = set(raw_trials_by_subject.keys())
    folds = leave_one_subject_out([sid for sid in subject_ids if sid in available_subjects])

    rows = []
    for fold_idx, fold in enumerate(folds):
        train_ids = fold["train_ids"]
        val_ids = fold["val_ids"]
        if not train_ids or not val_ids:
            continue

        train_trials = np.concatenate([raw_trials_by_subject[sid][0] for sid in train_ids], axis=0)
        train_labels = np.concatenate([raw_trials_by_subject[sid][1] for sid in train_ids], axis=0)
        val_sid = val_ids[0]
        val_trials, val_labels = raw_trials_by_subject[val_sid]

        train_features = extract_band_power_features(train_trials, channels=BICIV2A_CHANNELS, sfreq=sfreq)
        val_features = extract_band_power_features(val_trials, channels=BICIV2A_CHANNELS, sfreq=sfreq)

        clf = train_rf_baseline(train_features, train_labels)
        val_pred = clf.predict(val_features)
        val_proba = clf.predict_proba(val_features)[:, 1]

        r = evaluate(val_labels, val_pred, val_proba)
        rows.append({
            "fold": fold_idx, "held_out_subject": val_sid,
            "n_train_trials": len(train_labels), "n_val_trials": len(val_labels),
            "accuracy": r.accuracy, "balanced_accuracy": r.balanced_accuracy,
            "f1": r.f1, "f1_class0": r.f1_class0, "f1_class1": r.f1_class1,
            "sensitivity": r.sensitivity, "specificity": r.specificity, "auc": r.auc,
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise ValueError("run_loso_rf_biciv2a: no fold had usable subjects.")

    numeric_cols = [c for c in df.columns if c not in ("fold", "held_out_subject")]
    mean_row = df[numeric_cols].mean()
    mean_row["fold"] = "MEAN"
    std_row = df[numeric_cols].std()
    std_row["fold"] = "STD"
    return pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
