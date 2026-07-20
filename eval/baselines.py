"""
grn_balladeer.eval.baselines
=============================
Module 10 (baselines) — provides a working evaluation harness
(SVM, RF, theta/beta ratio) BEFORE starting on the GRN itself.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
    precision_recall_fscore_support, confusion_matrix, roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from grn_balladeer.data.labels import stratified_subject_kfold

# ---------------------------------------------------------------------------
# Real confirmed channels (UB0004 file, Module 2a). Do not guess other
# names — use inspect_cgx_header / inspect_emotiv_header for any new
# dataset (see configs/ for cross-dataset adaptation).
# ---------------------------------------------------------------------------

CGX_CHANNELS = [
    "AF7", "Fpz", "F7", "Fz", "T7", "FC6", "Fp1", "F4", "C4", "Oz",
    "CP6", "Cz", "PO8", "CP5", "O2", "O1", "P3", "P4", "P7", "P8",
    "Pz", "PO7", "T8", "C3", "Fp2", "F3", "F8", "FC5", "AF8", "A2",
]

EMOTIV_CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2",
    "P8", "T8", "FC6", "F4", "F8", "AF4",
]

COMMON_CHANNELS = [
    "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8",
]

EEG_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def set_seed(seed: int = 42) -> None:
    """Fixes numpy/random seeds. Add torch.manual_seed(seed) once torch
    is introduced (Module 4+)."""
    random.seed(seed)
    np.random.seed(seed)


def extract_band_power_features(
    epochs, channels: List[str] = None, sfreq: float = 500.0
) -> np.ndarray:
    """Extracts per-band power (delta..gamma) per channel + global
    theta/beta ratio, from an mne.Epochs object (or a np.ndarray of shape
    (n_epochs, n_channels, n_samples)).

    Returns a np.ndarray (n_epochs, n_channels * n_bands + 1).
    """
    if hasattr(epochs, "get_data"):
        data = epochs.get_data()
        sfreq = epochs.info["sfreq"]
    else:
        data = epochs

    n_epochs, n_channels, _ = data.shape
    n_bands = len(EEG_BANDS)
    features = np.zeros((n_epochs, n_channels * n_bands))

    theta_power_all = np.zeros((n_epochs, n_channels))
    beta_power_all = np.zeros((n_epochs, n_channels))

    for ep in range(n_epochs):
        for ch in range(n_channels):
            freqs, psd = welch(data[ep, ch, :], fs=sfreq, nperseg=min(256, data.shape[-1]))
            for b_idx, (band_name, (lo, hi)) in enumerate(EEG_BANDS.items()):
                mask = (freqs >= lo) & (freqs <= hi)
                band_power = np.trapz(psd[mask], freqs[mask]) if mask.any() else 0.0
                features[ep, ch * n_bands + b_idx] = band_power
                if band_name == "theta":
                    theta_power_all[ep, ch] = band_power
                elif band_name == "beta":
                    beta_power_all[ep, ch] = band_power

    theta_beta_ratio = (theta_power_all.mean(axis=1) + 1e-12) / (
        beta_power_all.mean(axis=1) + 1e-12
    )
    return np.hstack([features, theta_beta_ratio.reshape(-1, 1)])


def aggregate_epochs_to_subject(
    features: np.ndarray, subject_ids: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """Averages features per subject."""
    df = pd.DataFrame(features)
    df["subject_id"] = subject_ids
    grouped = df.groupby("subject_id").mean()
    return grouped.values, grouped.index.tolist()


def train_svm_baseline(features: np.ndarray, labels: np.ndarray) -> Tuple[SVC, StandardScaler]:
    """SVM (RBF) — standardization required before fitting."""
    scaler = StandardScaler().fit(features)
    clf = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
    clf.fit(scaler.transform(features), labels)
    return clf, scaler


def train_rf_baseline(features: np.ndarray, labels: np.ndarray) -> RandomForestClassifier:
    """Random Forest — no standardization needed."""
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, class_weight="balanced", random_state=42, n_jobs=-1
    )
    clf.fit(features, labels)
    return clf


@dataclass
class EvalResult:
    accuracy: float
    f1: float
    auc: float
    n: int
    # EXTENDED this session - the original 3 metrics alone can hide a model
    # that mostly predicts the majority class, given the real ~64%/36%
    # ADHD/Control imbalance in the 138-subject cohort. balanced_accuracy
    # averages per-class recall (not fooled by imbalance); f1_class0/1 and
    # sensitivity/specificity give the clinically-relevant asymmetry
    # (missing a real ADHD case is not the same cost as a false positive).
    balanced_accuracy: float
    f1_class0: float
    f1_class1: float
    sensitivity: float  # recall on class 1 (ADHD, per data.labels' label=1 convention)
    specificity: float  # recall on class 0 (Control)


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """Finds the decision threshold maximizing Youden's J statistic
    (sensitivity + specificity - 1) on the ROC curve, rather than the
    default fixed 0.5 cutoff.

    Added this session to address a real observed pattern: a model can
    have a genuinely informative AUC (e.g. 0.68-0.77 on the EDA-real-only
    dual-branch subset) while STILL always predicting the majority class
    at threshold=0.5 (sensitivity=1.0, specificity=0.0) when classes are
    imbalanced (~77%/23% ADHD/Control on that subset) -- the model IS
    ranking correctly, the cutoff is just miscalibrated for this
    imbalance.

    CRITICAL CORRECT USAGE -- calibrate the threshold on TRAINING data's
    own predicted probabilities (or a held-out slice of train), THEN
    apply that fixed threshold to validation predictions. Calling this
    function on validation labels/probabilities and using the result to
    re-score performance on those SAME validation samples is a form of
    leakage (choosing the cutoff to fit the exact data you're evaluating
    on) and will overstate performance -- only use this on train data
    when the goal is an honest, reportable validation metric.

    Returns {'threshold', 'sensitivity_at_threshold', 'specificity_at_threshold',
    'youden_j'}.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return {
        "threshold": float(thresholds[best_idx]),
        "sensitivity_at_threshold": float(tpr[best_idx]),
        "specificity_at_threshold": float(1 - fpr[best_idx]),
        "youden_j": float(j_scores[best_idx]),
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> EvalResult:
    """Accuracy, F1, AUC, balanced accuracy, per-class F1, sensitivity/
    specificity. Call globally, then per sub-group via
    evaluate_disaggregated(). Assumes binary labels {0, 1} with 1=ADHD,
    matching data.labels.build_label_table's convention - sensitivity/
    specificity naming would be wrong for a different label convention."""
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (float("nan"),) * 4
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return EvalResult(
        accuracy=accuracy_score(y_true, y_pred),
        f1=f1_score(y_true, y_pred),
        auc=roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else float("nan"),
        n=len(y_true),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        f1_class0=float(f1_per_class[0]),
        f1_class1=float(f1_per_class[1]),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
    )


def evaluate_disaggregated(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, group: np.ndarray
) -> Dict[str, EvalResult]:
    """Repeats evaluate() per sub-group (sex, age bin)."""
    out = {"global": evaluate(y_true, y_pred, y_proba)}
    for g in np.unique(group):
        mask = group == g
        if mask.sum() > 1 and len(set(y_true[mask])) > 1:
            out[str(g)] = evaluate(y_true[mask], y_pred[mask], y_proba[mask])
    return out


def run_baseline_cv(
    label_df: pd.DataFrame,
    subject_features: np.ndarray,
    subject_ids: List[str],
    model: str = "svm",
    k: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Full loop: stratified subject split -> train -> eval -> aggregate
    the k folds. model in {'svm', 'rf'}."""
    set_seed(seed)
    id_to_idx = {sid: i for i, sid in enumerate(subject_ids)}
    folds = stratified_subject_kfold(label_df, k=k, seed=seed)

    rows = []
    for fold_idx, fold in enumerate(folds):
        train_ids = [sid for sid in fold["train_ids"] if sid in id_to_idx]
        val_ids = [sid for sid in fold["val_ids"] if sid in id_to_idx]

        train_idx = [id_to_idx[s] for s in train_ids]
        val_idx = [id_to_idx[s] for s in val_ids]

        label_map = dict(zip(label_df["user_id"], label_df["label"]))
        y_train = np.array([label_map[s] for s in train_ids])
        y_val = np.array([label_map[s] for s in val_ids])

        X_train = subject_features[train_idx]
        X_val = subject_features[val_idx]

        if model == "svm":
            clf, scaler = train_svm_baseline(X_train, y_train)
            X_val_t = scaler.transform(X_val)
            y_pred = clf.predict(X_val_t)
            y_proba = clf.predict_proba(X_val_t)[:, 1]
        elif model == "rf":
            clf = train_rf_baseline(X_train, y_train)
            y_pred = clf.predict(X_val)
            y_proba = clf.predict_proba(X_val)[:, 1]
        else:
            raise ValueError("model must be 'svm' or 'rf'")

        res = evaluate(y_val, y_pred, y_proba)
        rows.append(
            {"fold": fold_idx, "model": model, "accuracy": res.accuracy, "f1": res.f1, "auc": res.auc, "n_val": res.n}
        )

    return pd.DataFrame(rows)
