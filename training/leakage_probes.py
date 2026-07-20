"""
grn_balladeer.training.leakage_probes
=========================================
Diagnostic tool, not part of the main training/eval path: checks
whether the model's learned representations (omega, or pooled h)
encode SEX information, as a proxy for whether the model might be
partly detecting sex rather than (or in addition to) ADHD.

Motivated by real literature on this exact risk in EEG-based disease
classifiers:
- Sex is robustly decodable from EEG by ML models (65-81% balanced
  accuracy across TUEG/TUAB/NMT datasets - Truong et al./Nature Sci Rep
  2025; ~81% in van Putten et al.-style CNN classifiers on 142 patients,
  Hum Brain Mapp 2023), explicitly flagged in that literature as a
  confound risk for any disease classifier where prevalence differs by
  sex - exactly the ADHD case (~3:1 male:female in epidemiology; 69.3%
  male in ADHD vs 56.0% male in Control in this project's own 138-subject
  cohort, per data.labels.report_confounds).
- ADHD-specific: the EEG signature of ADHD itself may differ by sex
  (widespread theta enhancement in ADHD boys vs frontally-localized
  theta enhancement in ADHD girls - Ellis et al.-style EEG+EDA findings),
  so this probe should not be read as "sex leakage = bad, eliminate it"
  without nuance - see this module's docstring note on interpretation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


def check_sex_leakage(
    embeddings: np.ndarray, sex_labels: np.ndarray, cv_folds: int = 5, seed: int = 42
) -> dict:
    """Fits a simple logistic regression to predict `sex_labels` (binary,
    0/1) from `embeddings` (n_samples, n_features - e.g. pooled omega or
    split_real_imag(h) per subject), under stratified cross-validation.

    Returns {'mean_cv_accuracy', 'std_cv_accuracy', 'chance_accuracy',
    'n_samples', 'above_chance'}. `chance_accuracy` is the majority-class
    baseline (not 0.5, unless classes are balanced) - always compare
    mean_cv_accuracy against THIS, not against 0.5.

    INTERPRETATION NOTE: a high score here does not automatically mean
    "the model is biased and must be fixed" - given real evidence that
    ADHD's EEG signature itself differs by sex, some sex-correlated
    signal in the embeddings could be legitimate diagnostic information,
    not pure confound. Use this as a flag to look closer (e.g. via
    training.evaluate's by-sex disaggregation) and to report explicitly
    in the paper, not as an automatic red flag demanding removal.

    n_samples/cv_folds too small (as with the current 4-subject toy
    dataset) makes this diagnostic UNRELIABLE - see the caveat in this
    session's Notion note. Intended for the real 138-subject cohort.
    """
    n_samples = len(sex_labels)
    if n_samples < cv_folds * 2:
        raise ValueError(
            f"check_sex_leakage: only {n_samples} samples for {cv_folds}-fold CV - "
            "need at least 2 per fold. Result would not be meaningful; increase "
            "cv_folds down or provide more subjects."
        )

    chance_accuracy = max(np.mean(sex_labels), 1 - np.mean(sex_labels))

    clf = LogisticRegression(max_iter=1000)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, embeddings, sex_labels, cv=skf, scoring="accuracy")

    return {
        "mean_cv_accuracy": float(scores.mean()),
        "std_cv_accuracy": float(scores.std()),
        "chance_accuracy": float(chance_accuracy),
        "n_samples": n_samples,
        "above_chance": bool(scores.mean() > chance_accuracy + 2 * scores.std()),
    }


def check_subject_identity_leakage(
    embeddings: np.ndarray, subject_ids: np.ndarray, cv_folds: int = 5, seed: int = 42
) -> dict:
    """Fits a multi-class logistic regression to predict WHICH subject
    each epoch-level embedding came from, under standard (non-subject-
    disjoint) stratified k-fold -- i.e. epochs from the same subject can
    appear in both train and test folds here, unlike the model's real
    training protocol. This is intentional: the question this probe
    answers is "how much of the discriminative signal in these
    embeddings is subject identity, as opposed to class-relevant EEG
    state" -- not a re-test of generalization to unseen subjects (that's
    what the real subject-disjoint CV already measures).

    MOTIVATION (added this session): the first full 114-subject 5-fold
    CV showed non-trivial in-sample (train-subject) AUC gains (0.546->
    0.610 across training) but chance-level held-out AUC (0.489±0.042)
    -- consistent with, though not proof of, the model latching onto
    per-subject idiosyncrasies (impedance, individual EEG baseline)
    that happen to correlate with class within a given training split,
    rather than a genuinely generalizable ADHD signal. A HIGH subject-
    identity recoverability here would support that interpretation
    directly. Reuses check_sex_leakage's structure/interpretation
    pattern above (see its own docstring for the sex-specific nuance,
    which doesn't apply here -- there is no legitimate reason embeddings
    SHOULD encode subject identity, unlike sex).

    Returns {'mean_cv_accuracy', 'std_cv_accuracy', 'chance_accuracy',
    'n_samples', 'n_subjects', 'above_chance'}. chance_accuracy here is
    the majority-class (most-epochs-per-subject) baseline, generalizing
    check_sex_leakage's binary chance calculation to the multi-class
    case.
    """
    n_samples = len(subject_ids)
    unique_subjects, counts = np.unique(subject_ids, return_counts=True)
    n_subjects = len(unique_subjects)

    if n_samples < cv_folds * n_subjects:
        raise ValueError(
            f"check_subject_identity_leakage: {n_samples} samples, {n_subjects} subjects, "
            f"{cv_folds}-fold CV needs at least {cv_folds} samples per subject on average - "
            "not satisfied here. Reduce cv_folds or provide more epochs per subject."
        )

    chance_accuracy = counts.max() / n_samples

    clf = LogisticRegression(max_iter=1000)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, embeddings, subject_ids, cv=skf, scoring="accuracy")

    return {
        "mean_cv_accuracy": float(scores.mean()),
        "std_cv_accuracy": float(scores.std()),
        "chance_accuracy": float(chance_accuracy),
        "n_samples": n_samples,
        "n_subjects": n_subjects,
        "above_chance": bool(scores.mean() > chance_accuracy + 2 * scores.std()),
    }
