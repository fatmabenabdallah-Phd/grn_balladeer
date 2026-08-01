"""
grn_balladeer.data.labels_biciv2a
====================================
Subject-disjoint split helper for BCI IV 2a, parallel in spirit to
data.labels.stratified_subject_kfold but deliberately simpler: with
only 9 subjects and no demographic metadata distributed with this
dataset (unlike BALLADEER's sex/age_bin), stratifying by anything
beyond the subject itself is not meaningful here. Leave-one-subject-out
(LOSO, 9 folds) is used instead of k-fold, since 9 subjects split into
k<9 folds would leave very few subjects per validation fold anyway --
LOSO uses every subject as its own held-out test exactly once, the
strictest and most standard subject-disjoint protocol for a cohort
this small.

Non-negotiable rule carried over from the ADHD project: split at the
SUBJECT level, never at the epoch/trial level.
"""

from __future__ import annotations

from typing import Dict, List

SUBJECT_IDS: List[str] = [f"A0{i}T" for i in range(1, 10)]


def leave_one_subject_out(subject_ids: List[str] = SUBJECT_IDS) -> List[Dict[str, List[str]]]:
    """Returns a list of {train_ids, val_ids} dicts, one per subject --
    val_ids always contains exactly one subject, train_ids the
    remaining n-1. Same dict shape as
    data.labels.stratified_subject_kfold's fold output, so
    training.cross_validation.run_cross_validation-style consumers do
    not need a different fold format for this dataset.
    """
    folds = []
    for held_out in subject_ids:
        folds.append({
            "train_ids": [sid for sid in subject_ids if sid != held_out],
            "val_ids": [held_out],
        })
    return folds
