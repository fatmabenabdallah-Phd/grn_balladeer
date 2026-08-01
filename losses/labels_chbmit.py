"""
grn_balladeer.data.labels_chbmit
====================================
Subject-disjoint split helper for CHB-MIT. The one non-obvious rule
here, verified against PhysioNet's own case description (not assumed):
chb01 and chb21 are the SAME subject (chb21 recorded 1.5 years later)
-- treating them as 2 independent subjects would leak subject identity
across a LOSO fold's train/val boundary, exactly the kind of leakage
this project's standing rule (split at the subject level, never
epoch/file level) exists to prevent.

chb24 is excluded from SUBJECT_IDS below: it was added to the
collection after SUBJECT-INFO was written and has no confirmed
demographic record (PhysioNet's own page notes this explicitly) --
not excluded for a data-quality reason like chb12's non-conforming
files, but flagged separately since a future decision to include it
should be a deliberate choice, not an oversight.
"""

from __future__ import annotations

from typing import Dict, List

# 22 unique subjects. chb01 and chb21 merged into a single subject_id
# ('chb01') via CASE_TO_SUBJECT below -- every other case maps to itself.
SUBJECT_IDS: List[str] = [
    "chb01", "chb02", "chb03", "chb04", "chb05", "chb06", "chb07", "chb08",
    "chb09", "chb10", "chb11", "chb12", "chb13", "chb14", "chb15", "chb16",
    "chb17", "chb18", "chb19", "chb20", "chb22", "chb23",
]

CASE_TO_SUBJECT: Dict[str, str] = {case: case for case in SUBJECT_IDS}
CASE_TO_SUBJECT["chb21"] = "chb01"  # same subject as chb01, 1.5 years later


def leave_one_subject_out(subject_ids: List[str] = SUBJECT_IDS) -> List[Dict[str, List[str]]]:
    """Returns a list of {train_ids, val_ids} dicts, one per subject.
    Same shape as data.labels_biciv2a.leave_one_subject_out -- consumers
    written for that dataset's LOSO loop need no format changes here.
    """
    folds = []
    for held_out in subject_ids:
        folds.append({
            "train_ids": [sid for sid in subject_ids if sid != held_out],
            "val_ids": [held_out],
        })
    return folds
