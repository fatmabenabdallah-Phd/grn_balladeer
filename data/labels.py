"""
grn_balladeer.data.labels
==========================
Module 1 (data and labels) + Module 10's stratified split helper.

Non-negotiable rule: label = `diagnosed` (yes/no), NEVER `group`.
26 subjects with group=Control are actually diagnosed=yes.
"""

from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# Google Colab Pro — Drive mounting and dataset paths
# ---------------------------------------------------------------------------
# Real dataset stored in Google Drive, folder "BALLADEER ADHD DATASET"
# (My Drive), containing: one sub-folder per subject (UBxxxx/),
# balladeer_embraceplus_data.csv, README.md, slackline_flags_info.json,
# users_demographics.json.

DRIVE_DATASET_DIR = "/content/drive/MyDrive/BALLADEER ADHD DATASET"

PATH_DEMOGRAPHICS = os.path.join(DRIVE_DATASET_DIR, "users_demographics.json")
PATH_EMBRACEPLUS = os.path.join(DRIVE_DATASET_DIR, "balladeer_embraceplus_data.csv")
PATH_SLACKLINE_FLAGS = os.path.join(DRIVE_DATASET_DIR, "slackline_flags_info.json")


def mount_drive_colab() -> None:
    """Call this first on Colab Pro. Mounts Google Drive and checks that
    the BALLADEER ADHD DATASET folder is actually reachable."""
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")
    if not os.path.isdir(DRIVE_DATASET_DIR):
        raise FileNotFoundError(
            f"Folder not found: {DRIVE_DATASET_DIR}. Check that "
            "'BALLADEER ADHD DATASET' is at the root of My Drive, "
            "or adjust DRIVE_DATASET_DIR."
        )


def subject_dir(user_id: str) -> str:
    """Path to a subject's sub-folder, e.g. subject_dir('UB0004') ->
    .../BALLADEER ADHD DATASET/UB0004"""
    return os.path.join(DRIVE_DATASET_DIR, user_id)


# ---------------------------------------------------------------------------
# Confirmed encoding of categorical fields in users_demographics.json
# (empirically verified on the 158 real records on 2026-07-08:
# gender==1 -> 69.3% among diagnosed='yes' and 56.0% among diagnosed='no',
# which exactly reproduces the previously documented male percentages).
# DO NOT guess a different mapping.
# ---------------------------------------------------------------------------

GENDER_MAP: Dict[int, str] = {1: "male", 2: "female"}
GROUP_MAP: Dict[int, str] = {1: "Experimental", 2: "Control"}


def load_demographics(path: str = PATH_DEMOGRAPHICS) -> pd.DataFrame:
    """Loads users_demographics.json (defaults to Google Drive path).
    Expected columns, confirmed on the 158 real records: user, group,
    gender, age, diagnosed (no other field, no continuous severity score)."""
    return pd.read_json(path)


def check_group_diagnosed_consistency(demo_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulation group x diagnosed — must show the 26 subjects with
    group=Control / diagnosed=yes (verified: exactly 26)."""
    return pd.crosstab(demo_df["group"], demo_df["diagnosed"], margins=True)


def build_label_table(demo_df: pd.DataFrame) -> pd.DataFrame:
    """Filters diagnosed in {'yes','no'}, EXCLUDES 'undetermined'.
    Label = diagnosed, never group.

    Expected output (verified): 88 ADHD (diagnosed='yes'),
    50 controls (diagnosed='no'), i.e. 138 usable subjects.
    """
    df = demo_df[demo_df["diagnosed"].isin(["yes", "no"])].copy()
    df["label"] = (df["diagnosed"] == "yes").astype(int)  # 1 = ADHD, 0 = control
    df["age_bin"] = pd.cut(
        df["age"], bins=[5, 9, 12, 15, 19], labels=["6-9", "10-12", "13-15", "16-18"]
    )
    # gender/group are encoded 1/2 in the raw JSON. Mapping empirically
    # confirmed (see GENDER_MAP/GROUP_MAP): gender==1 -> male.
    df["sex"] = df["gender"].map(GENDER_MAP)
    if df["sex"].isna().any():
        raise ValueError(
            "Value(s) of 'gender' outside GENDER_MAP {1,2} — check the "
            "actual encoding before proceeding."
        )
    out = df.rename(columns={"user": "user_id"})
    return out[["user_id", "label", "sex", "age", "age_bin"]]


def report_confounds(label_df: pd.DataFrame) -> dict:
    """% male and mean age per class — basis for k-fold stratification.
    Verified: ADHD 69.3% male / mean age 11.47;
    Control 56.0% male / mean age 13.74."""
    report = {}
    for label_value, name in [(1, "ADHD"), (0, "Control")]:
        sub = label_df[label_df["label"] == label_value]
        pct_male = (sub["sex"] == "male").mean() * 100
        report[name] = {
            "n": len(sub),
            "pct_male": round(pct_male, 1),
            "mean_age": round(sub["age"].mean(), 2),
        }
    return report


def stratified_subject_kfold(
    label_df: pd.DataFrame, k: int = 5, seed: int = 42
) -> List[dict]:
    """Split at the SUBJECT level, never at the epoch level. Stratified on
    label + sex + age bin simultaneously, via a composite key.

    Returns a list of {train_ids, val_ids} dicts of length k.
    """
    df = label_df.reset_index(drop=True).copy()
    df["strata"] = (
        df["label"].astype(str) + "_" + df["sex"].astype(str) + "_" + df["age_bin"].astype(str)
    )

    # Some strata may be too rare for StratifiedKFold(k) — fall back to the
    # "label"-only stratum for these subjects, to be documented.
    counts = df["strata"].value_counts()
    rare_strata = counts[counts < k].index
    df.loc[df["strata"].isin(rare_strata), "strata"] = "rare_" + df.loc[
        df["strata"].isin(rare_strata), "label"
    ].astype(str)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in skf.split(df, df["strata"]):
        folds.append(
            {
                "train_ids": df.loc[train_idx, "user_id"].tolist(),
                "val_ids": df.loc[val_idx, "user_id"].tolist(),
            }
        )
    return folds
