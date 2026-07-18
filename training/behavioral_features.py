"""
grn_balladeer/training/behavioral_features.py
==============================================
Extract fixed-length behavioral feature vectors from parsed TAGS events.

VALIDATED on real data (2026-07-18):
  - UB0136 (ADHD, M, 14y, 76 events): [1.22, 0.34, 1.00, 0.00, 0.053, 0.76]
  - UB0004 (Control, F, 11y, 29 events): [0.84, 0.20, 1.00, 0.00, 0.172, 1.00]
  Both produce clean float32 vectors with no NaNs.

ADHD-literature grounding for these features:
  - RT mean + std: ADHD associated with slower and more variable reaction
    times (Lijffijt et al. 2005 meta-analysis, d=0.68 for RT variability).
  - Commission rate (flagType=-1 events): impulsivity proxy.
  - Focus ratio (gaze on target at response time): attentional engagement.

IMPORTANT — flagType=-1 handling:
  These events are NOT documented in the BALLADEER README but appear in
  real data. They always have correct=False and appear to be commission
  errors (response to a non-target). They are EXCLUDED from RT/accuracy
  computation but counted separately as commission_rate.
"""

import ast
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# Fixed output dimension — must match AuxBranchEncoder behavioral_input_dim
BEHAVIORAL_FEATURE_DIM = 6


def parse_tags_file(tags_path: str) -> pd.DataFrame:
    """
    Load and parse a raw TAGS CSV file.

    The 'value' column is a Python dict literal (single-quoted) —
    ast.literal_eval is required, NOT json.loads.

    Returns
    -------
    pd.DataFrame with columns:
        timestamp_ms, reacted, reactionTime, correct,
        flagType, generalTime, focus
    """
    df = pd.read_csv(tags_path)
    records = []
    for _, row in df.iterrows():
        try:
            v = ast.literal_eval(row["value"])
            r = v["reactionOrOmission"][0]
        except (KeyError, IndexError, ValueError, SyntaxError) as e:
            continue
        records.append({
            "timestamp_ms":  float(row["timestamp"]),
            "reacted":       r["reacted"] == "True",
            "reactionTime":  float(r.get("reactionTime", np.nan)),
            "correct":       r["correct"] == "True",
            "flagType":      int(r["flagType"][0]),
            "generalTime":   float(r["generalTime"]),
            "focus":         r["focus"],
        })
    return pd.DataFrame(records)


def extract_behavioral_features(
    tags_df: pd.DataFrame,
    min_valid_events: int = 3,
) -> Optional[np.ndarray]:
    """
    Compute a 6-dimensional behavioral feature vector from parsed TAGS events.

    Features:
        0  rt_mean         : mean reaction time (s) over valid, reacted trials
        1  rt_std          : std of reaction time (intra-individual variability)
        2  accuracy        : fraction correct over valid trials (flagType != -1)
        3  omission_rate   : fraction of valid trials with no response
        4  commission_rate : fraction of all events with flagType=-1
        5  focus_ratio     : fraction of reacted trials where gaze was on target

    Parameters
    ----------
    tags_df          : output of parse_tags_file()
    min_valid_events : minimum number of flagType!=-1 events required;
                       returns None if fewer (session too short/broken)

    Returns
    -------
    np.ndarray shape (6,) dtype float32, or None if session is invalid.
    rt_std is 0.0 when only one reacted trial exists (not NaN).
    All values imputed to 0.0 if NaN (caller must z-score per fold).
    """
    if tags_df is None or len(tags_df) == 0:
        return None

    # Valid trials: exclude undocumented flagType=-1 commission-error events
    valid = tags_df[tags_df["flagType"] != -1].copy()
    if len(valid) < min_valid_events:
        return None

    # ── Reaction time ────────────────────────────────────────────────────
    reacted = valid[valid["reacted"]]
    rt = reacted["reactionTime"].dropna().values

    if len(rt) == 0:
        rt_mean = np.nan
        rt_std  = np.nan
    elif len(rt) == 1:
        rt_mean = float(rt[0])
        rt_std  = 0.0          # std undefined for n=1; use 0.0 (not NaN)
    else:
        rt_mean = float(np.mean(rt))
        rt_std  = float(np.std(rt, ddof=0))

    # ── Accuracy and omission ────────────────────────────────────────────
    accuracy      = float(valid["correct"].mean())
    omission_rate = float((~valid["reacted"]).mean())

    # ── Commission rate (flagType=-1 events / all events) ────────────────
    commission_rate = float((tags_df["flagType"] == -1).sum()) / len(tags_df)

    # ── Visual attention (focus on target at response time) ──────────────
    if len(reacted) == 0:
        focus_ratio = np.nan
    else:
        focus_ratio = float((reacted["focus"] == "Target").mean())

    features = np.array(
        [rt_mean, rt_std, accuracy, omission_rate, commission_rate, focus_ratio],
        dtype=np.float32,
    )

    # If more than 2 features are NaN, session too broken to use
    if np.isnan(features).sum() > 2:
        return None

    # Impute remaining NaNs with 0.0 (neutral after z-scoring)
    features = np.where(np.isnan(features), 0.0, features).astype(np.float32)
    return features


def get_behavioral_features_for_subject(
    tags_path: str,
    min_valid_events: int = 3,
) -> Optional[np.ndarray]:
    """
    Convenience wrapper: parse TAGS file and extract features in one call.
    Returns None if the file doesn't exist or the session is invalid.
    """
    p = Path(tags_path)
    if not p.exists():
        return None
    tags_df = parse_tags_file(str(p))
    return extract_behavioral_features(tags_df, min_valid_events)
