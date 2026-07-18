"""
grn_balladeer/training/eda_features.py
=======================================
Extract auxiliary EDA/biometric feature vectors from the
BALLADEER EmbracePlus CSV (balladeer_embraceplus_data.csv, sep=';').

VALIDATED on real data (UB0136, 2026-07-18):
  - All 4 sources (S1/S6/S11/Robots) produce shape-(6,) float32 vectors
    with no NaNs for UB0136.
  - Wearing-threshold gate returns None correctly.
  - Only UB0136 of our current 4 real subjects (UB0004/UB0022/UB0023/
    UB0136) is present in the EmbracePlus file — the other 3 are absent.
    The dual-branch fusion can only be tested on UB0136 until more
    EmbracePlus data becomes available. All results on this single subject
    must be labeled "mechanics validation only, not science."

COLUMN-NAME GOTCHA (real names differ from the README description):
  - README says: 'S1_eda_first_two_mean'
  - Real file has: 'S1_eda_values_first_two_mean'
  - Wearing column: 'S1_wearing_detection_mean_percentage'
  - Use the constants below, not the README, when addressing columns.

COVERAGE (across 103 subjects):
  - EDA, pulse_rate, temperature, wearing_detection: 95% valid → use these
  - PRV: 30% valid → excluded
  - Respiratory rate: 50% valid → excluded
"""

import numpy as np
import pandas as pd
from typing import Optional

# Activity sources with reliable first/last/middle split data.
# Cognifit excluded: its first_two/last_two columns are NaN even when present.
EDA_FEATURE_SOURCES = ["S1", "S6", "S11", "Robots"]

# Feature vector length (fixed, must match AuxBranchEncoder input_dim)
EDA_FEATURE_DIM = 6


def extract_eda_features(
    embrace_row: pd.Series,
    source: str = "S1",
    wearing_threshold: float = 80.0,
) -> Optional[np.ndarray]:
    """
    Extract a fixed-length auxiliary feature vector from one EmbracePlus row.

    Features (6 total, all from >=95%-coverage columns):
        0  delta_eda        : EDA last_two_mean - first_two_mean (reactivity)
        1  mid_eda          : EDA middle_mean (mean during task period)
        2  eda_slope        : linear slope across the ~9 per-minute EDA values
        3  eda_std          : std across the ~9 per-minute EDA values
        4  pulse_rate_mean  : mean HR in bpm
        5  temperature_mean : mean skin temperature in Celsius

    Parameters
    ----------
    embrace_row       : one row from balladeer_embraceplus_data.csv (sep=';')
    source            : activity source prefix — 'S1' | 'S6' | 'S11' | 'Robots'
    wearing_threshold : if wearing_detection_mean_percentage < this, return None
                        (bracelet not reliably worn — treat as missing data)

    Returns
    -------
    np.ndarray of shape (6,) dtype float32, or None if session is invalid.
    NaN values (at most 2 allowed) are imputed with 0.0 — the caller is
    responsible for z-scoring per fold before feeding into AuxBranchEncoder.
    """
    # ── Quality gate: bracelet must be worn reliably ─────────────────────
    wear_col = f"{source}_wearing_detection_mean_percentage"
    wear_pct = embrace_row.get(wear_col, np.nan)
    if pd.isna(wear_pct) or float(wear_pct) < wearing_threshold:
        return None

    # ── EDA reactivity (task-induced change) ─────────────────────────────
    first = embrace_row.get(f"{source}_eda_values_first_two_mean", np.nan)
    last  = embrace_row.get(f"{source}_eda_values_last_two_mean",  np.nan)
    mid   = embrace_row.get(f"{source}_eda_values_middle_mean",    np.nan)
    delta_eda = (float(last) - float(first)
                 if not (pd.isna(first) or pd.isna(last))
                 else np.nan)

    # ── EDA time-series slope and variability ────────────────────────────
    n_vals = int(embrace_row.get(f"{source}_eda_values_count", 0))
    vals = [
        float(embrace_row[f"{source}_eda_values_{i}"])
        for i in range(n_vals)
        if not pd.isna(embrace_row.get(f"{source}_eda_values_{i}", np.nan))
    ]
    if len(vals) >= 2:
        x = np.arange(len(vals), dtype=np.float64)
        eda_slope = float(np.polyfit(x, vals, 1)[0])
        eda_std   = float(np.std(vals))
    else:
        eda_slope = np.nan
        eda_std   = np.nan

    # ── Reliable biometrics (>=95% coverage) ─────────────────────────────
    hr   = float(embrace_row.get(f"{source}_pulse_rate_mean_bpm",      np.nan))
    temp = float(embrace_row.get(f"{source}_temperature_mean_celsius", np.nan))

    features = np.array(
        [delta_eda, float(mid) if not pd.isna(mid) else np.nan,
         eda_slope, eda_std, hr, temp],
        dtype=np.float32
    )

    # If more than 2 features are NaN, session is too incomplete to use
    if np.isnan(features).sum() > 2:
        return None

    # Impute remaining NaNs with 0.0 (neutral after z-scoring)
    features = np.where(np.isnan(features), 0.0, features).astype(np.float32)
    return features


def load_embrace_index(csv_path: str) -> pd.DataFrame:
    """
    Load the full EmbracePlus CSV and index by username.
    Use this once at dataset-build time, then look up rows by subject ID.

    Returns
    -------
    pd.DataFrame indexed by 'username' (103 rows x 402 feature columns)
    """
    df = pd.read_csv(csv_path, sep=";")
    df = df.set_index("username")
    return df


def get_eda_features_for_subject(
    subject_id: str,
    embrace_index: pd.DataFrame,
    source: str = "S1",
    wearing_threshold: float = 80.0,
) -> Optional[np.ndarray]:
    """
    Convenience wrapper: look up subject_id in the pre-loaded embrace_index
    and call extract_eda_features. Returns None if subject is absent from
    the EmbracePlus file (e.g. UB0004, UB0022, UB0023 in our current data).

    This is the function to call from the dual-branch DataLoader.
    """
    if subject_id not in embrace_index.index:
        return None
    return extract_eda_features(
        embrace_index.loc[subject_id],
        source=source,
        wearing_threshold=wearing_threshold,
    )
