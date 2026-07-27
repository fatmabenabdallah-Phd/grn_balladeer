"""
grn_balladeer.data.eyetracking_features
=========================================
Extracts a fixed-size feature vector from BALLADEER's raw eye-tracking
recordings (session 1, AttentionRobotsDesktop task), for fusion with
EEG in the same style as the existing EDA/behavioral auxiliary branch.

Raw data format (verified against a real subject's file this session):
one CSV per subject with columns [timeChecked, weight, looked_col,
looked_row] -- gaze mapped to a discrete grid (27 columns x 10 rows,
verified range), sampled at ~60Hz (timeChecked step ~0.0167s). `weight`
was observed constant (100) across all rows of the inspected file and
is not used as a feature; if it varies for some subjects it may encode
a detection-confidence signal worth revisiting.

Design rationale for the 6 features chosen (matching the existing
EDA/behavioral branches' dimensionality, not a specific literature
citation -- this is a new, exploratory modality for this project):
gaze dispersion (spatial spread, both axes), gaze-shift rate (proxy for
saccade frequency), spatial entropy (scattered vs. concentrated
attention), and longest stable run (proxy for sustained fixation/
attention duration) -- all plausibly relevant to ADHD's attentional
and impulsivity profile, though this is a hypothesis this module makes
testable, not a validated finding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


EYETRACKING_FEATURE_NAMES = [
    "gaze_col_std",
    "gaze_row_std",
    "gaze_shift_rate",
    "gaze_spatial_entropy",
    "longest_stable_run_s",
    "recording_duration_s",
]


def extract_eyetracking_features(et_df: pd.DataFrame) -> np.ndarray:
    """Computes a 6-dimensional feature vector from one subject's raw
    eye-tracking DataFrame (columns: timeChecked, weight, looked_col,
    looked_row).

    Returns a (6,) float32 numpy array, order matching
    EYETRACKING_FEATURE_NAMES.
    """
    required_cols = {"timeChecked", "looked_col", "looked_row"}
    if not required_cols.issubset(et_df.columns):
        raise ValueError(
            f"extract_eyetracking_features: expected columns {required_cols}, "
            f"got {set(et_df.columns)}"
        )
    if len(et_df) < 2:
        raise ValueError(
            f"extract_eyetracking_features: need at least 2 samples, got {len(et_df)}"
        )

    cols = et_df["looked_col"].to_numpy()
    rows = et_df["looked_row"].to_numpy()
    t = et_df["timeChecked"].to_numpy()

    # 1-2. Spatial dispersion: how spread out gaze positions are overall
    gaze_col_std = float(np.std(cols))
    gaze_row_std = float(np.std(rows))

    # 3. Gaze-shift rate: fraction of consecutive samples where gaze grid
    # position changed (either axis), per second -- a coarse proxy for
    # saccade/attention-switching frequency, since exact saccade
    # detection would need continuous coordinates, not this grid.
    position_changed = (np.diff(cols) != 0) | (np.diff(rows) != 0)
    duration_s = float(t[-1] - t[0])
    if duration_s <= 0:
        raise ValueError("extract_eyetracking_features: non-positive recording duration")
    gaze_shift_rate = float(position_changed.sum()) / duration_s

    # 4. Spatial entropy: Shannon entropy of the 2D grid-cell occupancy
    # distribution -- low entropy means attention concentrated on a few
    # cells, high entropy means gaze scattered broadly across the grid.
    grid_positions = list(zip(cols.tolist(), rows.tolist()))
    _, counts = np.unique(grid_positions, axis=0, return_counts=True)
    probs = counts / counts.sum()
    gaze_spatial_entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

    # 5. Longest stable run: longest consecutive stretch (in seconds)
    # gaze stayed on the exact same grid cell -- a proxy for sustained
    # fixation/attention duration.
    same_as_prev = np.concatenate([[False], ~position_changed])
    run_lengths = []
    current_run = 1
    for same in same_as_prev[1:]:
        if same:
            current_run += 1
        else:
            run_lengths.append(current_run)
            current_run = 1
    run_lengths.append(current_run)
    median_dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    longest_stable_run_s = float(max(run_lengths) * median_dt) if run_lengths else 0.0

    # 6. Total recording duration (raw session length, not normalized --
    # kept as a feature since session length itself could vary with
    # attentional engagement/task completion speed).
    recording_duration_s = duration_s

    return np.array([
        gaze_col_std, gaze_row_std, gaze_shift_rate,
        gaze_spatial_entropy, longest_stable_run_s, recording_duration_s,
    ], dtype=np.float32)
