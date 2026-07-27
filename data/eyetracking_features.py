"""
grn_balladeer.data.eyetracking_features
=========================================
Extracts a fixed-size feature vector from BALLADEER's raw eye-tracking
recordings (session 1, AttentionRobotsDesktop task), for fusion with
EEG in the same style as the existing EDA/behavioral auxiliary branch.

Raw data format (verified against real subject files this session):
one CSV per subject with columns [timeChecked, weight, looked_col,
looked_row] -- gaze mapped to a discrete grid, sampled at ~60Hz
(timeChecked step ~0.0167s). `weight` was observed constant (100)
across all rows of every inspected file and is not used as a feature.

CRITICAL FINDING this session: grid size (max looked_col/looked_row)
varies substantially across the 112-subject cohort (col_max ranging
9-47, row_max 1-14), most likely reflecting different screen/window
resolutions across acquisition setups rather than a fixed canonical
grid. Raw gaze-position dispersion/entropy would therefore NOT be
comparable across subjects without normalization -- this module
normalizes looked_col/looked_row to [0,1] using each subject's own
observed max in their file before computing any dispersion or entropy
feature. This assumes the observed max reflects that subject's actual
screen/grid extent (the gaze position that was actually reached),
which may itself be a mild underestimate if a subject never looked at
the true edge of their screen -- a limitation of this normalization
worth flagging rather than assuming away.

Also discovered this session: at least one subject (UB0110) has a
severely truncated recording (14 samples, 0.22s) that is almost
certainly corrupted/incomplete rather than a genuine short session --
extract_eyetracking_features raises ValueError for recordings below
MIN_DURATION_S so callers can detect and exclude these rather than
silently including a non-representative feature vector.

Design rationale for the 6 features chosen (matching the existing
EDA/behavioral branches' dimensionality, not a specific literature
citation -- this is a new, exploratory modality for this project):
gaze dispersion (spatial spread, both axes, now grid-normalized), gaze-
shift rate (proxy for saccade frequency), spatial entropy (scattered
vs. concentrated attention, now grid-normalized), and longest stable
run (proxy for sustained fixation/attention duration) -- all plausibly
relevant to ADHD's attentional and impulsivity profile, though this is
a hypothesis this module makes testable, not a validated finding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_DURATION_S = 10.0  # recordings shorter than this are treated as corrupted/incomplete
N_ENTROPY_BINS_PER_AXIS = 20  # spatial bins per axis for the normalized entropy grid

EYETRACKING_FEATURE_NAMES = [
    "gaze_col_std_norm",
    "gaze_row_std_norm",
    "gaze_shift_rate",
    "gaze_spatial_entropy_norm",
    "longest_stable_run_s",
    "recording_duration_s",
]


def extract_eyetracking_features(et_df: pd.DataFrame) -> np.ndarray:
    """Computes a 6-dimensional feature vector from one subject's raw
    eye-tracking DataFrame (columns: timeChecked, weight, looked_col,
    looked_row). Gaze position is normalized to [0,1] per axis using
    this subject's own observed max looked_col/looked_row, since grid
    size is not consistent across the cohort (see module docstring).

    Raises ValueError if the recording is shorter than MIN_DURATION_S
    (likely corrupted/incomplete, e.g. the truncated 0.22s recording
    found for one subject this session) or has fewer than 2 samples.

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

    t = et_df["timeChecked"].to_numpy()
    duration_s = float(t[-1] - t[0])
    if duration_s < MIN_DURATION_S:
        raise ValueError(
            f"extract_eyetracking_features: recording duration {duration_s:.2f}s "
            f"is below MIN_DURATION_S={MIN_DURATION_S}s -- likely corrupted/incomplete "
            f"(e.g. UB0110's 0.22s/14-sample recording found this session)."
        )

    cols_raw = et_df["looked_col"].to_numpy().astype(np.float64)
    rows_raw = et_df["looked_row"].to_numpy().astype(np.float64)

    # Normalize to [0,1] per axis using THIS subject's own observed grid
    # extent -- grid size varies across the cohort (verified this
    # session: col_max 9-47, row_max 1-14), so raw pixel/cell counts are
    # not comparable across subjects without this step.
    col_max = cols_raw.max()
    row_max = rows_raw.max()
    cols_norm = cols_raw / col_max if col_max > 0 else cols_raw
    rows_norm = rows_raw / row_max if row_max > 0 else rows_raw

    # 1-2. Spatial dispersion (normalized): how spread out gaze positions
    # are, relative to this subject's own grid extent.
    gaze_col_std_norm = float(np.std(cols_norm))
    gaze_row_std_norm = float(np.std(rows_norm))

    # 3. Gaze-shift rate: fraction of consecutive samples where gaze grid
    # position changed (either axis), per second -- a coarse proxy for
    # saccade/attention-switching frequency, since exact saccade
    # detection would need continuous coordinates, not this grid.
    # Computed on the RAW (not normalized) grid, since a position change
    # is a change regardless of grid scale.
    position_changed = (np.diff(cols_raw) != 0) | (np.diff(rows_raw) != 0)
    gaze_shift_rate = float(position_changed.sum()) / duration_s

    # 4. Spatial entropy (normalized): bins the normalized [0,1]x[0,1]
    # gaze positions into a fixed N_ENTROPY_BINS_PER_AXIS x
    # N_ENTROPY_BINS_PER_AXIS grid (same bin count for every subject,
    # regardless of their own raw grid resolution) before computing
    # Shannon entropy of cell-occupancy -- this makes the entropy value
    # comparable across subjects, unlike entropy computed directly on
    # each subject's own differently-sized raw grid.
    col_bins = np.clip((cols_norm * N_ENTROPY_BINS_PER_AXIS).astype(int), 0, N_ENTROPY_BINS_PER_AXIS - 1)
    row_bins = np.clip((rows_norm * N_ENTROPY_BINS_PER_AXIS).astype(int), 0, N_ENTROPY_BINS_PER_AXIS - 1)
    binned_positions = list(zip(col_bins.tolist(), row_bins.tolist()))
    _, counts = np.unique(binned_positions, axis=0, return_counts=True)
    probs = counts / counts.sum()
    gaze_spatial_entropy_norm = float(-np.sum(probs * np.log2(probs + 1e-12)))

    # 5. Longest stable run: longest consecutive stretch (in seconds)
    # gaze stayed on the exact same raw grid cell -- a proxy for
    # sustained fixation/attention duration.
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
        gaze_col_std_norm, gaze_row_std_norm, gaze_shift_rate,
        gaze_spatial_entropy_norm, longest_stable_run_s, recording_duration_s,
    ], dtype=np.float32)
