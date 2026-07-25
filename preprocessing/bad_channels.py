"""
grn_balladeer.preprocessing.bad_channels
============================================
Bad-channel detection and interpolation for CGX recordings. Motivated
directly by this session's discovery that the ExG reference channels
were silently flat (dead) throughout the project, which went
undetected because no channel-quality check existed anywhere in the
pipeline before this module. If reference channels can silently fail,
ordinary EEG channels plausibly can too (poor scalp contact, movement
artifact, etc.) -- this has never been checked.

Design choice: detect via a robust z-score on log-variance across
channels (median/MAD-based, insensitive to the bad channels
themselves skewing the statistic, unlike a mean/std-based z-score)
rather than a fixed absolute threshold, since raw CGX amplitude scale
varies across subjects/sessions and a fixed threshold would not
transfer. Flagged channels are INTERPOLATED (spherical spline, MNE's
built-in raw.interpolate_bads(), using the same standard_1020
electrode positions already used for the structural graph in
connectivity/structural_graph.py) rather than dropped, so every
subject's feature vector keeps the same fixed dimensionality --
required for classifiers like Random Forest that need consistent
input size across subjects.
"""

from __future__ import annotations

from typing import List

import numpy as np
import mne


def set_standard_montage(raw: mne.io.Raw, ch_names: List[str]) -> mne.io.Raw:
    """Attaches MNE's standard_1020 montage to `raw`'s EEG channels
    (required before interpolate_bads() can compute spherical-spline
    weights from real electrode geometry). Modifies and returns `raw`
    in place. Raises if any ch_names entry isn't in the standard
    montage -- fail loudly rather than silently skip electrode
    positions, consistent with connectivity/structural_graph.py's own
    get_standard_positions().
    """
    montage = mne.channels.make_standard_montage("standard_1020")
    montage_ch_names = set(montage.get_positions()["ch_pos"].keys())
    missing = [ch for ch in ch_names if ch not in montage_ch_names]
    if missing:
        raise KeyError(f"set_standard_montage: {missing} not found in standard_1020 montage.")
    raw.set_montage(montage, on_missing="ignore")  # ignore non-EEG channels (e.g. ExG) not in ch_names
    return raw


def detect_bad_channels(raw: mne.io.Raw, ch_names: List[str], z_thresh: float = 3.0) -> dict:
    """Flags channels whose log-variance is a robust-z-score outlier
    relative to the other channels in `ch_names` (median/MAD-based,
    not mean/std-based, so a handful of genuinely bad channels don't
    themselves distort the reference statistic used to detect them).

    Also explicitly flags flat channels (std == 0) regardless of
    z-score, since a flat channel's log-variance is -inf and would
    otherwise trivially dominate the MAD computation.

    Returns {"bad_channels": [...], "z_scores": {ch: z, ...},
    "log_variances": {ch: log_var, ...}} for full transparency/auditing
    -- never silently exclude a channel without a recoverable reason.
    """
    data = raw.get_data(picks=ch_names)  # (n_channels, n_samples)
    variances = data.var(axis=1)

    flat_mask = variances == 0.0
    flat_channels = [ch_names[i] for i in range(len(ch_names)) if flat_mask[i]]

    # Robust z-score on log-variance, computed only over NON-flat channels
    # (a flat channel's log(0) = -inf would otherwise corrupt the median/MAD).
    non_flat_idx = [i for i in range(len(ch_names)) if not flat_mask[i]]
    log_var_non_flat = np.log(variances[non_flat_idx])
    median_log_var = np.median(log_var_non_flat)
    mad = np.median(np.abs(log_var_non_flat - median_log_var))
    # 1.4826 scales MAD to be a consistent estimator of std under normality
    # (standard robust-z-score convention).
    robust_std = mad * 1.4826 if mad > 0 else 1e-12

    z_scores = {}
    log_variances = {}
    outlier_channels = []
    for i, ch in enumerate(ch_names):
        if flat_mask[i]:
            z_scores[ch] = float("inf")
            log_variances[ch] = float("-inf")
            continue
        log_v = np.log(variances[i])
        z = (log_v - median_log_var) / robust_std
        z_scores[ch] = float(z)
        log_variances[ch] = float(log_v)
        if abs(z) > z_thresh:
            outlier_channels.append(ch)

    bad_channels = sorted(set(flat_channels) | set(outlier_channels))
    return {"bad_channels": bad_channels, "z_scores": z_scores, "log_variances": log_variances}


def interpolate_bad_channels(raw: mne.io.Raw, bad_channels: List[str], ch_names: List[str]) -> mne.io.Raw:
    """Marks `bad_channels` as bad and interpolates them via MNE's
    spherical-spline method, using the montage already attached by
    set_standard_montage(). Requires a montage to already be set.

    ch_names: the full set of channels that have a valid montage
    position (e.g. CGX_CHANNELS) -- any OTHER channel present in `raw`
    (e.g. the ExG reference channels, which have no standard_1020
    position) is explicitly excluded from participating as an
    interpolation reference. Without this, MNE's spherical-spline
    interpolation matrix computation fails with a NaN/inf error, since
    it tries to use every channel's position, including ones that were
    never given a real position.

    Modifies and returns `raw` in place. No-op (returns raw unchanged)
    if bad_channels is empty.
    """
    if not bad_channels:
        return raw
    non_positioned = [ch for ch in raw.ch_names if ch not in ch_names]
    raw.info["bads"] = bad_channels
    raw.interpolate_bads(reset_bads=True, verbose=False, exclude=non_positioned)
    return raw
