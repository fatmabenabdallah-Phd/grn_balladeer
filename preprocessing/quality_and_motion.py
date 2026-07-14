"""
grn_balladeer.preprocessing.quality_and_motion
==================================================
Module 2b (part 2, steps 6-7) — Emotiv channel quality mask and CGX
motion amplitude extraction.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def compute_channel_quality_mask(
    df: pd.DataFrame, channels: List[str], cq_threshold: float = 4.0
) -> np.ndarray:
    """Boolean mask, shape (n_samples, n_channels), True = good contact
    quality. Uses the Emotiv 'CQ.<channel>' columns (0-4 scale, 4=best,
    confirmed on real UB0004 EPOCX data).

    Deliberately does NOT use 'EQ.<channel>' here: empirically, EQ
    columns are NaN for a large fraction of early samples in the real
    UB0004 file (Emotiv only reports EQ intermittently, unlike CQ which
    is populated every sample) — using EQ as a hard filter would
    silently drop usable data. If EQ is needed later (e.g. as a softer,
    continuous-valued feature rather than a hard mask), handle its NaNs
    explicitly rather than reusing this function as-is.

    cq_threshold=4.0 (require perfect contact) is a WORKING DEFAULT — not
    yet tested against a full dataset to check how much data it excludes;
    relax to e.g. >=3.0 if 4.0 turns out too strict once run at scale.
    """
    cols = [f"CQ.{ch}" for ch in channels]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"compute_channel_quality_mask: missing CQ columns: {missing}")

    values = df[cols].to_numpy()
    return values >= cq_threshold


def channel_quality_summary(mask: np.ndarray, channels: List[str]) -> pd.DataFrame:
    """Diagnostic helper: % of samples at good quality, per channel.
    Useful to sanity-check compute_channel_quality_mask's output before
    trusting it in the pipeline."""
    pct_good = mask.mean(axis=0) * 100
    return pd.DataFrame({"channel": channels, "pct_good_quality": pct_good.round(1)})


def extract_motion_amplitude(cgx_csv_path: str) -> Tuple[np.ndarray, float]:
    """Extracts a single motion-amplitude signal from the CGX
    accelerometer channels (ACC32/33/34), as the Euclidean norm of the
    3-axis vector at each sample: sqrt(x^2 + y^2 + z^2).

    CAVEAT (found on real UB0136 data): the column header says '(mg)'
    (milli-g), but real values run in the hundreds of thousands
    (e.g. amplitude mean ~904,000 on a real 200s recording) — far above
    what a calibrated milli-g accelerometer at rest (~1000 mg, gravity)
    would read. Either these are raw/uncalibrated ADC-like counts
    mislabeled as mg, or a scaling factor is missing. Treat the returned
    array as a RELATIVE motion-amplitude proxy (useful for the Module 11
    high-vs-low movement stratification, which only needs relative
    ordering) — do NOT report it as physical mg/g units in the paper
    without first clarifying the true scale with CGX's documentation or
    the acquisition team.

    Returns (amplitude_array, sfreq) — sfreq estimated the same way as
    in mne_loading.estimate_sampling_rate, from the 'timestamps' column.
    """
    df = pd.read_csv(cgx_csv_path)
    acc_cols = ["ACC32(mg)", "ACC33(mg)", "ACC34(mg)"]
    missing = [c for c in acc_cols if c not in df.columns]
    if missing:
        raise ValueError(f"extract_motion_amplitude: missing accelerometer columns: {missing}")

    acc = df[acc_cols].to_numpy()
    amplitude = np.sqrt((acc**2).sum(axis=1))

    idx = np.arange(len(df))
    slope, _ = np.polyfit(idx, df["timestamps"].to_numpy(), 1)
    sfreq = 1.0 / slope

    return amplitude, sfreq
