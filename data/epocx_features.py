"""
grn_balladeer.data.epocx_features
====================================
Loads Emotiv EPOCX recordings (BALLADEER session 1, AttentionRobotsDesktop
task) and extracts band-power features for a Random Forest classifier,
replacing an earlier N=35 cross-task validation result that could not
be reconstructed or verified this session (no saved subject list or
extraction code survived; re-deriving the filtering criterion from
scratch consistently gave ~97-100 subjects at any reasonable threshold,
not 35 -- rather than present an unverifiable number, this module
rebuilds the analysis transparently from raw files).

File format (verified against a real subject's file this session):
one CSV per subject, whose FIRST line is a metadata header (title,
timestamps, headset info -- not column names) and whose SECOND line
contains the real column headers (Timestamp, EEG.<channel>,
CQ.<channel>, etc.). Must be loaded with skiprows=1, not as a normal
CSV.

Per-channel contact quality (CQ.<channel>) is on a 0-4 scale (verified
this session: values observed are exactly {0,1,2,3,4}, not the 0-100
scale CQ.Overall might suggest by name -- CQ.Overall was found to be a
DIFFERENT, session-summary-like metric that rarely exceeds ~75 even in
a good recording, and is NOT the per-channel quality signal to filter
on). A subject/session is accepted if, for at least
MIN_SESSION_QUALITY_FRACTION of the recording's samples, at least
MIN_CHANNEL_QUALITY_FRACTION of the 14 channels have CQ >= MIN_CQ.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import signal
from scipy.integrate import trapezoid

EPOCX_CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2",
    "P8", "T8", "FC6", "F4", "F8", "AF4",
]
EPOCX_SFREQ = 128.0

MIN_CQ = 3
MIN_CHANNEL_QUALITY_FRACTION = 0.75
MIN_SESSION_QUALITY_FRACTION = 0.75

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

EPOCX_FEATURE_NAMES = (
    [f"{ch}_{band}" for ch in EPOCX_CHANNELS for band in BANDS]
    + ["theta_beta_ratio_mean"]
)


def load_epocx_recording(csv_path: str) -> pd.DataFrame:
    """Loads one EPOCX CSV, correctly skipping the metadata header line
    (verified this session: line 0 is metadata like 'title:UB0021,
    start timestamp:...', real column headers are on line 1)."""
    return pd.read_csv(csv_path, skiprows=1, low_memory=False)


def session_passes_quality_filter(eeg_df: pd.DataFrame) -> bool:
    """Checks whether a recording meets the quality bar: at
    MIN_SESSION_QUALITY_FRACTION of samples, at least
    MIN_CHANNEL_QUALITY_FRACTION of the 14 channels have per-channel
    CQ >= MIN_CQ (verified 0-4 scale this session, not 0-100)."""
    cq_cols = [f"CQ.{ch}" for ch in EPOCX_CHANNELS]
    if not all(c in eeg_df.columns for c in cq_cols):
        return False
    good_contact_fraction_per_sample = (eeg_df[cq_cols] >= MIN_CQ).mean(axis=1)
    pct_time_good = (good_contact_fraction_per_sample >= MIN_CHANNEL_QUALITY_FRACTION).mean()
    return bool(pct_time_good >= MIN_SESSION_QUALITY_FRACTION)


def extract_epocx_band_power_features(eeg_df: pd.DataFrame) -> np.ndarray:
    """Computes band-power features (Welch's method) for each of the 14
    EPOCX channels across 5 bands (delta/theta/alpha/beta/gamma), plus
    a theta/beta ratio averaged across channels -- the same style of
    feature engineering already used for the Random Forest baseline on
    BALLADEER and Nasrabadi, for a directly comparable methodology.

    Returns a (71,) float32 array: 14 channels x 5 bands + 1 ratio,
    order matching EPOCX_FEATURE_NAMES.
    """
    eeg_cols = [f"EEG.{ch}" for ch in EPOCX_CHANNELS]
    if not all(c in eeg_df.columns for c in eeg_cols):
        raise ValueError(
            f"extract_epocx_band_power_features: missing expected EEG columns, "
            f"have {[c for c in eeg_cols if c not in eeg_df.columns]} missing"
        )

    features = []
    theta_beta_ratios = []
    for ch in EPOCX_CHANNELS:
        raw = eeg_df[f"EEG.{ch}"].to_numpy(dtype=np.float64)
        raw = raw[~np.isnan(raw)]
        if len(raw) < int(EPOCX_SFREQ * 2):
            raise ValueError(
                f"extract_epocx_band_power_features: channel {ch} has only "
                f"{len(raw)} valid samples, need at least {int(EPOCX_SFREQ*2)} "
                f"for a reliable Welch estimate."
            )
        freqs, psd = signal.welch(raw, fs=EPOCX_SFREQ, nperseg=min(len(raw), int(EPOCX_SFREQ * 2)))

        band_powers = {}
        for band_name, (fmin, fmax) in BANDS.items():
            mask = (freqs >= fmin) & (freqs < fmax)
            band_powers[band_name] = float(trapezoid(psd[mask], freqs[mask])) if mask.any() else 0.0
            features.append(band_powers[band_name])

        if band_powers["beta"] > 0:
            theta_beta_ratios.append(band_powers["theta"] / band_powers["beta"])

    theta_beta_ratio_mean = float(np.mean(theta_beta_ratios)) if theta_beta_ratios else 0.0
    features.append(theta_beta_ratio_mean)

    return np.array(features, dtype=np.float32)
