"""
grn_balladeer.preprocessing.mne_loading
=========================================
Module 2b (part 1) — loading raw CGX / Emotiv EPOCX files into mne.Raw
objects, using device specs empirically confirmed on subject UB0004
(both real files inspected, 2026-07-10):

CGX  (UB0004_EEG_CGX_2023_06_08T15_35_53.csv):
  - 30 EEG channels + ExG 1/2 + ACC32/33/34 + Packet Counter + TRIGGER
  - 'timestamps' column in seconds, first row is the real header (no
    metadata line)
  - real sampling rate via linear regression on timestamps: ~500.07 Hz
    (matches the assumed 500 Hz)
  - Packet Counter wraps 0-127 (8-bit cyclic counter): a jump of -127 is
    a normal wraparound, NOT a dropped packet. Any other jump size would
    indicate real data loss.

Emotiv EPOCX (UB0004_EPOCX_..._md_pm_bp.csv):
  - FIRST LINE IS METADATA, not data (e.g. "title:..., sampling
    rate:eeg_128;mot_32;pm_0.1;pow_8, samples:40048..."). Real column
    header is on line 2 -> pd.read_csv(..., skiprows=1) is required.
  - 14 EEG channels under 'EEG.<name>' columns (AF3, F7, F3, FC5, T7, P7,
    O1, O2, P8, T8, FC6, F4, F8, AF4) — confirmed present and matching
    EMOTIV_CHANNELS exactly.
  - Per-channel quality columns present as 'CQ.<name>' (15 cols) and
    'EQ.<name>' (16 cols).
  - Real sampling rate via regression on 'Timestamp' column: ~128.07 Hz
    for this subject — already at the target rate, no resampling needed
    for this particular file (other subjects may still require it if
    their native rate differs; keep resample_emotiv_to_128hz available).
"""

from __future__ import annotations

from typing import List, Tuple

import mne
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Confirmed channel names (Module 2a) — do not guess other spellings.
# ---------------------------------------------------------------------------

CGX_CHANNELS: List[str] = [
    "AF7", "Fpz", "F7", "Fz", "T7", "FC6", "Fp1", "F4", "C4", "Oz",
    "CP6", "Cz", "PO8", "CP5", "O2", "O1", "P3", "P4", "P7", "P8",
    "Pz", "PO7", "T8", "C3", "Fp2", "F3", "F8", "FC5", "AF8", "A2",
]

EMOTIV_CHANNELS: List[str] = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2",
    "P8", "T8", "FC6", "F4", "F8", "AF4",
]

CGX_ASSUMED_SFREQ = 500.0
EMOTIV_TARGET_SFREQ = 128.0


def estimate_sampling_rate(timestamps_seconds: np.ndarray) -> float:
    """Linear regression of sample index against timestamp (seconds) ->
    slope is the sampling period. More robust than a naive mean-diff
    because it is not thrown off by a few large jitter outliers.
    Confirmed on real data: CGX ~500.07 Hz, Emotiv EPOCX ~128.07 Hz."""
    idx = np.arange(len(timestamps_seconds))
    slope, _ = np.polyfit(idx, timestamps_seconds, 1)
    return 1.0 / slope


def check_cgx_packet_integrity(packet_counter: np.ndarray, wrap_at: int = 128) -> dict:
    """Verifies that no CGX packet was actually dropped. The hardware
    Packet Counter wraps cyclically (0..wrap_at-1); a diff of
    -(wrap_at-1) is a normal wraparound, not data loss. Any other
    non-+1 diff indicates a real gap and should be investigated before
    epoching.

    Returns a dict with 'n_wraparounds', 'n_suspicious_jumps', and
    'suspicious_indices' (row indices right before a suspicious jump).
    """
    diffs = np.diff(packet_counter)
    wraparound = -(wrap_at - 1)
    is_wraparound = diffs == wraparound
    is_normal_step = diffs == 1
    suspicious = ~(is_wraparound | is_normal_step)
    return {
        "n_wraparounds": int(is_wraparound.sum()),
        "n_suspicious_jumps": int(suspicious.sum()),
        "suspicious_indices": np.where(suspicious)[0].tolist(),
    }


def load_eeg_cgx(filepath: str, include_exg_as_eog: bool = True) -> mne.io.Raw:
    """Loads a raw CGX csv export into an mne.Raw object with a standard
    10-20 montage restricted to the channels actually present. Runs an
    integrity check on the Packet Counter and raises if real packet loss
    is detected (not just the normal 0-127 wraparound).

    If include_exg_as_eog=True (default), the two auxiliary 'ExG 1'/'ExG 2'
    channels are also loaded and typed as 'eog' — this is a WORKING
    ASSUMPTION (the roadmap notes ExG can serve as an EOG/ECG reference,
    exact electrode placement not yet confirmed) — used by
    run_ica_artifact_removal for automatic blink-component detection.
    """
    df = pd.read_csv(filepath)

    integrity = check_cgx_packet_integrity(df["Packet Counter"].values)
    if integrity["n_suspicious_jumps"] > 0:
        raise ValueError(
            f"CGX file {filepath}: {integrity['n_suspicious_jumps']} suspicious "
            f"Packet Counter jump(s) detected (not a normal wraparound) at row "
            f"indices {integrity['suspicious_indices'][:10]}... — investigate "
            "before proceeding, do not silently epoch through a real data gap."
        )

    sfreq = estimate_sampling_rate(df["timestamps"].values)

    eeg_cols = [f"{ch}(uV)" for ch in CGX_CHANNELS]
    missing = [c for c in eeg_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CGX file {filepath} is missing expected channel columns: {missing}")

    ch_names = list(CGX_CHANNELS)
    ch_types = ["eeg"] * len(CGX_CHANNELS)
    cols = list(eeg_cols)

    exg_cols = [c for c in ["ExG 1(uV)", "ExG 2(uV)"] if c in df.columns]
    if include_exg_as_eog and exg_cols:
        for c in exg_cols:
            ch_names.append(c.replace("(uV)", ""))
            ch_types.append("eog")
            cols.append(c)

    data_uv = df[cols].to_numpy().T  # (n_channels, n_samples), in microvolts
    data_v = data_uv * 1e-6  # MNE expects volts

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data_v, info, verbose=False)

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="warn", verbose=False)

    return raw


def load_eeg_emotiv(filepath: str, target_sfreq: float = EMOTIV_TARGET_SFREQ) -> mne.io.Raw:
    """Loads a raw Emotiv EPOCX csv export. IMPORTANT: the real column
    header is on line 2, not line 1 (line 1 is a metadata line) —
    skiprows=1 is mandatory or parsing silently misaligns columns.
    Resamples to target_sfreq only if the real rate (estimated by
    regression on the 'Timestamp' column) differs from it."""
    df = pd.read_csv(filepath, skiprows=1, encoding="utf-8-sig")

    eeg_cols = [f"EEG.{ch}" for ch in EMOTIV_CHANNELS]
    missing = [c for c in eeg_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Emotiv file {filepath} is missing expected channel columns: {missing}")

    sfreq = estimate_sampling_rate(df["Timestamp"].values)

    data_uv = df[eeg_cols].to_numpy().T
    data_v = data_uv * 1e-6

    info = mne.create_info(ch_names=EMOTIV_CHANNELS, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data_v, info, verbose=False)

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="warn", verbose=False)

    if abs(sfreq - target_sfreq) > 0.5:
        raw.resample(target_sfreq, verbose=False)

    return raw


def get_common_channels(cgx_channels: List[str] = CGX_CHANNELS, emotiv_channels: List[str] = EMOTIV_CHANNELS) -> List[str]:
    """Exact intersection — determines the usable subset for the
    cross-device generalization test (Slackline/CGX -> Robots/Emotiv)."""
    return [ch for ch in cgx_channels if ch in emotiv_channels]
