"""
data/epoching.py
================
Cuts CGX EEG recordings into epochs locked on Slackline events, after
synchronization via sync.py.

Conventions:
  - An epoch = a [tmin, tmax] second window around a flag's appearance.
  - Only events where reacted=True AND correct=True are included by default.
  - flagType=-1 is excluded (undocumented code, see sync.py).
  - Normalization is intra-subject/intra-session (per-channel z-score over
    the session) because BALLADEER has NO resting-state recording.

Author: GRN-BALLADEER project
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field

from grn_balladeer.data.sync import unix_ms_to_eeg_idx, CGX_SFREQ

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Data structure: one annotated epoch
# ---------------------------------------------------------------------------

@dataclass
class Epoch:
    """An EEG segment with its metadata."""
    data:        np.ndarray          # shape [n_samples_epoch, n_channels]
    label:       int                 # 0 = control, 1 = ADHD
    subject_id:  str
    session_id:  str
    flag_type:   int                 # 0/1/2/3
    flag_time_s: float               # generalTime (s since game start)
    reacted:     bool
    correct:     bool
    focus:       str                 # 'Target' | 'non_focusable'
    reaction_time_s: float


# ---------------------------------------------------------------------------
# 2. Per-channel preprocessing
# ---------------------------------------------------------------------------

def bandpass_filter_np(
    data: np.ndarray,
    sfreq: float,
    l_freq: float,
    h_freq: float
) -> np.ndarray:
    """
    FIR bandpass filter via numpy/scipy. Used before PLV computation.
    For full MNE preprocessing, prefer load_eeg_cgx -> mne.Raw instead.
    """
    from scipy.signal import firwin, filtfilt
    n_taps = int(sfreq) + 1  # filter duration = 1 second
    n_taps = n_taps if n_taps % 2 == 1 else n_taps + 1
    coeffs = firwin(n_taps, [l_freq, h_freq],
                    pass_zero=False, fs=sfreq)
    return filtfilt(coeffs, [1.0], data, axis=0)


def notch_filter_np(
    data: np.ndarray,
    sfreq: float,
    freq: float = 50.0
) -> np.ndarray:
    """Notch filter at freq Hz (mains noise, default 50 Hz for EU/Africa/Spain)."""
    from scipy.signal import iirnotch, filtfilt
    b, a = iirnotch(freq, Q=30.0, fs=sfreq)
    return filtfilt(b, a, data, axis=0)


def zscore_normalize_session(data: np.ndarray) -> np.ndarray:
    """
    PER-CHANNEL z-score normalization over the whole session.
    Mandatory since there is no resting-state in BALLADEER to normalize
    across sessions. Applied BEFORE cutting into epochs.
    """
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    std[std == 0] = 1.0  # avoid division by zero on flat channels
    return (data - mean) / std


def detect_motion_artifacts(
    eeg_data: np.ndarray,
    accel_data: Optional[np.ndarray],
    epoch_indices: np.ndarray,
    n_samples_epoch: int,
    accel_threshold_mg: float = 200.0
) -> np.ndarray:
    """
    Boolean mask: True = epoch contaminated by motion.
    Uses the CGX accelerometer columns (available in the file).
    Used for the real robustness test in Phase 4 (high/low motion
    amplitude stratification).

    Parameters
    ----------
    accel_data         : [n_samples, 3] — X,Y,Z axes in mg. None = disabled.
    epoch_indices      : start indices of the epochs in the EEG signal.
    accel_threshold_mg : maximum acceptable amplitude threshold.
    """
    if accel_data is None:
        return np.zeros(len(epoch_indices), dtype=bool)

    accel_norm = np.linalg.norm(accel_data, axis=1)  # overall magnitude
    masks = []
    for idx in epoch_indices:
        end = min(idx + n_samples_epoch, len(accel_norm))
        segment = accel_norm[idx:end]
        masks.append(segment.max() > accel_threshold_mg)
    return np.array(masks, dtype=bool)


# ---------------------------------------------------------------------------
# 3. Main cutting logic
# ---------------------------------------------------------------------------

def cut_epochs(
    eeg_times:     np.ndarray,
    eeg_data:      np.ndarray,
    tags_df:       pd.DataFrame,
    subject_id:    str,
    session_id:    str,
    label:         int,
    offset_ms:     float,
    tmin:          float = -0.5,
    tmax:          float = 2.0,
    sfreq:         float = CGX_SFREQ,
    include_only_correct: bool = True,
    exclude_flag_types:   Optional[List[int]] = None,
    accel_data:    Optional[np.ndarray] = None
) -> Tuple[List[Epoch], dict]:
    """
    Cuts the EEG signal into epochs locked on TAGS events.

    Parameters
    ----------
    tmin / tmax           : window in seconds around the flag (default: -0.5s to +2.0s).
    include_only_correct : if True, excludes incorrect trials AND non-responses.
    exclude_flag_types   : list of flagTypes to exclude (e.g. [-1] for the unknown code).
    accel_data           : accelerometer data [n_samples, 3] to detect artifacts.

    Returns
    -------
    epochs : list of Epoch objects
    stats  : dict of cutting statistics (for logs/reporting)
    """
    if exclude_flag_types is None:
        exclude_flag_types = [-1]

    n_before = int(abs(tmin) * sfreq)
    n_after  = int(tmax * sfreq)
    n_epoch  = n_before + n_after

    epochs = []
    stats = {
        'total_events':    len(tags_df),
        'excluded_flag':   0,
        'excluded_correct': 0,
        'excluded_boundary': 0,
        'excluded_motion': 0,
        'kept':            0,
    }

    # Filter events
    working = tags_df.copy()

    # Exclude unwanted flagTypes
    mask_flag = working['flagType'].isin(exclude_flag_types)
    stats['excluded_flag'] = int(mask_flag.sum())
    working = working[~mask_flag]

    # Exclude incorrect trials if requested
    if include_only_correct:
        mask_bad = ~(working['reacted'] & working['correct'])
        stats['excluded_correct'] = int(mask_bad.sum())
        working = working[~mask_bad]

    # Convert TAGS timestamps to EEG indices
    event_unix_ms = working['timestamp_ms'].values
    # We use generalTime (game anchor) rather than reaction timestamp
    # because generalTime marks the flag's APPEARANCE, not the response
    # -> recompute the index from the EEG time corresponding to generalTime
    general_times_s = working['generalTime'].values
    # Convert generalTime (session-relative) -> Unix timestamp ms
    general_times_unix_ms = general_times_s * 1000.0 + offset_ms
    event_indices = unix_ms_to_eeg_idx(general_times_unix_ms, eeg_times, offset_ms)

    # Detect motion artifacts
    motion_mask = detect_motion_artifacts(
        eeg_data, accel_data, event_indices, n_epoch
    )

    for k, (idx, row) in enumerate(zip(event_indices, working.itertuples())):
        start = idx - n_before
        end   = idx + n_after

        # Outside the signal boundaries
        if start < 0 or end > len(eeg_data):
            stats['excluded_boundary'] += 1
            continue

        # Motion artifact
        if motion_mask[k]:
            stats['excluded_motion'] += 1
            continue

        segment = eeg_data[start:end, :]  # [n_epoch, n_channels]

        epoch = Epoch(
            data=segment,
            label=label,
            subject_id=subject_id,
            session_id=session_id,
            flag_type=int(row.flagType),
            flag_time_s=float(row.generalTime),
            reacted=bool(row.reacted),
            correct=bool(row.correct),
            focus=str(row.focus),
            reaction_time_s=float(row.reactionTime)
                if not np.isnan(row.reactionTime) else np.nan,
        )
        epochs.append(epoch)
        stats['kept'] += 1

    logger.info(
        "Subject %s session %s: %d/%d epochs kept "
        "(excl. flag=%d, correct=%d, boundary=%d, motion=%d)",
        subject_id, session_id,
        stats['kept'], stats['total_events'],
        stats['excluded_flag'], stats['excluded_correct'],
        stats['excluded_boundary'], stats['excluded_motion'],
    )

    return epochs, stats


# ---------------------------------------------------------------------------
# 4. Full preprocessing of a session
# ---------------------------------------------------------------------------

def preprocess_and_epoch_session(
    session: dict,
    label: int,
    subject_id: str,
    tmin: float = -0.5,
    tmax: float = 2.0,
    apply_notch: bool = True,
    apply_normalize: bool = True
) -> Tuple[List[Epoch], dict]:
    """
    Chains notch + z-score normalization + epoch cutting on an already
    synchronized session (output of sync.sync_session).

    The per-band bandpass is done DOWNSTREAM in the connectivity/ module,
    which needs the wide-band signal to compute per-band PLV. Here we only
    apply a wide [1-45 Hz] filter to remove low-frequency artifacts and
    aliasing.
    """
    eeg_times = session['eeg_times']
    eeg_data  = session['eeg_data'].copy()   # [n_samples, n_channels]
    tags_df   = session['tags_df']
    offset_ms = session['offset_ms']

    # Mains notch filter (50 Hz, Spain)
    if apply_notch:
        eeg_data = notch_filter_np(eeg_data, sfreq=CGX_SFREQ, freq=50.0)

    # Wide-band filter [1-45 Hz]
    eeg_data = bandpass_filter_np(eeg_data, sfreq=CGX_SFREQ,
                                   l_freq=1.0, h_freq=45.0)

    # Intra-session normalization
    if apply_normalize:
        eeg_data = zscore_normalize_session(eeg_data)

    # Extract accelerometer data if available in the original EEG file
    # ('Accel X', 'Accel Y', 'Accel Z' fields — see README)
    accel_data = session.get('accel_data', None)

    epochs, stats = cut_epochs(
        eeg_times=eeg_times,
        eeg_data=eeg_data,
        tags_df=tags_df,
        subject_id=subject_id,
        session_id=str(session.get('level', 'unknown')),
        label=label,
        offset_ms=offset_ms,
        tmin=tmin,
        tmax=tmax,
        sfreq=CGX_SFREQ,
        include_only_correct=True,
        exclude_flag_types=[-1],
        accel_data=accel_data,
    )

    return epochs, stats


# ---------------------------------------------------------------------------
# 5. Building the full-subject dataset
# ---------------------------------------------------------------------------

def build_subject_epoch_array(
    epochs: List[Epoch]
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """
    Converts a list of Epoch objects into numpy arrays ready for the GRN.

    Returns
    -------
    X     : [n_epochs, n_channels, n_samples] — standard PyTorch conv format
    y     : [n_epochs] — labels (0/1)
    meta  : list of dicts with subject_id, flag_type, reaction_time_s, focus
    """
    X = np.stack([e.data.T for e in epochs], axis=0)  # transpose -> [ch, time]
    y = np.array([e.label for e in epochs], dtype=np.int64)
    meta = [
        {
            'subject_id':     e.subject_id,
            'flag_type':      e.flag_type,
            'reaction_time_s': e.reaction_time_s,
            'focus':          e.focus,
        }
        for e in epochs
    ]
    return X, y, meta
