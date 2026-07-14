"""
grn_balladeer.preprocessing.epoching
=======================================
Module 2b (part 2, step 5) — epoching locked to flag spawn events
(stimulus-locked, the standard EEG convention — not response-locked to
the TAGS reaction).

flag_spawn_time (from slackline_flags_info.json) is in the same time
domain as TAGS' general_time (see event_alignment.py docstring for the
empirical proof: general_time - reaction_time matches flag_spawn_time
within ~0.34s mean error over 76 real events). This module reuses that
same domain directly on flag_spawn_time, giving stimulus-locked epochs
without needing the TAGS reaction data at all — only slackline_flags_info.json
and the same session_start_general_time=0.0 assumption documented in
event_alignment.py.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import mne
import numpy as np


def flags_to_samples(
    level_flags: List[dict], sfreq: float, session_start_general_time: float = 0.0
) -> Tuple[np.ndarray, List[str]]:
    """Converts a level's flag list (as found under
    slackline_levels_flags_info[i]['flags'] in slackline_flags_info.json)
    into (sample_indices, flag_type_labels), using the same alignment
    logic and assumption as align_events_to_eeg.

    Each flag dict is expected to have 'flag_type' (str) and
    'flag_spawn_time' (seconds, session-relative).
    """
    spawn_times = np.array([f["flag_spawn_time"] for f in level_flags], dtype=float)
    flag_types = [f["flag_type"] for f in level_flags]

    elapsed = spawn_times - session_start_general_time
    sample_indices = np.round(elapsed * sfreq).astype(int)

    if (sample_indices < 0).any():
        raise ValueError(
            "Some flag_spawn_time values map to a negative sample index — "
            "session_start_general_time is likely wrong for this file."
        )

    return sample_indices, flag_types


def epoch_by_flag_events(
    raw: mne.io.Raw,
    flag_sample_indices: np.ndarray,
    flag_types: List[str],
    tmin: float = -0.2,
    tmax: float = 1.0,
    baseline: Tuple[float, float] = (None, 0),
) -> mne.Epochs:
    """Builds stimulus-locked epochs around each flag spawn sample.
    Events falling outside the Raw's available sample range are dropped
    (not an error — expected when working with a truncated recording
    excerpt, as with the partial UB0136 CGX files used during
    development). Returns an mne.Epochs object with event_id mapping
    each distinct flag_type string to an integer code.

    tmin/tmax default to -0.2s/+1.0s: covers the pre-stimulus baseline
    plus the window in which nearly all real reaction times observed in
    the UB0136 TAGS file fall (median ~0.85s, max ~6.3s for the single
    'non_focusable'/incorrect outlier) — WORKING DEFAULT, not yet tuned
    against the full task design; revisit once more subjects are available.
    """
    n_samples = raw.get_data().shape[1]
    valid_mask = (flag_sample_indices >= 0) & (flag_sample_indices < n_samples)
    n_dropped = int((~valid_mask).sum())

    kept_indices = flag_sample_indices[valid_mask]
    kept_types = [ft for ft, keep in zip(flag_types, valid_mask) if keep]

    unique_types = sorted(set(kept_types))
    event_id = {t: i + 1 for i, t in enumerate(unique_types)}

    events = np.zeros((len(kept_indices), 3), dtype=int)
    events[:, 0] = kept_indices
    events[:, 2] = [event_id[t] for t in kept_types]

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        verbose=False,
    )

    if n_dropped:
        print(f"epoch_by_flag_events: dropped {n_dropped} flag(s) outside the available Raw range.")

    return epochs
