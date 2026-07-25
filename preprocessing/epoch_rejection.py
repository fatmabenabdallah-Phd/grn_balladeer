"""
grn_balladeer.preprocessing.epoch_rejection
===============================================
Epoch-level amplitude rejection (peak-to-peak threshold) -- a standard
EEG quality-control step never implemented anywhere in this pipeline
before this session. Individual epochs containing a large-amplitude
transient (movement, muscle burst, electrode pop) are rejected outright
rather than averaged in, complementing (not replacing) the channel-
level cleaning in preprocessing/bad_channels.py: a channel can be
consistently bad across a whole recording (handled by bad_channels.py),
while an epoch can be transiently corrupted on an otherwise-good
channel (handled here).

Default threshold (150 microvolts peak-to-peak) follows common EEG
artifact-rejection practice (e.g. the widely-used manual/autoreject
convention for scalp EEG), not independently derived from this
dataset -- documented explicitly so it is not mistaken for a
data-driven choice.
"""

from __future__ import annotations

from typing import List, Tuple

import mne
import numpy as np


def compute_peak_to_peak(epochs_data: np.ndarray) -> np.ndarray:
    """epochs_data: (n_epochs, n_channels, n_samples). Returns
    (n_epochs,) -- for each epoch, the MAXIMUM peak-to-peak amplitude
    across all its channels (the standard convention: one bad channel
    within an otherwise-good epoch is enough to flag it, since a single
    corrupted channel still corrupts that channel's own feature for
    this epoch)."""
    per_channel_p2p = epochs_data.max(axis=2) - epochs_data.min(axis=2)  # (n_epochs, n_channels)
    return per_channel_p2p.max(axis=1)  # (n_epochs,)


def reject_bad_epochs(
    epochs: mne.Epochs, threshold: float = 150e-6
) -> Tuple[mne.Epochs, dict]:
    """Drops epochs whose peak-to-peak amplitude (on any channel)
    exceeds `threshold` (volts; default 150 microvolts, standard EEG
    convention -- see module docstring). Returns (filtered_epochs,
    report), where report = {"n_total": ..., "n_rejected": ...,
    "rejection_rate": ..., "threshold_uv": ...} for transparency/
    auditing -- never silently drop epochs without a recoverable
    reason.

    Does not modify `epochs` in place (mne.Epochs indexing returns a
    new object).
    """
    data = epochs.get_data()  # (n_epochs, n_channels, n_samples)
    p2p = compute_peak_to_peak(data)
    keep_mask = p2p <= threshold

    n_total = len(epochs)
    n_rejected = int((~keep_mask).sum())
    report = {
        "n_total": n_total,
        "n_rejected": n_rejected,
        "rejection_rate": n_rejected / n_total if n_total > 0 else 0.0,
        "threshold_uv": threshold * 1e6,
    }

    filtered_epochs = epochs[keep_mask]
    return filtered_epochs, report
