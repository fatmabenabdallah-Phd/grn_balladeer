"""
grn_balladeer.model.cqt_encoder
==================================
Module 4 — CQT harmonic encoder.

ARCHITECTURAL FINDING (empirically confirmed this session): a CQT down
to fmin=1 Hz at bins_per_octave=12 (Q~16.8) needs ~16.8s of temporal
context to build its lowest-frequency kernel — far longer than a single
stimulus-locked epoch (~1.2s here). Computing the CQT independently per
short epoch either crashes (kernel longer than the signal) or, if forced
via a much smaller filter_scale, collapses to ~2 usable time frames with
degraded frequency selectivity — not usable for a harmonic encoder whose
whole point is frequency resolution.

RESOLUTION: compute the CQT ONCE on the continuous, filtered, per-channel
signal (not per epoch), then pool the relevant time-frequency region for
each event afterward (pool_cqt_to_node_features). This is standard
practice for CQT/wavelet-style transforms with a low fmin — the "epoch"
becomes a pooling window on the continuous spectrogram, not a hard input
boundary.
"""

from __future__ import annotations

from typing import List

import torch
from nnAudio.features.cqt import CQT


def compute_cqt_features(
    channel_signal: torch.Tensor,
    sfreq: float,
    fmin: float = 1.0,
    fmax: float = 45.0,
    bins_per_octave: int = 12,
    hop_length: int = 32,
) -> torch.Tensor:
    """CQT magnitude spectrogram of a full CONTINUOUS single-channel
    signal (not a short epoch — see module docstring). channel_signal:
    1D tensor, (n_samples,). hop_length=32 (not nnAudio's 512 default):
    at sfreq~500Hz that is ~64ms per frame, giving ~18 frames across a
    1.2s epoch window later — enough for pool_cqt_to_node_features to
    average over something meaningful, rather than 1-2 frames with the
    default hop_length.

    Returns (n_freq_bins, n_frames) real-valued magnitude spectrogram.
    """
    if channel_signal.dim() != 1:
        raise ValueError(f"compute_cqt_features expects a 1D signal, got shape {tuple(channel_signal.shape)}")

    cqt = CQT(
        sr=sfreq, fmin=fmin, fmax=fmax, bins_per_octave=bins_per_octave, hop_length=hop_length, verbose=False
    )
    spec = cqt(channel_signal.unsqueeze(0).float())  # (1, n_bins, n_frames)
    return spec.squeeze(0)


def pool_cqt_to_node_features(
    cqt_spectrogram: torch.Tensor,
    event_sample_idx: int,
    sfreq: float,
    hop_length: int,
    tmin: float = -0.2,
    tmax: float = 1.0,
) -> torch.Tensor:
    """Pools (temporal mean) the region of a continuous CQT spectrogram
    corresponding to one event's [tmin, tmax] window around
    event_sample_idx, into a fixed-length vector.

    cqt_spectrogram: (n_freq_bins, n_frames), as produced by
    compute_cqt_features on the FULL continuous signal that
    event_sample_idx indexes into. Returns (n_freq_bins,).

    Raises if the resulting frame window is empty (event too close to
    the recording boundary) rather than silently returning zeros.
    """
    start_sample = event_sample_idx + int(round(tmin * sfreq))
    end_sample = event_sample_idx + int(round(tmax * sfreq))
    start_frame = max(0, start_sample // hop_length)
    end_frame = min(cqt_spectrogram.shape[1], end_sample // hop_length + 1)

    if end_frame <= start_frame:
        raise ValueError(
            f"pool_cqt_to_node_features: empty frame window for event_sample_idx={event_sample_idx} "
            f"(start_frame={start_frame}, end_frame={end_frame}) — event too close to recording boundary."
        )

    return cqt_spectrogram[:, start_frame:end_frame].mean(dim=1)


def build_node_feature_matrix(per_channel_pooled: List[torch.Tensor]) -> torch.Tensor:
    """Stacks per-channel pooled CQT vectors (each (n_freq_bins,), same
    length) into a node feature matrix X in R^{n_channels x d}, the GRN
    encoder's input for one event/epoch."""
    lengths = {t.shape[0] for t in per_channel_pooled}
    if len(lengths) != 1:
        raise ValueError(f"build_node_feature_matrix: inconsistent feature lengths across channels: {lengths}")
    return torch.stack(per_channel_pooled, dim=0)
