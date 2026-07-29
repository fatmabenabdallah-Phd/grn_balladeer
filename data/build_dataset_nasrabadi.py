"""
grn_balladeer.data.build_dataset_nasrabadi
==============================================
Adapts GRN's exact graph-construction pipeline (CQT features, PLV/PLI
connectivity, magnetic-Laplacian) from build_dataset.py's event-locked
epoching to Nasrabadi's continuous recording, via fixed-length,
non-overlapping windows -- the same windowing already used for the
RF/EEGNet Nasrabadi experiments (4s / 512 samples at 128Hz), so results
are directly comparable.

THE MOTIVATING QUESTION this module exists to answer: GRN had not
previously been tested on Nasrabadi (its event-locked epoching was
considered incompatible with continuous data). This left an unresolved
ambiguity other cross-dataset findings could not settle: is GRN's
fixed-connectivity, single-scalar-per-node design fundamentally too
constrained to learn ANY EEG-ADHD signal (as distinct from BALLADEER's
data-quality problems), or would it also succeed here the way
RF/EEGNet/TCN-alone did? This module makes that test possible.

All connectivity, CQT, and Laplacian machinery is imported unchanged
from the existing project modules (connectivity.phase_connectivity,
model.cqt_encoder, model.magnetic_laplacian_conv) -- nothing about
GRN's own architecture is altered, only how epochs/graphs are
constructed from a continuous, non-event-locked recording.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from grn_balladeer.connectivity.phase_connectivity import (
    extract_band_signal,
    compute_instantaneous_phase,
    compute_plv_matrix,
    compute_pli_matrix,
    compute_mean_phase_diff,
    build_complex_edge_weights,
    build_magnetic_laplacian,
)
from grn_balladeer.model.magnetic_laplacian_conv import compute_normalized_laplacian
from grn_balladeer.model.cqt_encoder import (
    compute_cqt_features,
    pool_cqt_to_node_features,
    build_node_feature_matrix,
)

NASRABADI_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz",
]


def build_subject_dataset_nasrabadi(
    channel_data: np.ndarray,
    sfreq: float = 128.0,
    window_samples: int = 512,
    bands: List[Tuple[float, float]] = ((4.0, 8.0), (8.0, 13.0), (13.0, 30.0)),
    hop_length: int = 32,
    connectivity_metric: str = "plv",
    frozen_connectivity: bool = False,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Builds GRN-ready (X_i, L_norm_i) graphs from one subject's
    continuous Nasrabadi recording, one graph per non-overlapping
    window_samples-length window (default 512 samples = 4.0s at
    128Hz, matching the RF/EEGNet Nasrabadi windowing for a direct,
    apples-to-apples comparison).

    channel_data: (n_channels, n_samples) real-valued numpy array,
    ordered to match NASRABADI_CHANNELS (19 channels). Raw signal, no
    preprocessing applied here -- matching the RF/EEGNet Nasrabadi
    experiments, which also used the unprocessed signal directly, so
    this comparison isolates the architecture, not a preprocessing
    difference.

    bands, hop_length, connectivity_metric: identical meaning and
    defaults to build_dataset.py's build_subject_dataset, for the CGX/
    BALLADEER pipeline -- kept the same here so GRN's own architecture
    and hyperparameters are unchanged, only the epoching source differs.

    frozen_connectivity: by default (False), PLV/PLI connectivity is
    recomputed fresh for every window, a real recurring per-window
    cost, not a one-time calibration step. When True, connectivity is
    computed ONCE from the subject's full continuous recording and the
    same Laplacian is reused for every window (only the CQT node
    features still vary per window). Tests whether per-window dynamic
    connectivity is necessary for GRN's performance or whether a single
    frozen graph -- computationally far cheaper for continuous
    deployment -- performs comparably. See build_dataset.py's identical
    parameter for the full rationale.

    Returns a list of (X_i, L_norm_i) graphs, one per window, in the
    exact same format GRN's training loop already expects.
    """
    n_channels, n_samples = channel_data.shape
    if n_channels != len(NASRABADI_CHANNELS):
        raise ValueError(
            f"build_subject_dataset_nasrabadi: expected {len(NASRABADI_CHANNELS)} channels, "
            f"got {n_channels} -- check channel_data ordering matches NASRABADI_CHANNELS."
        )

    data_continuous_t = torch.from_numpy(channel_data).float()
    cqt_per_channel = [
        compute_cqt_features(data_continuous_t[ci], sfreq=sfreq, hop_length=hop_length)
        for ci in range(n_channels)
    ]

    n_windows = n_samples // window_samples
    if n_windows == 0:
        raise ValueError(
            f"build_subject_dataset_nasrabadi: recording has {n_samples} samples, "
            f"shorter than one window_samples={window_samples} -- no windows possible."
        )

    if connectivity_metric == "plv":
        strength_fn = compute_plv_matrix
    elif connectivity_metric == "pli":
        strength_fn = compute_pli_matrix
    else:
        raise ValueError(f"connectivity_metric must be 'plv' or 'pli', got '{connectivity_metric}'")

    def _compute_laplacian(signal_segment: np.ndarray) -> torch.Tensor:
        """Factored out so both the per-window (default) and frozen
        (computed once) paths share identical logic."""
        W_per_band = []
        strength_per_band = []
        for band in bands:
            band_signal = extract_band_signal(signal_segment, band, sfreq)
            phases = compute_instantaneous_phase(band_signal)
            strength_band = strength_fn(phases)
            phase_diff_band = compute_mean_phase_diff(phases)
            W_per_band.append(build_complex_edge_weights(strength_band, phase_diff_band))
            strength_per_band.append(strength_band)
        W = np.mean(W_per_band, axis=0)
        strength = np.mean(strength_per_band, axis=0)
        L_C = build_magnetic_laplacian(W, strength)
        return compute_normalized_laplacian(torch.from_numpy(L_C).to(torch.complex64))

    frozen_L_norm = _compute_laplacian(channel_data) if frozen_connectivity else None

    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for w in range(n_windows):
        window_start_sample = w * window_samples

        # CQT node features: pool the continuous per-channel CQT
        # spectrogram over exactly this window's [0, window_duration]
        # span, using pool_cqt_to_node_features's existing tmin/tmax
        # mechanism (window_start_sample plays the role event_sample_idx
        # plays in the CGX pipeline; tmin=0 here since we window from
        # the start of each block, not centered on an event).
        window_duration_s = window_samples / sfreq
        per_channel_pooled = [
            pool_cqt_to_node_features(
                cqt_per_channel[ci], window_start_sample, sfreq, hop_length,
                tmin=0.0, tmax=window_duration_s,
            )
            for ci in range(n_channels)
        ]
        X_i = build_node_feature_matrix(per_channel_pooled)

        if frozen_connectivity:
            L_norm_i = frozen_L_norm
        else:
            window_signal = channel_data[:, window_start_sample:window_start_sample + window_samples]
            L_norm_i = _compute_laplacian(window_signal)

        dataset.append((X_i, L_norm_i))

    return dataset
