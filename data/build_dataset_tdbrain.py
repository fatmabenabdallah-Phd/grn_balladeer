"""
grn_balladeer.data.build_dataset_tdbrain
============================================
Adapts GRN's exact graph-construction pipeline (CQT features, PLV/PLI
connectivity, magnetic-Laplacian) to TDBRAIN's continuous resting-state
recording, via fixed-length, non-overlapping windows -- the same
windowing already used for the RF baseline TDBRAIN experiments (4.0s /
2000 samples at 500Hz), so results are directly comparable.

THE MOTIVATING QUESTION this module exists to answer: after finding a
stable, reproducible (split-half rho=0.70-0.87), though partially
confound-affected, PLV connectivity effect on TDBRAIN (ADHD/ADD
confirmed vs. Healthy, age-matched), the natural next test is whether
GRN -- already validated as capable of learning genuine EEG-ADHD signal
on Nasrabadi -- can learn this same TDBRAIN signal, as a third
cross-dataset data point.

All connectivity, CQT, and Laplacian machinery is imported unchanged
from the existing project modules (connectivity.phase_connectivity,
model.cqt_encoder, model.magnetic_laplacian_conv) -- nothing about
GRN's own architecture is altered, only how epochs/graphs are
constructed from TDBRAIN's continuous, non-event-locked recording.
This module deliberately mirrors build_dataset_nasrabadi.py's structure
exactly, since both datasets share the same continuous-recording
adaptation problem.

IMPORTANT CAVEAT carried over from the TDBRAIN investigation log: the
underlying connectivity signal this module lets GRN attempt to learn
is itself only partially understood -- a sub-cohort/site batch
confound (ADHD_NF) explains much of the effect-size inflation on
restEC but not restEO, an EMG/movement-artifact hypothesis was tested
and rejected on both conditions, and the effect remains statistically
too widespread (65-95% of all channel pairs) to confidently interpret
as a focal neural connectivity biomarker. GRN's own result on this
representation should be read with that context, not as a clean
validation independent of it.
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

TDBRAIN_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC3", "FCz", "FC4",
    "T7", "C3", "Cz", "C4", "T8", "CP3", "CPz", "CP4",
    "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2",
]


def build_subject_dataset_tdbrain(
    channel_data: np.ndarray,
    sfreq: float = 500.0,
    window_samples: int = 2000,
    bands: List[Tuple[float, float]] = ((8.0, 13.0),),
    hop_length: int = 32,
    connectivity_metric: str = "plv",
    frozen_connectivity: bool = False,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Builds GRN-ready (X_i, L_norm_i) graphs from one subject's
    continuous TDBRAIN recording, one graph per non-overlapping
    window_samples-length window (default 2000 samples = 4.0s at
    500Hz, matching the RF baseline TDBRAIN windowing for a direct,
    apples-to-apples comparison).

    channel_data: (26, n_samples) real-valued numpy array, ordered to
    match TDBRAIN_CHANNELS. Raw signal (no additional preprocessing
    applied here beyond what load_tdbrain_raw_epochs already does --
    channel selection and windowing), matching the RF baseline
    experiments, so this comparison isolates the architecture, not a
    preprocessing difference.

    bands: default single alpha band (8-13Hz), matching this project's
    default single-band connectivity convention (and the band used for
    the already-validated TDBRAIN permutation screening/split-half
    checks -- switching bands here would no longer be directly
    comparable to those results).

    hop_length, connectivity_metric: identical meaning and defaults to
    build_dataset_nasrabadi.py, for the same reason (only the epoching
    source differs between datasets, not GRN's own hyperparameters).

    frozen_connectivity: see build_dataset_nasrabadi.py's identical
    parameter for the full rationale (fixed per-recording connectivity,
    computed once, vs. recomputed fresh per window).

    Returns a list of (X_i, L_norm_i) graphs, one per window, in the
    exact same format GRN's training loop already expects.
    """
    n_channels, n_samples = channel_data.shape
    if n_channels != len(TDBRAIN_CHANNELS):
        raise ValueError(
            f"build_subject_dataset_tdbrain: expected {len(TDBRAIN_CHANNELS)} channels, "
            f"got {n_channels} -- check channel_data ordering matches TDBRAIN_CHANNELS."
        )

    data_continuous_t = torch.from_numpy(channel_data).float()
    cqt_per_channel = [
        compute_cqt_features(data_continuous_t[ci], sfreq=sfreq, hop_length=hop_length)
        for ci in range(n_channels)
    ]

    n_windows = n_samples // window_samples
    if n_windows == 0:
        raise ValueError(
            f"build_subject_dataset_tdbrain: recording has {n_samples} samples, "
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
