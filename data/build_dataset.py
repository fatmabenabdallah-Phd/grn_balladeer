"""
grn_balladeer.data.build_dataset
===================================
Reusable subject-level dataset builder: load -> filter -> ICA ->
continuous CQT -> per-epoch connectivity -> normalized magnetic
Laplacian. This is the recipe that was hand-run separately for
UB0136, UB0004, and UB0022 (see context-transfer docs v3-v6) --
formalized here so a 4th subject does not require another copy-paste.

Produces a list of (X_i, L_norm_i) real graphs, one per kept epoch,
matching the format already saved as real_dataset_UB0136.pt /
real_dataset_UB0004.pt / real_dataset_UB0022.pt.

NOTE: per standing instruction, real subject data and any .pt
checkpoint built from it must stay local -- this module is committed,
its OUTPUT is not.
"""

from __future__ import annotations

import json
from typing import List, Tuple

import numpy as np
import torch

from grn_balladeer.preprocessing.mne_loading import load_eeg_cgx, CGX_CHANNELS
from grn_balladeer.preprocessing.filtering import apply_standard_filters
from grn_balladeer.preprocessing.ica import run_ica_artifact_removal
from grn_balladeer.preprocessing.epoching import flags_to_samples, epoch_by_flag_events
from grn_balladeer.connectivity.phase_connectivity import (
    extract_band_signal,
    compute_instantaneous_phase,
    compute_plv_matrix,
    compute_mean_phase_diff,
    build_complex_edge_weights,
    build_magnetic_laplacian,
)
from grn_balladeer.model.magnetic_laplacian_conv import compute_normalized_laplacian
from grn_balladeer.model.cqt_encoder import compute_cqt_features, pool_cqt_to_node_features, build_node_feature_matrix


def build_subject_dataset(
    cgx_path: str,
    flags_path: str,
    level: str,
    bands: List[Tuple[float, float]] = ((4.0, 8.0), (8.0, 13.0), (13.0, 30.0)),
    hop_length: int = 32,
    return_epochs: bool = False,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Runs the full Module 2b -> 3 -> 4 chain on one subject's real CGX
    file and returns a list of (X_i, L_norm_i) graphs, one per epoch
    kept by epoch_by_flag_events (events outside the recording's
    available range are dropped, not an error -- see epoching.py).

    return_epochs: NEW this session -- if True, ALSO returns the raw
    mne.Epochs object used internally to build the graphs, as
    (dataset, epochs) instead of just dataset. Added so baseline
    comparisons (eval.baselines.extract_band_power_features, which
    needs raw epoched EEG, not the CQT-encoded complex graph tensors
    this function otherwise returns) can be computed on EXACTLY the
    same epochs GRN sees, rather than a separately re-run preprocessing
    pass that could silently diverge (different ICA components dropped,
    different kept-epoch count, etc.) and make the comparison unfair
    without anyone noticing.

    level: one of the 'level' values in slackline_flags_info.json
    (e.g. 'Level1'). This function does NOT infer the level from the
    subject's TAGS file -- that determination (timing cross-check
    against slackline_flags_info.json, or direct session metadata) is
    the caller's responsibility; pass the confirmed level explicitly.

    bands: CHANGED this session -- was a single band tuple (default
    alpha 8-13Hz only), now a list of bands, default theta+alpha+beta
    ((4,8),(8,13),(13,30)). Motivation: the harmonic/symbolic losses
    (L_harm, L_symb) are grounded in CROSS-frequency phase synchrony
    literature (Palva et al.) and the theta/beta ratio ADHD literature
    (Barry et al., Snyder & Hall) -- both about relationships BETWEEN
    bands -- but the adjacency itself was previously built from a
    SINGLE band's phase (alpha only), meaning the graph structure the
    model actually sees carried no direct cross-band information
    despite the loss terms' theoretical grounding assuming it might.
    The complex edge-weight matrices W_band from each band are
    averaged (before Laplacian construction) into a single combined W,
    from which ONE magnetic Laplacian is built -- i.e. still one graph
    per epoch, now reflecting theta+alpha+beta synchrony jointly
    rather than alpha alone. Passing a single-element list (e.g.
    [(8.0, 13.0)]) exactly reproduces the old alpha-only behavior --
    this change is backward compatible, not a breaking one.
    """
    raw = load_eeg_cgx(cgx_path)
    raw_filt = apply_standard_filters(raw)
    raw_clean, ica_report = run_ica_artifact_removal(raw_filt)

    with open(flags_path) as f:
        flags_info = json.load(f)["slackline_levels_flags_info"]
    matching_levels = [lv for lv in flags_info if lv["level"] == level]
    if not matching_levels:
        raise ValueError(f"build_subject_dataset: level '{level}' not found in {flags_path}")
    level_flags = matching_levels[0]["flags"]

    sfreq = raw_clean.info["sfreq"]
    sample_indices, flag_types = flags_to_samples(level_flags, sfreq)
    epochs = epoch_by_flag_events(raw_clean, sample_indices, flag_types)

    data_continuous = raw_clean.get_data(picks=CGX_CHANNELS)
    data_continuous_t = torch.from_numpy(data_continuous).float()
    cqt_per_channel = [
        compute_cqt_features(data_continuous_t[ci], sfreq=sfreq, hop_length=hop_length)
        for ci in range(data_continuous_t.shape[0])
    ]

    kept_events = epochs.events
    epoch_data_all = epochs.get_data(picks=CGX_CHANNELS)

    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(len(epochs)):
        event_sample_idx = int(kept_events[i, 0])

        per_channel_pooled = [
            pool_cqt_to_node_features(cqt_per_channel[ci], event_sample_idx, sfreq, hop_length)
            for ci in range(len(CGX_CHANNELS))
        ]
        X_i = build_node_feature_matrix(per_channel_pooled)

        epoch_signal = epoch_data_all[i]

        # NEW this session: average connectivity across all requested bands
        # (default theta+alpha+beta) into ONE combined adjacency, rather than
        # using a single band (alpha only, as before) -- see this function's
        # docstring for why this matters given L_harm/L_symb's cross-frequency
        # theoretical grounding. plv is also averaged (not just W) since
        # build_magnetic_laplacian needs a real-valued plv magnitude matrix
        # alongside the complex W, and averaging each band's own valid PLV
        # (each in [0,1]) keeps that property (average of values in [0,1] is
        # still in [0,1]), unlike e.g. re-deriving plv from the averaged W.
        W_per_band = []
        plv_per_band = []
        for band in bands:
            band_signal = extract_band_signal(epoch_signal, band, sfreq)
            phases = compute_instantaneous_phase(band_signal)
            plv_band = compute_plv_matrix(phases)
            phase_diff_band = compute_mean_phase_diff(phases)
            W_per_band.append(build_complex_edge_weights(plv_band, phase_diff_band))
            plv_per_band.append(plv_band)

        W = np.mean(W_per_band, axis=0)
        plv = np.mean(plv_per_band, axis=0)
        L_C = build_magnetic_laplacian(W, plv)
        L_norm_i = compute_normalized_laplacian(torch.from_numpy(L_C).to(torch.complex64))

        dataset.append((X_i, L_norm_i))

    if return_epochs:
        return dataset, epochs
    return dataset
