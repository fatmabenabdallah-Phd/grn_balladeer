"""
grn_balladeer.data.build_dataset_lightweight
================================================
Dataset builder for the lightweight TCN + structural-graph architecture.
Deliberately bypasses CQT feature extraction AND per-epoch PLV/PLI
connectivity entirely -- both are unnecessary here since (1) the graph
is a FIXED structural k-NN adjacency (computed once, reused for every
subject/epoch, see connectivity/structural_graph.py) and (2) the TCN
operates directly on raw per-node time series rather than a frequency-
transformed representation, matching ICCCI2026's "raw signal processing
(end-to-end learning)" branch of their feature-extraction taxonomy
(Sec. 2.1) rather than GRN's CQT-based approach.

This should be substantially cheaper to build than build_dataset.py's
full pipeline (no CQT, no per-epoch PLV/PLI/magnetic-Laplacian
computation), on top of already being architecturally lighter at
training/inference time.
"""

from __future__ import annotations

import json
from typing import List, Tuple

import numpy as np
import torch

import mne
from grn_balladeer.preprocessing.mne_loading import load_eeg_cgx, CGX_CHANNELS
from grn_balladeer.preprocessing.filtering import apply_standard_filters
from grn_balladeer.preprocessing.ica import run_ica_artifact_removal
from grn_balladeer.preprocessing.epoching import flags_to_samples, epoch_by_flag_events
from grn_balladeer.preprocessing.bad_channels import (
    set_standard_montage, detect_bad_channels, interpolate_bad_channels,
)
from grn_balladeer.preprocessing.epoch_rejection import reject_bad_epochs
from grn_balladeer.eval.baselines import extract_band_power_features


def build_subject_dataset_lightweight(
    cgx_path: str,
    flags_path: str,
    level: str,
    skip_ica: bool = False,
    clean_bad_channels: bool = False,
    reject_epochs: bool = False,
    epoch_rejection_threshold: float = 150e-6,
    common_average_reference: bool = False,
    surface_laplacian: bool = False,
) -> List[Tuple[torch.Tensor, np.ndarray]]:
    """Runs preprocessing (load -> filter -> [bad-channel cleaning] ->
    [re-reference/CSD] -> [ICA] -> epoch -> [epoch rejection])
    identically to build_dataset.py, but stops there -- no CQT, no
    per-epoch connectivity computation. Returns a list of
    (raw_epoch_tensor, band_power_features), one per KEPT epoch:
      - raw_epoch_tensor: (n_channels, n_timepoints) real-valued torch
        tensor, fed directly to LightweightTCNEncoder.
      - band_power_features: (n_features,) numpy array from
        eval.baselines.extract_band_power_features (band power per
        channel + theta/beta ratio) -- the same features that let a
        plain Random Forest reach AUC=0.668 this session, reused here
        as an explicit, near-zero-cost auxiliary signal fused with the
        TCN's learned representation rather than discarded.

    skip_ica: tests the hypothesis that ICA-based artifact removal may
    be stripping genuine, frontally-weighted ADHD-relevant signal along
    with real ocular artifacts.

    clean_bad_channels: detects and interpolates bad EEG channels
    (preprocessing.bad_channels) before epoching. Motivated by the
    discovery that the ExG reference channels were silently dead
    throughout this project.

    reject_epochs: drops individual epochs whose peak-to-peak amplitude
    (any channel) exceeds epoch_rejection_threshold
    (preprocessing.epoch_rejection), a standard EEG QC step never
    previously implemented on BALLADEER. Complements clean_bad_channels:
    a channel can be consistently bad across a whole recording (handled
    there), while an epoch can be transiently corrupted on an
    otherwise-good channel (handled here).

    common_average_reference: re-references EEG channels to the common
    average, MNE's built-in set_eeg_reference('average'). Applied AFTER
    bad-channel cleaning (if enabled) so a genuinely bad channel does
    not corrupt the average it contributes to, and BEFORE ICA.

    surface_laplacian: NEW this session -- applies surface Laplacian /
    Current Source Density re-referencing (MNE's built-in
    compute_current_source_density()), a spatial filter that sharpens
    localization by emphasizing local, radially-oriented sources over
    volume-conducted signal shared across the scalp -- motivated by the
    hypothesis that this might sharpen the frontal theta/beta signal
    ADHD's biomarker literature focuses on. Requires electrode
    positions (uses the same standard_1020 montage as
    clean_bad_channels/connectivity.structural_graph). Mutually
    exclusive with common_average_reference in practice: CSD is itself
    a reference-free spatial transform, and applying both is redundant
    rather than complementary -- if both are requested, CSD is applied
    and common_average_reference is skipped with a note in this
    docstring (not a runtime warning, since this is a deliberate,
    documented design choice, not an error condition).

    level: same convention as build_dataset.py's build_subject_dataset
    (e.g. 'Level1') -- pass the confirmed level explicitly. Flag-file
    parsing here matches build_dataset.py's own inline logic exactly
    (same json structure, same flags_to_samples/epoch_by_flag_events
    calls) to guarantee identical epoch selection between the two
    architectures -- a prerequisite for a fair GRN-vs-TCN comparison.
    """
    raw = load_eeg_cgx(cgx_path)
    raw_filt = apply_standard_filters(raw)

    if clean_bad_channels:
        set_standard_montage(raw_filt, CGX_CHANNELS)
        report = detect_bad_channels(raw_filt, CGX_CHANNELS)
        interpolate_bad_channels(raw_filt, report["bad_channels"], CGX_CHANNELS)

    if surface_laplacian:
        if not clean_bad_channels:  # montage not yet attached in that branch
            set_standard_montage(raw_filt, CGX_CHANNELS)
        raw_filt = mne.preprocessing.compute_current_source_density(raw_filt, verbose=False)
    elif common_average_reference:
        raw_filt.set_eeg_reference("average", ch_type="eeg", verbose=False)

    if skip_ica:
        raw_clean = raw_filt
    else:
        raw_clean, _ = run_ica_artifact_removal(raw_filt)

    with open(flags_path) as f:
        flags_info = json.load(f)["slackline_levels_flags_info"]
    matching_levels = [lv for lv in flags_info if lv["level"] == level]
    if not matching_levels:
        raise ValueError(f"build_subject_dataset_lightweight: level '{level}' not found in {flags_path}")
    level_flags = matching_levels[0]["flags"]

    sfreq = raw_clean.info["sfreq"]
    sample_indices, flag_types = flags_to_samples(level_flags, sfreq)
    epochs = epoch_by_flag_events(raw_clean, sample_indices, flag_types)

    if reject_epochs:
        epochs, _ = reject_bad_epochs(epochs, threshold=epoch_rejection_threshold)

    epoch_data_all = epochs.get_data(picks=CGX_CHANNELS)  # (n_epochs, n_channels, n_timepoints)
    band_power_feats = extract_band_power_features(epochs, channels=CGX_CHANNELS)  # (n_epochs, n_features)

    dataset = []
    for i in range(len(epochs)):
        raw_epoch = torch.from_numpy(epoch_data_all[i]).float()
        dataset.append((raw_epoch, band_power_feats[i]))

    return dataset
