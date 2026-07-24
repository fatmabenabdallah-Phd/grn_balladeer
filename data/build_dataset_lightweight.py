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

from grn_balladeer.preprocessing.mne_loading import load_eeg_cgx, CGX_CHANNELS
from grn_balladeer.preprocessing.filtering import apply_standard_filters
from grn_balladeer.preprocessing.ica import run_ica_artifact_removal
from grn_balladeer.preprocessing.epoching import flags_to_samples, epoch_by_flag_events
from grn_balladeer.eval.baselines import extract_band_power_features


def build_subject_dataset_lightweight(
    cgx_path: str,
    flags_path: str,
    level: str,
) -> List[Tuple[torch.Tensor, np.ndarray]]:
    """Runs preprocessing (load -> filter -> ICA -> epoch) identically
    to build_dataset.py, but stops there -- no CQT, no per-epoch
    connectivity computation. Returns a list of (raw_epoch_tensor,
    band_power_features), one per kept epoch:
      - raw_epoch_tensor: (n_channels, n_timepoints) real-valued torch
        tensor, fed directly to LightweightTCNEncoder.
      - band_power_features: (n_features,) numpy array from
        eval.baselines.extract_band_power_features (band power per
        channel + theta/beta ratio) -- the same features that let a
        plain Random Forest reach AUC=0.668 this session, reused here
        as an explicit, near-zero-cost auxiliary signal fused with the
        TCN's learned representation rather than discarded.

    level: same convention as build_dataset.py's build_subject_dataset
    (e.g. 'Level1') -- pass the confirmed level explicitly. Flag-file
    parsing here matches build_dataset.py's own inline logic exactly
    (same json structure, same flags_to_samples/epoch_by_flag_events
    calls) to guarantee identical epoch selection between the two
    architectures -- a prerequisite for a fair GRN-vs-TCN comparison.
    """
    raw = load_eeg_cgx(cgx_path)
    raw_filt = apply_standard_filters(raw)
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

    epoch_data_all = epochs.get_data(picks=CGX_CHANNELS)  # (n_epochs, n_channels, n_timepoints)
    band_power_feats = extract_band_power_features(epochs)  # (n_epochs, n_features)

    dataset = []
    for i in range(len(epochs)):
        raw_epoch = torch.from_numpy(epoch_data_all[i]).float()
        dataset.append((raw_epoch, band_power_feats[i]))

    return dataset
