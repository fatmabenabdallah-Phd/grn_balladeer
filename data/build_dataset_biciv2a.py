"""
grn_balladeer.data.build_dataset_biciv2a
============================================
Adapts GRN's exact graph-construction pipeline (CQT features, PLV/PLI
connectivity, magnetic-Laplacian) from build_dataset.py's event-locked
epoching to BCI Competition IV 2a's motor-imagery trials -- the third
domain in the architectural-generality study (after ADHD/BALLADEER and
ADHD/Nasrabadi), this time on a non-clinical, spatially distinct task
(left- vs right-hand motor imagery, ERD/ERS centered on C3/C4, not the
frontal theta-gamma coupling GRN's neurosymbolic losses were originally
anchored to for ADHD).

THE MOTIVATING QUESTION this module exists to answer: does GRN's fixed
PLV/magnetic-Laplacian connectivity and CQT-based node encoding extract
usable signal from a spatio-spectral structure that has nothing to do
with the frontal/theta-gamma literature it was built around, or is the
prior cross-dataset success (Nasrabadi) itself somehow ADHD-signal
specific? This module makes that test possible, independent of the
L_symb cluster question (see losses.motor_imagery_channels).

DATA SOURCE: BCI Competition IV 2a, obtained via the public GitHub
mirror bregydoc/bcidatasetIV2a (bbci.de itself is not reachable from
this environment's network allowlist). Only the T (training) session
files are used -- the E (evaluation) session's true labels are
distributed separately by the original competition and are NOT present
in this mirror (verified: A0xE.npz event codes are all generic '783',
not 769-772). Using T-session data only across all 9 subjects, split
leave-one-subject-out, avoids depending on an unverified third-party
label source for E, at the cost of a smaller n per subject -- an
explicit, documented tradeoff, not an oversight.

TRIAL EXTRACTION -- VERIFIED QUIRK: a rejected trial's GDF event stream
is `768 (trial start), 1023 (rejected), <true cue code>, ...` -- the
1023 marker PRECEDES the real cue code, it does not replace it. Naively
reading `etyp[idx+1]` right after each 768 as "the cue" misclassifies
15/288 trials per subject as artifact-free noise (silently keeping
mislabeled trials) or drops real cue information. Confirmed by direct
inspection of A01T.npz: with the `1023-precedes-cue` logic below, class
counts are exactly balanced (72/72/72/72 across all four MI classes,
matching the literature's reported paradigm), and the derived
artifact-flag vector matches the file's own `artifacts` array exactly
(np.array_equal, verified for all 9 subjects) -- both were false before
correcting the offset.

Restricting to LEFT (769) and RIGHT (770) hand classes only, artifact
trials excluded, real counts across all 9 subjects' T sessions (this
session's actual run, not an estimate):
  A01T: 138 (L=69, R=69)   A02T: 136 (L=67, R=69)   A03T: 137 (L=69, R=68)
  A04T: 129 (L=62, R=67)   A05T: 129 (L=63, R=66)   A06T: 113 (L=56, R=57)
  A07T: 133 (L=67, R=66)   A08T: 132 (L=66, R=66)   A09T: 116 (L=53, R=63)
  TOTAL: 1163 clean binary trials across 9 subjects.

All connectivity, CQT, and Laplacian machinery is imported unchanged
from the existing project modules -- nothing about GRN's own
architecture is altered, only how epochs/graphs are constructed from
BCI IV 2a's specific event-coding scheme.
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

# Channel order as stored in the bregydoc/bcidatasetIV2a .npz 's' array
# (verified against the repo's own worked example: index 7 == 'C3',
# which only matches this exact ordering). 22 EEG channels first, then
# 3 EOG (columns 22-24 of 's', excluded here -- GRN operates on EEG
# nodes only).
BICIV2A_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
    "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
]

TRIAL_START_CODE = 768
REJECTED_TRIAL_CODE = 1023
LEFT_HAND_CODE = 769
RIGHT_HAND_CODE = 770
SFREQ_HZ = 250.0


def extract_binary_trials(
    npz_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parses one subject's raw BCI IV 2a .npz event stream into clean
    (non-artifact) left/right-hand trial cue positions and labels.

    Handles the verified 1023-precedes-cue quirk documented in this
    module's docstring: for each `768` trial-start event, the cue code
    is `etyp[idx+1]` normally, but `etyp[idx+2]` when `etyp[idx+1] ==
    1023` (rejected trial) -- reading `idx+1` unconditionally would
    silently misclassify rejected trials' cue codes.

    Returns (cue_sample_positions, labels, all_artifact_flags) where:
      - cue_sample_positions: (n_clean,) int array, sample index of
        cue onset for each kept (non-rejected, left/right-only) trial.
      - labels: (n_clean,) int array, 0 = left hand (769), 1 = right
        hand (770).
      - all_artifact_flags: (288,) uint8 array, the file's own
        artifact flag for every trial (not just left/right) -- returned
        for the caller to optionally cross-check against
        data['artifacts'], as done in this module's own verification.

    Raises if the derived artifact-flag vector does not match the
    file's stored 'artifacts' array exactly -- this would indicate the
    1023-offset assumption is wrong for this file and must not be
    silently trusted.
    """
    data = np.load(npz_path, allow_pickle=True)
    etyp = data["etyp"].flatten()
    epos = data["epos"].flatten()
    stored_artifacts = data["artifacts"].flatten()

    trial_start_idx = np.where(etyp == TRIAL_START_CODE)[0]

    classes: List[int] = []
    positions: List[int] = []
    artifact_flags: List[int] = []
    for idx in trial_start_idx:
        if etyp[idx + 1] == REJECTED_TRIAL_CODE:
            cue_type, cue_pos, art = etyp[idx + 2], epos[idx + 2], 1
        else:
            cue_type, cue_pos, art = etyp[idx + 1], epos[idx + 1], 0
        classes.append(int(cue_type))
        positions.append(int(cue_pos))
        artifact_flags.append(art)

    classes_arr = np.array(classes)
    positions_arr = np.array(positions)
    artifact_flags_arr = np.array(artifact_flags, dtype=np.uint8)

    if not np.array_equal(artifact_flags_arr, stored_artifacts):
        raise ValueError(
            f"extract_binary_trials: derived artifact flags do not match "
            f"{npz_path}'s own 'artifacts' array -- the 1023-precedes-cue "
            f"assumption may not hold for this file, do not trust the "
            f"extracted labels without investigating."
        )

    keep = np.isin(classes_arr, [LEFT_HAND_CODE, RIGHT_HAND_CODE]) & (artifact_flags_arr == 0)
    labels = (classes_arr[keep] == RIGHT_HAND_CODE).astype(int)  # 0=left, 1=right
    return positions_arr[keep], labels, stored_artifacts


def build_subject_dataset_biciv2a(
    npz_path: str,
    sfreq: float = SFREQ_HZ,
    mi_window_s: float = 4.0,
    bands: List[Tuple[float, float]] = ((8.0, 13.0), (13.0, 30.0)),
    hop_length: int = 32,
    connectivity_metric: str = "plv",
    frozen_connectivity: bool = False,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], np.ndarray]:
    """Builds GRN-ready (X_i, L_norm_i) graphs for one subject's clean
    left/right-hand trials, from the raw BCI IV 2a .npz file.

    mi_window_s: pooling window length, in seconds, starting at cue
    onset (default 4.0s). Per the verified paradigm timing (fixation
    0-2s, cue at t=2s, motor imagery cued for the following 4s, i.e.
    absolute t=[2,6]s -- Tangermann et al., "Review of the BCI
    Competition IV", Frontiers in Neuroscience, 2012; consistent across
    multiple independent re-implementations), the full 4s post-cue
    window is used rather than a narrower sub-window, matching common
    practice in the BCI IV 2a literature (e.g. Zhao et al., EEG-DCNet,
    2024, uses the same [cue, cue+4s] convention).

    bands: default (8-13Hz mu/alpha, 13-30Hz beta) -- NOT the
    theta+alpha+beta set used for ADHD (build_dataset.py's default),
    deliberately: mu/beta ERD/ERS is the literature-established
    spatio-spectral signature of motor imagery (Pfurtscheller & Lopes
    da Silva, 1999, Clin Neurophysiol 110(11):1842-1857; Pfurtscheller &
    Neuper, 2001, Proc IEEE 89(7):1123-1134), whereas ADHD's frontal
    theta-gamma coupling motivated the wider band set there. Using the
    ADHD band set here would silently reintroduce a task-specific
    assumption from the wrong domain into what is meant to be a fair
    architectural-generality test.

    Returns (dataset, subject_labels) where dataset is a list of
    (X_i, L_norm_i) graphs (same format GRN's training loop already
    expects) and subject_labels is the (n_trials,) int array of 0/1
    labels aligned with dataset, in the same trial order.
    """
    cue_positions, labels, _ = extract_binary_trials(npz_path)
    if len(cue_positions) == 0:
        raise ValueError(f"build_subject_dataset_biciv2a: no clean left/right trials found in {npz_path}")

    raw = np.load(npz_path, allow_pickle=True)
    s = raw["s"]  # (n_samples, 25) -- 22 EEG + 3 EOG
    n_channels = len(BICIV2A_CHANNELS)
    channel_data = s[:, :n_channels].T  # (n_channels, n_samples), EEG only

    if np.isnan(channel_data).any():
        raise ValueError(f"build_subject_dataset_biciv2a: NaN values found in {npz_path}'s EEG data")

    data_continuous_t = torch.from_numpy(channel_data).float()
    cqt_per_channel = [
        compute_cqt_features(data_continuous_t[ci], sfreq=sfreq, hop_length=hop_length)
        for ci in range(n_channels)
    ]

    if connectivity_metric == "plv":
        strength_fn = compute_plv_matrix
    elif connectivity_metric == "pli":
        strength_fn = compute_pli_matrix
    else:
        raise ValueError(f"connectivity_metric must be 'plv' or 'pli', got '{connectivity_metric}'")

    def _compute_laplacian(signal_segment: np.ndarray) -> torch.Tensor:
        """Factored out so both the per-trial (default) and frozen
        (computed once) paths share identical logic -- mirrors
        build_dataset.py / build_dataset_nasrabadi.py exactly."""
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

    window_samples = int(round(mi_window_s * sfreq))
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for cue_sample in cue_positions:
        per_channel_pooled = [
            pool_cqt_to_node_features(
                cqt_per_channel[ci], int(cue_sample), sfreq, hop_length,
                tmin=0.0, tmax=mi_window_s,
            )
            for ci in range(n_channels)
        ]
        X_i = build_node_feature_matrix(per_channel_pooled)

        if frozen_connectivity:
            L_norm_i = frozen_L_norm
        else:
            trial_signal = channel_data[:, cue_sample:cue_sample + window_samples]
            L_norm_i = _compute_laplacian(trial_signal)

        dataset.append((X_i, L_norm_i))

    return dataset, labels
