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
from grn_balladeer.preprocessing.bad_channels import (
    set_standard_montage, detect_bad_channels, interpolate_bad_channels,
)
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
from grn_balladeer.model.cqt_encoder import compute_cqt_features, pool_cqt_to_node_features, build_node_feature_matrix


def build_subject_dataset(
    cgx_path: str,
    flags_path: str,
    level: str,
    bands: List[Tuple[float, float]] = ((4.0, 8.0), (8.0, 13.0), (13.0, 30.0)),
    hop_length: int = 32,
    return_epochs: bool = False,
    connectivity_metric: str = "plv",
    clean_bad_channels: bool = False,
    exclude_bad_channels: bool = False,
    fixed_exclude_channels: "List[str] | None" = None,
    frozen_connectivity: bool = False,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Runs the full Module 2b -> 3 -> 4 chain on one subject's real CGX
    file and returns a list of (X_i, L_norm_i) graphs, one per epoch
    kept by epoch_by_flag_events (events outside the recording's
    available range are dropped, not an error -- see epoching.py).

    return_epochs: if True, ALSO returns the raw
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

    bands: a list of bands (previously a single band tuple, default
    alpha 8-13Hz only), default theta+alpha+beta
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

    connectivity_metric: 'plv' (default, unchanged
    behavior) or 'pli' (Phase Lag Index, an alternative connectivity
    measure less sensitive to volume conduction/zero-lag artifacts than
    PLV, since PLI discards exact zero-phase-lag synchrony by
    construction -- see connectivity/phase_connectivity.py's own
    compute_pli_matrix docstring). Only the amplitude/strength matrix
    changes between the two metrics (PLV vs PLI); the mean
    phase-difference matrix feeding into the complex edge weights is
    computed identically either way, since PLI itself has no natural
    phase-difference counterpart (it discards phase sign by
    construction).

    clean_bad_channels: detects and interpolates
    bad EEG channels (preprocessing.bad_channels) BEFORE the PLV/PLI
    connectivity graph and magnetic Laplacian are built, rather than
    only at the band-power-feature stage (as in
    build_dataset_lightweight.py). This is the definitive test of a
    hypothesis raised by GRN's own cross-dataset validation (see the
    manuscript's Discussion section): GRN's fixed connectivity graph
    may propagate a corrupted channel's noise to every node connected
    to it via the magnetic-Laplacian convolution, a failure mode
    channel-independent architectures (Random Forest, TCN alone) do
    not share. If this hypothesis is correct, cleaning channels BEFORE
    the graph is constructed (not just before band-power features are
    computed) should matter specifically for GRN, in a way it did not
    for the classical baselines.

    exclude_bad_channels: mutually exclusive with
    clean_bad_channels. Instead of interpolating detected bad channels
    (which reconstructs each one as a linear combination of its clean
    neighbors, potentially inflating PLV between them with spurious,
    non-physiological synchrony), this option DROPS them entirely from
    the channel set used to build the graph -- fewer nodes, no
    fabricated connectivity. Motivated by an unexpected finding: on
    BALLADEER, interpolation-based cleaning made GRN's performance
    WORSE (MEAN AUC 0.440 vs. 0.517 uncleaned), the opposite of what
    the noise-propagation hypothesis predicted -- suggesting
    interpolation itself, not just leftover channel noise, may be the
    problem.

    CORRECTION vs. an earlier draft of this docstring: per-subject
    adaptive exclusion (detecting and dropping whichever channels are
    bad for THAT subject) produces a DIFFERENT number of graph nodes
    per subject (confirmed empirically: 19-30 nodes across the cohort,
    mean 27), which breaks training.cross_validation.train_fold's
    batched tensor stacking (torch.stack requires uniform shape across
    the whole training/validation batch) -- GRNEncoder itself tolerates
    variable node counts when called on one subject at a time, but the
    shared training loop does not. Use fixed_exclude_channels (below)
    for anything going through train_fold.

    fixed_exclude_channels: a fixed list of channel names (e.g. the
    dataset-wide top bad channels already characterized --
    Fp1, Fp2, Fpz, AF7, AF8, F7, CP6, Cz, A2, C4) to drop identically
    for EVERY subject, regardless of that individual subject's own
    detected bad channels. Guarantees uniform node count across the
    cohort, compatible with train_fold. Takes precedence over
    exclude_bad_channels's per-subject adaptive detection if both are
    set; leave exclude_bad_channels=False when using this parameter.
    Tests a related but distinct question from per-subject exclusion:
    not "remove whichever channels are bad for this specific person"
    but "does removing the channels known to be chronically
    problematic across the cohort, uniformly, help GRN."

    frozen_connectivity: by default (False), PLV/PLI
    connectivity (and the resulting magnetic Laplacian) is recomputed
    FRESH for every single epoch, inside the per-epoch loop below --
    this is a real, recurring computational cost at inference time, not
    a one-time calibration step, contrary to an intuitive but incorrect
    reading of "fixed connectivity" as "computed once when the headset
    is fitted." When frozen_connectivity=True, connectivity is instead
    computed ONCE from the subject's full continuous recording (a
    calibration-like single connectivity graph), and this SAME Laplacian
    is reused for every epoch -- only the CQT node features (X_i) still
    vary per epoch, not the graph structure (L_norm). This tests
    whether GRN's per-epoch dynamic connectivity (potentially reflecting
    moment-to-moment attention fluctuations relevant to ADHD) is
    actually necessary for its performance, or whether a single frozen
    connectivity graph -- computationally far cheaper for continuous
    edge deployment, since PLV would then be an upfront cost rather than
    a recurring one -- performs comparably.
    """
    raw = load_eeg_cgx(cgx_path)
    raw_filt = apply_standard_filters(raw)

    active_channels = list(CGX_CHANNELS)

    if fixed_exclude_channels:
        active_channels = [ch for ch in CGX_CHANNELS if ch not in fixed_exclude_channels]
        if len(active_channels) < 2:
            raise ValueError(
                f"build_subject_dataset: fixed_exclude_channels left only "
                f"{len(active_channels)} channels -- too few to form a graph."
            )
    elif clean_bad_channels or exclude_bad_channels:
        set_standard_montage(raw_filt, CGX_CHANNELS)
        bad_channel_report = detect_bad_channels(raw_filt, CGX_CHANNELS)
        if clean_bad_channels:
            interpolate_bad_channels(raw_filt, bad_channel_report["bad_channels"], CGX_CHANNELS)
        elif exclude_bad_channels:
            active_channels = [ch for ch in CGX_CHANNELS if ch not in bad_channel_report["bad_channels"]]
            if len(active_channels) < 2:
                raise ValueError(
                    f"build_subject_dataset: exclude_bad_channels left only "
                    f"{len(active_channels)} channels -- too few to form a graph "
                    f"(bad_channels={bad_channel_report['bad_channels']})."
                )

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

    data_continuous = raw_clean.get_data(picks=active_channels)
    data_continuous_t = torch.from_numpy(data_continuous).float()
    cqt_per_channel = [
        compute_cqt_features(data_continuous_t[ci], sfreq=sfreq, hop_length=hop_length)
        for ci in range(data_continuous_t.shape[0])
    ]

    kept_events = epochs.events
    epoch_data_all = epochs.get_data(picks=active_channels)

    if connectivity_metric == "plv":
        strength_fn = compute_plv_matrix
    elif connectivity_metric == "pli":
        strength_fn = compute_pli_matrix
    else:
        raise ValueError(f"connectivity_metric must be 'plv' or 'pli', got '{connectivity_metric}'")

    def _compute_laplacian(signal_segment: np.ndarray) -> torch.Tensor:
        """Computes one (X-independent) magnetic Laplacian from a signal
        segment -- factored out so both the per-epoch (default) and
        frozen (computed once) paths share identical logic."""
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

    # frozen_connectivity: compute ONE Laplacian from the full continuous
    # recording, reused for every epoch below -- a single calibration-like
    # cost instead of a per-epoch recurring one (see docstring above).
    frozen_L_norm = _compute_laplacian(data_continuous) if frozen_connectivity else None

    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(len(epochs)):
        event_sample_idx = int(kept_events[i, 0])

        per_channel_pooled = [
            pool_cqt_to_node_features(cqt_per_channel[ci], event_sample_idx, sfreq, hop_length)
            for ci in range(len(active_channels))
        ]
        X_i = build_node_feature_matrix(per_channel_pooled)

        if frozen_connectivity:
            L_norm_i = frozen_L_norm
        else:
            epoch_signal = epoch_data_all[i]
            L_norm_i = _compute_laplacian(epoch_signal)

        dataset.append((X_i, L_norm_i))

    if return_epochs:
        return dataset, epochs
    return dataset
