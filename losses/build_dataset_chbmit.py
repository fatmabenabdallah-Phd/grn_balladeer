"""
grn_balladeer.data.build_dataset_chbmit
===========================================
Third domain of the architectural-generality study (after ADHD/
BALLADEER+Nasrabadi and motor-imagery/BCI-IV-2a): epileptic seizure
detection (ictal vs interictal) on CHB-MIT, a spatio-spectral signature
sharing nothing with either prior domain -- broadband rhythmic
discharges, no fixed hemispheric cluster like C3/C4 or a literature-
anchored frontal cluster like ADHD's theta-gamma coupling (seizure
onset zone is patient-specific, not scalp-fixed -- see
losses.epilepsy_channels for how this is handled, deliberately
differently from the previous two domains).

DATA SOURCE: PhysioNet (physionet.org/content/chbmit/1.0.0/), NOT
reachable from a restricted sandbox network (verified this session,
same host_not_allowed pattern as bbci.de for BCI IV 2a) but reachable
from Colab via plain wget -- no GitHub mirror was needed or used here,
unlike BCI IV 2a.

VERIFIED THIS SESSION (real .edf files, not assumed from documentation):
- 23 raw channels per file, but channels 19-23 (P7-T7, T7-FT9, FT9-FT10,
  FT10-T8, T8-P8) duplicate/invert channels 3 and 15 (T8-P8 appears
  twice -- literally the same signal twice; P7-T7/T7-P7 are the same
  electrode pair with reversed polarity, a trivial ~180-degree phase
  relationship, not real synchrony). Confirmed via direct MNE loading
  of chb01_01.edf: MNE silently renames the literal duplicate to
  'T8-P8-0'/'T8-P8-1' rather than raising -- do not include channels
  19-23 in the graph, both to avoid this redundancy and to avoid
  feeding L_harm/L_symb pairs with trivial, non-physiological PLV=1
  relationships by construction.
- CHBMIT_CHANNELS (18 unique, kept) below is exactly channels 1-18
  from every conforming case's *-summary.txt (identical across chb01,
  chb09, chb23, individually verified).
- chb12_27/28/29 are NOT montage variants in the ordinary sense --
  direct MNE loading of chb12_27.edf shows a completely different
  referencing scheme (channels named '<electrode>-CS2', a common-
  reference montage, not longitudinal bipolar) plus 5 channels with
  literally no scaling factor defined (empty/unusable). Excluded
  entirely, consistent with independent published practice (Chung et
  al. 2024, Frontiers in Neurology, excluded the same 3 files for the
  same reason, losing 13 seizures).
- Seizure timing in *-summary.txt is sample-accurate: chb01_03.edf's
  annotated [2996s, 3036s] window, extracted via raw[:, start:end],
  gives exactly 10240 samples = 40.0s at 256Hz -- confirmed directly,
  not assumed.
- chb09's summary file contains a mid-case 'Channels changed:' block
  -- a single case is not guaranteed to have a constant channel set
  throughout; per-file channel validation (not just per-case) is
  required, see build_subject_dataset_chbmit's channel check.
- chb01 and chb21 are the SAME subject (chb21 recorded 1.5 years
  later) -- must be merged for subject-disjoint splitting, never
  treated as 2 independent subjects. See data.labels_chbmit.

NOT yet verified: the full set of 23 cases' summary files (only chb01,
chb09, chb12, chb23 individually checked); binary .edf content for
cases beyond chb01/chb12 (network-constrained in this development
session, first full run happens on Colab).
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

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

# Channels 1-18 only (see module docstring for why 19-23 are dropped).
CHBMIT_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3",
    "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8",
    "T8-P8", "P8-O2", "FZ-CZ", "CZ-PZ",
]

SFREQ_HZ = 256.0

# Confirmed non-conforming files (common-reference montage, not
# longitudinal bipolar; 5 channels with no scaling factor) -- excluded
# entirely, not just down-weighted. Keys are (case_id, file_stem)
# without extension, matching how file names are parsed below.
EXCLUDED_FILES = {
    ("chb12", "chb12_27"),
    ("chb12", "chb12_28"),
    ("chb12", "chb12_29"),
}


def parse_summary_file(summary_text: str) -> List[Dict]:
    """Parses a chbNN-summary.txt file's content into a list of
    per-file records: {'file_name': str, 'seizures': [(start_s, end_s), ...]}.

    Handles BOTH summary formats seen in this dataset (verified on
    real files this session):
      - Single-seizure files: 'Seizure Start Time: NNNN seconds' /
        'Seizure End Time: NNNN seconds' (e.g. chb01).
      - Multi-seizure files: 'Seizure 1 Start Time: ...' / 'Seizure 1
        End Time: ...', 'Seizure 2 Start Time: ...', etc. (e.g.
        chb12_06.edf, which has 2 seizures).

    A file with 'Number of Seizures in File: 0' has an empty seizures
    list, not a missing key -- callers should treat an empty list as
    "usable as an interictal source", not as a parsing failure.
    """
    records = []
    file_blocks = summary_text.split("File Name: ")[1:]  # first split chunk is the header, discard

    for block in file_blocks:
        file_name = block.split("\n", 1)[0].strip()
        n_seizures_match = re.search(r"Number of Seizures in File:\s*(\d+)", block)
        n_seizures = int(n_seizures_match.group(1)) if n_seizures_match else 0

        seizures: List[Tuple[float, float]] = []
        if n_seizures > 0:
            # Multi-seizure format: "Seizure N Start/End Time"
            numbered = re.findall(
                r"Seizure\s+\d+\s+Start Time:\s*(\d+)\s*seconds\s*Seizure\s+\d+\s+End Time:\s*(\d+)\s*seconds",
                block,
            )
            if numbered:
                seizures = [(float(s), float(e)) for s, e in numbered]
            else:
                # Single-seizure format: "Seizure Start Time" / "Seizure End Time"
                single = re.findall(
                    r"Seizure Start Time:\s*(\d+)\s*seconds\s*Seizure End Time:\s*(\d+)\s*seconds",
                    block,
                )
                seizures = [(float(s), float(e)) for s, e in single]

            if len(seizures) != n_seizures:
                raise ValueError(
                    f"parse_summary_file: {file_name} declares {n_seizures} seizures "
                    f"but {len(seizures)} were parsed -- format assumption may not hold "
                    f"for this file, do not trust the result without investigating."
                )

        records.append({"file_name": file_name, "seizures": seizures})

    return records


def extract_binary_segments(
    channel_data: np.ndarray,
    seizures: List[Tuple[float, float]],
    sfreq: float = SFREQ_HZ,
    window_s: float = 4.0,
    n_interictal_per_file: int = 4,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts fixed-length, non-overlapping ictal windows from within
    each annotated seizure, and an equal-ISH number of interictal
    windows sampled from the parts of the recording OUTSIDE any
    seizure (with a margin, see below).

    channel_data: (n_channels, n_samples) for ONE continuous .edf file.
    seizures: list of (start_s, end_s) from parse_summary_file, for
    THIS file (empty list if the file has no seizure -- in that case
    only interictal windows are drawn, up to n_interictal_per_file).

    Margin: interictal windows are drawn only from samples at least
    `window_s` away from any seizure boundary, avoiding pre-ictal/
    post-ictal transition periods being mislabeled as clean interictal
    -- a real risk given seizure onset is gradual, not instantaneous.

    Returns (windows, labels): windows shape (n_windows, n_channels,
    window_samples), labels shape (n_windows,), 1=ictal/0=interictal.
    Raises if a declared seizure window is out of the recording's
    bounds -- this would indicate a parsing or alignment error, not
    something to silently skip.
    """
    n_channels, n_samples = channel_data.shape
    window_samples = int(round(window_s * sfreq))
    margin_samples = window_samples

    rng = np.random.default_rng(seed)
    windows: List[np.ndarray] = []
    labels: List[int] = []

    occupied = np.zeros(n_samples, dtype=bool)  # marks ictal + margin, for interictal exclusion

    for start_s, end_s in seizures:
        start_sample = int(round(start_s * sfreq))
        end_sample = int(round(end_s * sfreq))
        if end_sample > n_samples or start_sample < 0:
            raise ValueError(
                f"extract_binary_segments: seizure [{start_s}, {end_s}]s out of bounds "
                f"for a recording of {n_samples / sfreq:.1f}s -- check alignment."
            )
        n_ictal_windows = (end_sample - start_sample) // window_samples
        for w in range(n_ictal_windows):
            w_start = start_sample + w * window_samples
            windows.append(channel_data[:, w_start:w_start + window_samples])
            labels.append(1)

        margin_start = max(0, start_sample - margin_samples)
        margin_end = min(n_samples, end_sample + margin_samples)
        occupied[margin_start:margin_end] = True

    free_starts = np.where(~occupied[:n_samples - window_samples])[0]
    if len(free_starts) > 0:
        n_draw = min(n_interictal_per_file, len(free_starts) // window_samples)
        if n_draw > 0:
            chosen_starts = rng.choice(free_starts, size=min(n_draw * 5, len(free_starts)), replace=False)
            drawn = 0
            for s in chosen_starts:
                if drawn >= n_draw:
                    break
                if not occupied[s:s + window_samples].any():
                    windows.append(channel_data[:, s:s + window_samples])
                    labels.append(0)
                    occupied[s:s + window_samples] = True  # avoid overlapping draws
                    drawn += 1

    if not windows:
        return np.empty((0, n_channels, window_samples)), np.empty((0,), dtype=int)

    return np.stack(windows), np.array(labels, dtype=int)


def build_subject_dataset_chbmit(
    channel_data: np.ndarray,
    seizures: List[Tuple[float, float]],
    sfreq: float = SFREQ_HZ,
    window_s: float = 4.0,
    n_interictal_per_file: int = 4,
    bands: List[Tuple[float, float]] = ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)),
    hop_length: int = 32,
    connectivity_metric: str = "plv",
    seed: int = 42,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], np.ndarray]:
    """Builds GRN-ready (X_i, L_norm_i) graphs for ONE .edf file's
    ictal/interictal windows. channel_data must already be restricted
    to CHBMIT_CHANNELS (18 channels), in that exact order -- callers
    are responsible for this selection (per-file, since chb09-style
    mid-case channel changes mean channel order/presence cannot be
    assumed constant across an entire case).

    bands: default spans delta through low-gamma (1-45Hz) -- broader
    than either ADHD's theta/alpha/beta or motor-imagery's mu/beta,
    deliberately: ictal discharges are broadband and not confined to a
    single canonical band the way ADHD's theta/beta ratio or motor
    imagery's mu/beta ERD are. Narrowing this without a specific
    literature anchor would silently import an assumption from the
    wrong domain, the same mistake flagged for L_symb in
    losses.epilepsy_channels.

    Returns (dataset, labels), same paired-list convention as
    build_subject_dataset_biciv2a: dataset is a list of (X_i, L_norm_i)
    graphs, labels is the aligned (n_windows,) 0/1 array.
    """
    if channel_data.shape[0] != len(CHBMIT_CHANNELS):
        raise ValueError(
            f"build_subject_dataset_chbmit: expected {len(CHBMIT_CHANNELS)} channels "
            f"(CHBMIT_CHANNELS), got {channel_data.shape[0]} -- check channel selection/order."
        )

    windows, labels = extract_binary_segments(
        channel_data, seizures, sfreq=sfreq, window_s=window_s,
        n_interictal_per_file=n_interictal_per_file, seed=seed,
    )
    if len(windows) == 0:
        return [], labels

    n_channels = len(CHBMIT_CHANNELS)
    if connectivity_metric == "plv":
        strength_fn = compute_plv_matrix
    elif connectivity_metric == "pli":
        strength_fn = compute_pli_matrix
    else:
        raise ValueError(f"connectivity_metric must be 'plv' or 'pli', got '{connectivity_metric}'")

    def _compute_laplacian(signal_segment: np.ndarray) -> torch.Tensor:
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

    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for window_signal in windows:
        window_t = torch.from_numpy(window_signal).float()
        cqt_per_channel = [
            compute_cqt_features(window_t[ci], sfreq=sfreq, hop_length=hop_length)
            for ci in range(n_channels)
        ]
        per_channel_pooled = [
            pool_cqt_to_node_features(
                cqt_per_channel[ci], 0, sfreq, hop_length, tmin=0.0, tmax=window_s,
            )
            for ci in range(n_channels)
        ]
        X_i = build_node_feature_matrix(per_channel_pooled)
        L_norm_i = _compute_laplacian(window_signal)
        dataset.append((X_i, L_norm_i))

    return dataset, labels
