"""
grn_balladeer.losses.epilepsy_channels
==========================================
L_symb cluster definition for the epilepsy domain (CHB-MIT), parallel
to losses.symbolic_loss's FRONTAL_CHANNELS (ADHD) and
losses.motor_imagery_channels' MOTOR_CHANNELS (BCI IV 2a) -- but with a
materially weaker literature anchor than either, documented explicitly
rather than glossed over.

UNLIKE the previous two domains, epilepsy has no fixed scalp cluster
analogous to C3/C4 (motor imagery) or the ADHD frontal theta-gamma
cluster: the seizure onset zone is PATIENT-SPECIFIC (temporal, frontal,
parietal, or occipital depending on the individual), not determined by
the recording montage. CHB-MIT's own public metadata
(chbNN-summary.txt) does not label a per-patient onset zone -- a
third-party study (Chung et al., 2024, Frontiers in Neurology) had
neurologists review recordings specifically to add this, work not
available here.

EPILEPSY_CHANNELS = ["FP1-F7", "FP2-F8"] is the best GENERIC anchor
found, not a patient-specific one: anterior temporal derivations
(F7/F8 in the 10-20 system) are repeatedly cited across independent
clinical sources as the single most common site of epileptiform
discharges in scalp EEG generally, with well over 90% concordance
specifically for temporal-lobe epilepsy -- but explicitly NOT
universal (frontal, parietal, and occipital epilepsies are documented
too, typically with fewer scalp-detectable discharges). Given this
mismatch between a population-level anchor and a per-patient
phenomenon, this project deliberately tests BOTH conditions as an
ablation (lambda2=1 with this cluster vs. lambda2=0, L_symb disabled
entirely) rather than presenting either as the "correct" choice --
see training.cross_validation_chbmit for how both are run.

Sources (population-level anchor, not per-patient): Tatum, W.O.,
"Abnormal EEG: Epileptiform" (clinical reference text) and Ebersole &
Pedley-style epileptiform-discharge overviews independently describe
F7/F8 maximal negativity as characteristic of anterior temporal
discharges, the most frequently identified focal interictal pattern in
scalp EEG.

Both channels (FP1-F7, FP2-F8) are present verbatim in
data.build_dataset_chbmit.CHBMIT_CHANNELS (they are two of the 18
canonical bipolar derivations, not electrode names requiring
translation) -- no montage-mismatch risk of the kind found for the
ADHD frontal cluster on BCI IV 2a (only 1/5 channels present there).

get_frontal_pairs (losses.symbolic_loss) is reused unchanged, same
convention as motor_imagery_channels.get_motor_pairs.
"""

from __future__ import annotations

from typing import List

import torch

from grn_balladeer.losses.symbolic_loss import get_frontal_pairs

EPILEPSY_CHANNELS: List[str] = ["FP1-F7", "FP2-F8"]


def get_epilepsy_pairs(ch_names: List[str], epilepsy_channels: List[str] = EPILEPSY_CHANNELS) -> torch.Tensor:
    """Returns (n_pairs, 2) long tensor of (i, j) index pairs for the
    generic anterior-temporal cluster, restricted to channels actually
    present in ch_names. With only 2 channels, this yields exactly 1
    pair -- same structural situation as motor_imagery_channels'
    get_motor_pairs (C3/C4), for the same reason: the underlying
    literature claim here is specifically about an FP1-F7/FP2-F8
    asymmetry, not a larger regional cluster.
    """
    return get_frontal_pairs(ch_names, frontal_channels=epilepsy_channels)
