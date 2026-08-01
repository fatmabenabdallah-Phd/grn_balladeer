"""
grn_balladeer.losses.motor_imagery_channels
===============================================
L_symb cluster definition for the motor-imagery domain (BCI IV 2a),
parallel to losses.symbolic_loss's FRONTAL_CHANNELS but grounded in a
completely different literature -- this is the condition-(b) arm of
the architectural-generality study (see
contexte_GRN_generalite_architecturale.md's "point critique"): testing
GRN's full neurosymbolic mechanism with a cluster that is actually
justified for THIS task, rather than reusing the ADHD frontal cluster
out of convenience.

MOTOR_CHANNELS is centered on C3/C4 (left/right primary sensorimotor
cortex), the well-established site of event-related
desynchronization/synchronization (ERD/ERS) in the mu (8-12Hz) and beta
(16-24Hz) rhythms during left- vs right-hand motor imagery:

  Pfurtscheller, G., & Lopes da Silva, F. H. (1999). Event-related
  EEG/MEG synchronization and desynchronization: basic principles.
  Clinical Neurophysiology, 110(11), 1842-1857.
  (General ERD/ERS principles; contralateral mu/beta desynchronization
  over sensorimotor cortex during movement/motor imagery.)

  Pfurtscheller, G., & Neuper, C. (2001). Motor imagery and direct
  brain-computer communication. Proceedings of the IEEE, 89(7),
  1123-1134.
  (Motor imagery specifically -- C3/C4 asymmetry as the basis for
  left/right hand BCI classification; this is the paper the BCI
  Competition IV 2a paradigm itself descends from.)

Both references independently verified against multiple citing sources
(PubMed, ScienceDirect, and independent bibliography entries in later
papers) before use here -- per this project's standing rule to check
every reference individually rather than trust it by association.

NOTE ON REUSE: get_frontal_pairs (losses.symbolic_loss) is already
written generically -- it takes a `frontal_channels` argument, it does
not hard-code the ADHD cluster. It is reused UNCHANGED here rather than
duplicated; only the channel list passed to it differs. This module
exists solely to hold that channel list and its literature
justification, not to reimplement pair-finding logic.
"""

from __future__ import annotations

from typing import List

import torch

from grn_balladeer.losses.symbolic_loss import get_frontal_pairs

MOTOR_CHANNELS: List[str] = ["C3", "C4"]


def get_motor_pairs(ch_names: List[str], motor_channels: List[str] = MOTOR_CHANNELS) -> torch.Tensor:
    """Returns (n_pairs, 2) long tensor of (i, j) index pairs for the
    motor-imagery cluster, restricted to channels actually present in
    ch_names. Thin wrapper around get_frontal_pairs (no logic
    duplicated) -- named separately only so callers building the
    motor-imagery L_symb do not have to import a function whose name
    ("frontal") is misleading in this context.

    With only 2 channels (C3, C4), this yields exactly 1 pair -- unlike
    the ADHD frontal cluster's 5 channels / 10 pairs. This is expected
    and not an error: the motor-imagery literature's asymmetry claim is
    specifically about the C3-C4 relationship, not a larger regional
    cluster, so a single pair is the correct, literature-faithful
    constraint here, not an impoverished version of the ADHD case.
    """
    return get_frontal_pairs(ch_names, frontal_channels=motor_channels)
