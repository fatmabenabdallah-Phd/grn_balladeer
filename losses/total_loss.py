"""
grn_balladeer.losses.total_loss
===================================
Module 7 — combines task, harmonic, symbolic, and (from Week 5) triplet
losses into a single scalar. L_triplet defaults to 0.0 since Module 7b
(triplet mining/loss) is scheduled for Week 5 - total_loss is usable
now for the EEG-only training loop (Week 4, Module 9) and simply gains
a nonzero triplet term once Module 7b lands, no signature change needed
beyond passing a real tensor instead of the default.
"""

from __future__ import annotations

import torch


def total_loss(
    l_task: torch.Tensor,
    l_harm: torch.Tensor,
    l_symb: torch.Tensor,
    l_triplet: torch.Tensor = torch.tensor(0.0),
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    lambda3: float = 1.0,
) -> torch.Tensor:
    """L_total = L_task + lambda1*L_harm + lambda2*L_symb + lambda3*L_triplet.

    lambda1/2/3 are working defaults (1.0 each) - not yet tuned; expect
    to sweep these once a first EEG-only training run (Week 4 milestone)
    gives a baseline to compare against.
    """
    return l_task + lambda1 * l_harm + lambda2 * l_symb + lambda3 * l_triplet
