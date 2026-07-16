"""
grn_balladeer.losses.symbolic_loss
======================================
Module 7 — fuzzy symbolic implication loss (L_symb), restricted to
frontal channel pairs (per the Week 2 PLV heatmap finding: the frontal
cluster AF7/Fp1/Fpz/F7/Fz showed PLV up to 0.77 on real UB0136 data,
which is why L_symb targets this specific cluster rather than all pairs).
"""

from __future__ import annotations

from typing import List, Tuple

import torch

FRONTAL_CHANNELS: List[str] = ["AF7", "Fp1", "Fpz", "F7", "Fz"]


def get_frontal_pairs(ch_names: List[str], frontal_channels: List[str] = FRONTAL_CHANNELS) -> torch.Tensor:
    """Returns (n_pairs, 2) long tensor of (i, j) index pairs (i < j),
    restricted to channels in frontal_channels that are actually present
    in ch_names. Raises if fewer than 2 frontal channels are found (the
    loss needs at least one pair)."""
    frontal_idx = [ch_names.index(c) for c in frontal_channels if c in ch_names]
    if len(frontal_idx) < 2:
        raise ValueError(
            f"get_frontal_pairs: only {len(frontal_idx)} of {frontal_channels} found in "
            f"ch_names - need at least 2 to form a pair."
        )
    pairs = [(i, j) for a, i in enumerate(frontal_idx) for j in frontal_idx[a + 1 :]]
    return torch.tensor(pairs, dtype=torch.long)


def determine_rule_direction(mu_values: torch.Tensor, labels: torch.Tensor) -> Tuple[str, float]:
    """EMPIRICAL DIAGNOSTIC - run this BEFORE fixing the symbolic rule's
    direction (per the plan's explicit warning). Determines whether high
    consonance degree (mu) empirically co-occurs with label=1 ('direct')
    or label=0 ('inverse'), via Pearson correlation between mu_values and
    labels across samples/subjects.

    mu_values: (n_samples,) real tensor - e.g. mean frontal consonance
        degree per subject/epoch.
    labels: (n_samples,) tensor of 0/1 (e.g. 0=control, 1=ADHD - confirm
        against data.labels.GROUP_MAP/diagnosed convention before use).

    Returns (direction, correlation) where direction is 'direct' if
    corr > 0 else 'inverse'. Does NOT decide the loss formula itself -
    symbolic_implication_loss's direction argument should be set from
    this function's output, not guessed.

    CAVEAT: correlation over a single subject or a handful of epochs
    from one subject is not a valid empirical basis for this decision -
    it needs a real cross-subject sample (multiple ADHD + multiple
    control subjects). Only real data currently available in this
    conversation is a single subject (UB0136), so this function is
    validated below only on synthetic data with known ground-truth
    correlation, not yet on real BALLADEER labels.
    """
    if mu_values.shape[0] < 2:
        raise ValueError("determine_rule_direction: need at least 2 samples to compute a correlation.")
    mu_c = mu_values - mu_values.mean()
    y_c = labels.float() - labels.float().mean()
    denom = torch.sqrt((mu_c**2).sum() * (y_c**2).sum())
    if denom == 0:
        raise ValueError("determine_rule_direction: zero variance in mu_values or labels - correlation undefined.")
    corr = (mu_c * y_c).sum() / denom
    direction = "direct" if corr.item() > 0 else "inverse"
    return direction, corr.item()


def symbolic_implication_loss(
    mu_ij: torch.Tensor, confidence: torch.Tensor, direction: str = "direct"
) -> torch.Tensor:
    """L_symb = mean(1 - truth_ij), where:
      truth_ij = 1 - mu_ij + mu_ij * confidence          if direction == 'direct'
      truth_ij = 1 - mu_ij + mu_ij * (1 - confidence)    if direction == 'inverse'

    Fuzzy implication (mu_ij => confidence), consonance implies model
    confidence in the direction empirically determined by
    determine_rule_direction. mu_ij: (n_pairs,) in [0, 1], from
    compute_consonance_degree restricted to frontal pairs. confidence:
    scalar or (n_pairs,) broadcastable - the classification head's
    predicted probability for the correct class.
    """
    if direction == "direct":
        truth = 1 - mu_ij + mu_ij * confidence
    elif direction == "inverse":
        truth = 1 - mu_ij + mu_ij * (1 - confidence)
    else:
        raise ValueError(f"symbolic_implication_loss: direction must be 'direct'/'inverse', got '{direction}'")

    return (1 - truth).mean()
