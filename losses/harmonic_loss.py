"""
grn_balladeer.losses.harmonic_loss
======================================
Module 7 — harmonic loss (Option A: learned omega_i from
extract_resonance_frequency) and the shared consonance-degree helper
reused by both the hard loss here and the soft symbolic loss.

WORKING ASSUMPTION flagged for confirmation: CONSONANCE_RATIOS below is
a standard musical-consonance set (unison, major third, fourth, fifth,
octave). It has not yet been cross-checked against a specific literature
source (e.g. Sanchis et al. Heliyon 2024) for this exact list/ordering -
treat as a placeholder until confirmed, not as a validated constant.
"""

from __future__ import annotations

from typing import List, Tuple

import torch

CONSONANCE_RATIOS: List[float] = [1.0, 1.25, 1.333, 1.5, 2.0]


def compute_consonance_degree(
    omega_i: torch.Tensor,
    omega_j: torch.Tensor,
    ratios: List[float] = CONSONANCE_RATIOS,
    sigma: float = 0.1,
) -> torch.Tensor:
    """Soft consonance degree mu_ij = exp(-min_k|omega_i/omega_j - rho_k|^2 / sigma^2).

    Same functional form as model.grn_encoder.fixed_consonance_prior
    (Option B) - kept as a separate function here since Option A operates
    on LEARNED omega_i (from extract_resonance_frequency) rather than
    fixed known frequencies. If this duplication becomes a maintenance
    issue, fixed_consonance_prior could import this function instead of
    re-implementing it - not done here to avoid touching Week 3's
    already-validated model/grn_encoder.py.

    omega_i, omega_j: (n_pairs,) real tensors. Returns (n_pairs,) in [0, 1].
    """
    ratio = omega_i / omega_j
    ratios_t = torch.tensor(ratios, dtype=ratio.dtype, device=ratio.device)
    diffs_sq = (ratio.unsqueeze(-1) - ratios_t) ** 2  # (n_pairs, n_ratios)
    min_diff_sq = diffs_sq.min(dim=-1).values
    return torch.exp(-min_diff_sq / (sigma**2))


def harmonic_loss(
    omega: torch.Tensor,
    edge_pairs: torch.Tensor,
    ratios: List[float] = CONSONANCE_RATIOS,
    reduction: str = "mean",
) -> torch.Tensor:
    """L_harm = reduce_{(i,j) in E} min_k |omega_i/omega_j - rho_k|^2.

    Hard quantization distance (no exp/sigma - contrast with
    compute_consonance_degree's soft version used in the symbolic loss).

    omega: (n_nodes,) real tensor, one resonance frequency per node
        (e.g. from extract_resonance_frequency).
    edge_pairs: (n_edges, 2) long tensor of (i, j) node index pairs
        defining which pairs are constrained by this loss.
    reduction: 'mean' (default, scale-invariant to graph/edge-set size)
        or 'sum'.
    """
    if edge_pairs.numel() == 0:
        raise ValueError("harmonic_loss: edge_pairs is empty - nothing to constrain.")

    i_idx = edge_pairs[:, 0]
    j_idx = edge_pairs[:, 1]
    ratio = omega[i_idx] / omega[j_idx]

    ratios_t = torch.tensor(ratios, dtype=ratio.dtype, device=ratio.device)
    diffs_sq = (ratio.unsqueeze(-1) - ratios_t) ** 2  # (n_edges, n_ratios)
    min_diff_sq = diffs_sq.min(dim=-1).values  # (n_edges,)

    if reduction == "mean":
        return min_diff_sq.mean()
    elif reduction == "sum":
        return min_diff_sq.sum()
    else:
        raise ValueError(f"harmonic_loss: unknown reduction '{reduction}', expected 'mean'/'sum'")


def all_pairs_edge_index(n_nodes: int) -> torch.Tensor:
    """Convenience helper: (i, j) for all i != j, as an (n_pairs, 2)
    long tensor. Useful when the harmonic constraint applies to every
    node pair rather than a restricted subset (contrast with the
    symbolic loss, which restricts to frontal pairs only)."""
    idx = torch.combinations(torch.arange(n_nodes), r=2)
    return idx
