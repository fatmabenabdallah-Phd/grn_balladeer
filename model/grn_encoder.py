"""
grn_balladeer.model.grn_encoder
==================================
Module 5 (continued) — GRNEncoder (stacks MagneticLaplacianConv layers)
and extract_resonance_frequency (the learnable g_theta head, Option A
of the harmonic loss design).
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from grn_balladeer.model.magnetic_laplacian_conv import MagneticLaplacianConv


class GRNEncoder(nn.Module):
    """Stacks 2-3 MagneticLaplacianConv layers (all sharing the same
    L_norm, recomputed once per forward call at the call site — the
    graph topology is fixed per epoch, only node features change layer
    to layer). Output: h_i, complex embedding per node.

    hidden_channels: list of output sizes for each layer, e.g. [64, 32]
    for 2 layers (in_channels -> 64 -> 32), or [64, 64, 32] for 3.
    The last layer has activation=False by default (raw complex
    embedding passed on to extract_resonance_frequency / the
    classification head — no CReLU truncating the final representation).
    """

    def __init__(self, in_channels: int, hidden_channels: List[int], K: int = 3):
        super().__init__()
        if not hidden_channels:
            raise ValueError("GRNEncoder: hidden_channels must have at least one entry")

        dims = [in_channels] + list(hidden_channels)
        layers = []
        for i in range(len(dims) - 1):
            is_last = i == len(dims) - 2
            layers.append(
                MagneticLaplacianConv(dims[i], dims[i + 1], K=K, activation=not is_last)
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, X: torch.Tensor, L_norm: torch.Tensor) -> torch.Tensor:
        h = X
        for layer in self.layers:
            h = layer(h, L_norm)
        return h


def extract_resonance_frequency(
    h: torch.Tensor, head: nn.Module, omega_min: float = 1.0, omega_max: float = 45.0
) -> torch.Tensor:
    """Applies the learnable g_theta head to per-node complex embeddings,
    producing a scalar resonance frequency omega_i per node. h: (n_nodes,
    d) complex. head: an nn.Linear(2*d, 1) (real-valued) — Re/Im parts
    are concatenated before the linear layer, since a plain nn.Linear
    does not accept complex input directly. Returns (n_nodes,) real,
    bounded to [omega_min, omega_max].

    BUG FIX (found empirically this session, real UB0136 training run):
    the head's raw linear output is unconstrained and can drift toward 0
    during training (observed: min|omega| going 0.019 -> 0.010 over a
    few epochs). Since harmonic_loss/compute_consonance_degree compute
    omega_i/omega_j, an omega_j approaching 0 makes that ratio explode
    towards infinity - loss_harm was observed reaching >4000 within a
    handful of epochs. Gradient clipping alone does NOT fix this: it
    bounds the gradient STEP size, not the forward-pass value once
    omega is already near 0 - the explosion recurred (worse: never
    recovered) even with clip_grad_norm_(max_norm=1.0) applied, under a
    different real training run/seed. Fix: squash the raw linear output
    through a sigmoid and rescale to [omega_min, omega_max] (default
    1-45 Hz, matching the EEG bandpass range already used throughout
    preprocessing/filtering.py) - omega can now never reach 0 or diverge,
    by construction, regardless of what the raw linear layer outputs.

    The head is passed in rather than constructed here so its parameters
    are owned and trained alongside the rest of the model (GRNEncoder +
    this head trained jointly) — see build_resonance_head() for the
    matching constructor.
    """
    h_concat = torch.cat([h.real, h.imag], dim=-1)  # (n_nodes, 2*d)
    raw = head(h_concat).squeeze(-1)  # unconstrained, (n_nodes,)
    return omega_min + torch.sigmoid(raw) * (omega_max - omega_min)


def build_resonance_head(embedding_dim: int) -> nn.Linear:
    """Constructs the g_theta head matching extract_resonance_frequency's
    expected input size (2*embedding_dim, from concatenated Re/Im). The
    raw nn.Linear output is unconstrained by design - extract_resonance_
    frequency applies the bounding sigmoid, not this constructor, so the
    same head can in principle be reused with different omega_min/max
    without rebuilding it."""
    return nn.Linear(2 * embedding_dim, 1)


def fixed_consonance_prior(
    f_i: torch.Tensor, f_j: torch.Tensor, ratios: List[float], sigma: float
) -> torch.Tensor:
    """Option B (ablation counterpart to the learned omega_i from
    extract_resonance_frequency): injects a FIXED, non-learned
    consonance prior directly into the edge weighting, using known
    frequencies (e.g. EEG band center frequencies) instead of a
    learned g_theta output.

    Same functional form as Module 7's compute_consonance_degree:
    mu_ij = exp(-min_k|f_i/f_j - rho_k|^2 / sigma^2). f_i, f_j: (n_pairs,)
    real tensors of fixed frequencies. Returns (n_pairs,) in [0, 1].
    """
    ratio = f_i / f_j
    ratios_t = torch.tensor(ratios, dtype=ratio.dtype, device=ratio.device)
    diffs_sq = (ratio.unsqueeze(-1) - ratios_t) ** 2  # (n_pairs, n_ratios)
    min_diff_sq = diffs_sq.min(dim=-1).values
    return torch.exp(-min_diff_sq / (sigma**2))
