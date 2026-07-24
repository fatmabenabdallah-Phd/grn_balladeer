"""
grn_balladeer.model.tcn_encoder
==================================
Lightweight Temporal Convolutional Network (TCN) encoder + fixed
structural graph aggregation -- an edge-deployment-oriented alternative
to GRN's magnetic-Laplacian encoder, motivated directly by this
session's findings: (1) GRN did not beat a classical Random Forest
baseline on band-power features (0.517 vs 0.668 AUC), and (2)
ICCCI2026's own complexity analysis (Table 1) identifies TCN + static
structural graph as the best latency/parameter/accuracy trade-off among
surveyed architectures (8.2ms FP32 / 2.1ms INT8 latency, ~15k
parameters, vs. 1.8GB/60ms+ for attention-based alternatives).

Design choices, each with a one-line rationale:
- Dilated CAUSAL 1D convolutions (not LSTM): parallelizable, O(T)
  complexity, no sequential bottleneck.
- SHARED weights across channels/nodes (not one TCN per channel):
  keeps parameter count small regardless of montage size (30 channels
  here) -- this is what makes the ~15k-parameter budget achievable.
- Structural graph aggregation (fixed k-NN, precomputed ONCE via
  connectivity/structural_graph.py) instead of magnetic Laplacian:
  near-zero adjacency cost at inference, a deliberate trade validated
  by this session's own PLV-vs-PLI ablation showing no measurable
  accuracy cost to simplifying connectivity on this cohort.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """1D convolution with causal (left-only) padding, so output at
    time t depends only on inputs at time <= t -- required for the TCN
    receptive-field construction to be meaningful for streaming/
    real-time use (per ICCCI2026's Sec. 4.2 dilated-causal-conv design).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """One dilated causal conv layer + ReLU + (light) residual connection."""

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x)) + x  # residual, keeps gradient flow shallow-network-friendly


class LightweightTCNEncoder(nn.Module):
    """Encodes raw per-node (per-electrode) EEG time series into a
    small per-node embedding, using a SHARED (channel-agnostic) TCN
    stack, then aggregates neighbor embeddings via one fixed structural
    graph convolution step (mean aggregation over k-NN neighbors,
    normalized adjacency -- see connectivity/structural_graph.py).

    Parameter budget target: with hidden_channels=8, n_layers=4,
    kernel_size=3, this stack has on the order of a few thousand
    parameters (verify via count_parameters below) -- comparable to
    ICCCI2026's cited ~15k-parameter TCN-GNN benchmark, not the
    substantially larger GRN encoder (magnetic Laplacian stack +
    resonance head) used earlier this session.
    """

    def __init__(self, hidden_channels: int = 8, n_layers: int = 4, kernel_size: int = 3):
        super().__init__()
        self.input_proj = nn.Conv1d(1, hidden_channels, kernel_size=1)  # 1 raw channel -> hidden_channels
        self.blocks = nn.ModuleList([
            TCNBlock(hidden_channels, kernel_size, dilation=2 ** i)
            for i in range(n_layers)
        ])
        self.hidden_channels = hidden_channels

    def forward(self, x: torch.Tensor, adjacency_norm: torch.Tensor) -> torch.Tensor:
        """x: (B, N, T) real-valued raw per-node time series (N = number
        of electrodes, T = number of timepoints in the epoch).
        adjacency_norm: (N, N) precomputed, normalized structural
        adjacency (same for every sample -- NOT epoch-dependent, unlike
        GRN's per-epoch PLV/magnetic Laplacian).

        Returns (B, N, hidden_channels): per-node embeddings after one
        structural graph aggregation step.
        """
        B, N, T = x.shape
        x = x.reshape(B * N, 1, T)          # treat each (sample, node) as its own 1-channel sequence
        x = self.input_proj(x)               # (B*N, hidden_channels, T)
        for block in self.blocks:
            x = block(x)
        node_embeddings = x.mean(dim=-1)     # (B*N, hidden_channels) -- temporal pooling
        node_embeddings = node_embeddings.reshape(B, N, self.hidden_channels)

        # ONE structural graph aggregation step: node embeddings become a
        # normalized-adjacency-weighted average of their k-NN neighbors'
        # embeddings (standard GCN propagation, Kipf & Welling 2017) --
        # deliberately shallow (one hop) to keep inference cost minimal.
        aggregated = torch.einsum("ij,bjc->bic", adjacency_norm, node_embeddings)
        return aggregated


def count_parameters(model: nn.Module) -> int:
    """Total trainable parameter count -- report this directly in the
    paper's edge-deployment comparison table (Section on efficiency
    metrics), rather than only citing FLOPs/latency without the
    parameter count ICCCI2026's own Table 1 reports alongside them."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
