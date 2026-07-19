"""
grn_balladeer.model.magnetic_laplacian_conv
===============================================
Module 5 — MagneticLaplacianConv, the GRN's custom complex-valued
message-passing layer, adapted from MagNet (Zhang et al. 2021).
Chebyshev polynomial approximation of the magnetic Laplacian spectral
convolution — NOT a full eigendecomposition at every forward pass (that
would be prohibitively slow to backprop through repeatedly).

Stays in full complex64/float32 throughout (no mixed precision on this
layer, per project decision — complex-valued autograd is already
delicate enough without half precision).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def compute_normalized_laplacian(L_C: torch.Tensor) -> torch.Tensor:
    """Rescales the magnetic Laplacian's eigenvalues into [-1, 1], the
    range Chebyshev polynomials are defined/stable over:
    L_norm = 2*L_C/lambda_max - I.

    L_C is complex Hermitian (by construction — see build_magnetic_laplacian
    in Module 3), so its eigenvalues are guaranteed real; lambda_max is
    obtained via torch.linalg.eigvalsh (exact, cheap at graph sizes here
    — 30 EEG channels — no need for power-iteration approximation).
    """
    if L_C.shape[0] != L_C.shape[1]:
        raise ValueError(f"compute_normalized_laplacian: L_C must be square, got {tuple(L_C.shape)}")
    eigenvalues = torch.linalg.eigvalsh(L_C)
    lambda_max = eigenvalues.max().real
    if lambda_max <= 0:
        raise ValueError(f"compute_normalized_laplacian: lambda_max={lambda_max.item()} <= 0 — check L_C.")
    n = L_C.shape[0]
    identity = torch.eye(n, dtype=L_C.dtype, device=L_C.device)
    return (2.0 / lambda_max) * L_C - identity


def complex_relu(z: torch.Tensor) -> torch.Tensor:
    """CReLU — ReLU applied independently to real and imaginary parts.
    A standard, simple choice for complex-valued networks (not the only
    option — ModReLU is an alternative — but the simplest one to reason
    about and debug, chosen as the working default here)."""
    return torch.complex(torch.relu(z.real), torch.relu(z.imag))


class MagneticLaplacianConv(nn.Module):
    """One Chebyshev-approximated spectral graph convolution over the
    (normalized) magnetic Laplacian.

    forward(X, L_norm):
      X: (n_nodes, in_channels) — real or complex; cast to complex64.
      L_norm: (n_nodes, n_nodes) complex, from compute_normalized_laplacian.
      Returns (n_nodes, out_channels) complex.
    """

    def __init__(self, in_channels: int, out_channels: int, K: int = 3, activation: bool = True):
        super().__init__()
        if K < 1:
            raise ValueError(f"MagneticLaplacianConv: K must be >= 1, got {K}")
        self.K = K
        self.activation = activation

        scale = (1.0 / (in_channels * K)) ** 0.5
        weight_real = torch.randn(K, in_channels, out_channels) * scale
        weight_imag = torch.randn(K, in_channels, out_channels) * scale
        self.weight = nn.Parameter(torch.complex(weight_real, weight_imag))
        self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.complex64))

    def forward(self, X: torch.Tensor, L_norm: torch.Tensor) -> torch.Tensor:
        """Accepts EITHER a single graph (X: (N,Cin), L_norm: (N,N)) --
        the original, still-default usage everywhere else in the
        project -- OR a batch of graphs sharing the same node count
        but each with its own connectivity (X: (B,N,Cin), L_norm:
        (B,N,N)), auto-detected from X.dim(). Added this session to
        vectorize train_epoch's per-sample Python loop into one real
        GPU batch call; NOT yet wired into train_epoch itself until
        numerically verified identical to the per-sample loop (see
        training/train_epoch_batched.py's own correctness test).

        torch.matmul batches automatically over any leading dims, so
        the Chebyshev recursion below is IDENTICAL code for both the
        single-graph and batched case -- only self.weight[k] (K,Cin,Cout,
        no batch dim) and self.bias (Cout,) need explicit unsqueezing to
        broadcast correctly against a batched X.
        """
        if X.shape[-2] != L_norm.shape[-2]:
            raise ValueError(
                f"MagneticLaplacianConv: X has {X.shape[-2]} nodes but L_norm is {tuple(L_norm.shape)}"
            )
        if not torch.is_complex(X):
            X = X.to(torch.complex64)
        if not torch.is_complex(L_norm):
            L_norm = L_norm.to(torch.complex64)

        batched = X.dim() == 3  # (B, N, Cin) vs (N, Cin)

        # Chebyshev recursion: T_0 = X, T_1 = L_norm @ X, T_k = 2*L_norm@T_{k-1} - T_{k-2}
        # torch.matmul batches transparently: (B,N,N)@(B,N,C) -> (B,N,C),
        # or (N,N)@(N,C) -> (N,C) in the non-batched case -- same call.
        Tx_list = [X]
        if self.K > 1:
            Tx_list.append(L_norm @ X)
        for k in range(2, self.K):
            Tx_list.append(2 * (L_norm @ Tx_list[-1]) - Tx_list[-2])

        if batched:
            B = X.shape[0]
            out = self.bias.clone().view(1, 1, -1).expand(B, X.shape[1], -1).clone()
        else:
            out = self.bias.clone().unsqueeze(0).expand(X.shape[0], -1).clone()

        for k in range(self.K):
            # Tx_list[k]: (B,N,Cin) or (N,Cin); self.weight[k]: (Cin,Cout).
            # matmul broadcasts the weight (no batch dim) against the
            # batched Tx automatically -- no unsqueeze needed here.
            out = out + Tx_list[k] @ self.weight[k]

        if self.activation:
            out = complex_relu(out)
        return out
