"""
grn_balladeer/model/aux_branch_decoder.py
=============================================
Decoder for the auxiliary (EDA + behavioral) branch, mirroring
AuxBranchEncoder. Reconstructs the raw 12-dim aux vector from
Encoder2's embedding -- a self-supervised regularizer for the
auxiliary branch, distinct from train_epoch_dual_branch.py's default
training loop (task + harmonic + symbolic + triplet losses only, no
decoder or reconstruction loss).

Rationale: the auxiliary branch is trained on only 78 subjects
(EDA-real-only subset) -- a reconstruction objective adds a
self-supervised training signal that requires no labels, potentially
regularizing AuxBranchEncoder toward a richer representation than
classification loss alone can shape at this sample size. This is
distinct from EEG-side autoencoder pretraining (found ineffective for
the EEG branch alone): here reconstruction is a joint multi-task
training signal on the auxiliary branch, not a separate pretraining
phase on EEG.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from grn_balladeer.model.aux_branch_encoder import AUX_INPUT_DIM, DEFAULT_HIDDEN_DIM


class AuxBranchDecoder(nn.Module):
    """Mirrors AuxBranchEncoder's 3-layer MLP structure in reverse:
    Linear(hidden_dim -> hidden_dim) -> LayerNorm -> ReLU -> Dropout
    Linear(hidden_dim -> hidden_dim) -> LayerNorm -> ReLU -> Dropout
    Linear(hidden_dim -> input_dim)

    No final activation -- the raw aux vector (EDA + behavioral
    features) is not bounded to any fixed range after z-scoring, so an
    unconstrained linear output is appropriate (matching
    AuxBranchEncoder's own choice of no final activation on its output).
    """

    def __init__(
        self,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        output_dim: int = AUX_INPUT_DIM,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z_aux: torch.Tensor) -> torch.Tensor:
        """z_aux: (batch, hidden_dim) -- AuxBranchEncoder's output.
        Returns (batch, output_dim) -- reconstructed raw aux vector."""
        return self.net(z_aux)
