"""
grn_balladeer/model/cross_attention_fusion.py
==============================================
Mid-level interactive dual-branch fusion via cross-attention.

Fuses the EEG embedding (z_eeg, from GRNEncoder) and the auxiliary
embedding (z_aux, from AuxBranchEncoder) through bidirectional
cross-attention, then projects the result to a single joint embedding
that feeds both the classification head and the triplet loss.

Architecture:
    z_eeg [B,D] ─┬─ Q ─► MHA_eeg(Q=eeg, K=V=aux) ─► eeg' ─► LayerNorm(eeg + eeg')
                 │                                                          ─► z_eeg_fused [B,D]
    z_aux [B,D] ─┼─ Q ─► MHA_aux(Q=aux, K=V=eeg) ─► aux' ─► LayerNorm(aux + aux')
                 │                                                          ─► z_aux_fused [B,D]
                 └─ concat([z_eeg_fused, z_aux_fused]) [B,2D]
                         ─► Linear(2D→D) ─► L2-normalize ─► z_joint [B,D]

Design choices:
  - n_heads=1: hidden_dim=64 is too small for meaningful multi-head split.
  - seq_len=1: both branches produce a single pooled vector (no temporal axis).
    Attention weights are therefore [B,1,1] — a learned scalar gate per sample,
    not a spatial attention map. This is intentional: it learns how much each
    branch should modulate the other, per sample.
  - Residual connections: if cross-attention is uninformative (e.g. EDA is
    zero-padded for subjects without EmbracePlus), the original embedding
    is preserved via the residual path.
  - L2 normalisation of z_joint: required before triplet loss (cosine distance).

VALIDATED on real data (2026-07-18):
  - z_eeg [66, 64] (simulated GRN output) + real z_aux [66, 64] from
    UB0136/UB0004 aux vectors → z_joint [66, 64], L2-norms all 1.0.
  - No NaNs, gradients flow. Total module params: 41,792.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention fusion for EEG and auxiliary branches.

    Parameters
    ----------
    hidden_dim : int   — embedding dimension for both branches (default 64)
    n_heads    : int   — number of attention heads (default 1)
    dropout    : float — applied inside MHA (default 0.1)
    """

    def __init__(
        self,
        hidden_dim: int   = 64,
        n_heads:    int   = 1,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()

        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})"
            )

        self.hidden_dim = hidden_dim

        # EEG attends to AUX
        self.mha_eeg = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        # AUX attends to EEG
        self.mha_aux = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_eeg = nn.LayerNorm(hidden_dim)
        self.norm_aux = nn.LayerNorm(hidden_dim)

        # Project concatenated [2*hidden_dim] back to [hidden_dim]
        self.fusion_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self,
        z_eeg: torch.Tensor,
        z_aux: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        z_eeg : torch.Tensor [batch, hidden_dim] — GRN pooled EEG embedding
        z_aux : torch.Tensor [batch, hidden_dim] — auxiliary branch embedding

        Returns
        -------
        z_joint   : torch.Tensor [batch, hidden_dim] — L2-normalised joint embedding
        z_eeg_fused : torch.Tensor [batch, hidden_dim] — EEG after cross-attention
        z_aux_fused : torch.Tensor [batch, hidden_dim] — AUX after cross-attention

        Note: z_eeg_fused and z_aux_fused are returned for interpretability
        (inspect which branch contributed more to z_joint per sample).
        """
        # Add seq_len=1 dimension for MHA: [B, D] → [B, 1, D]
        e = z_eeg.unsqueeze(1)   # [B, 1, D]
        a = z_aux.unsqueeze(1)   # [B, 1, D]

        # EEG attends to AUX (Q=eeg, K=V=aux)
        eeg_prime, _ = self.mha_eeg(query=e, key=a, value=a)
        eeg_prime = eeg_prime.squeeze(1)                         # [B, D]
        z_eeg_fused = self.norm_eeg(z_eeg + eeg_prime)          # residual

        # AUX attends to EEG (Q=aux, K=V=eeg)
        aux_prime, _ = self.mha_aux(query=a, key=e, value=e)
        aux_prime = aux_prime.squeeze(1)                         # [B, D]
        z_aux_fused = self.norm_aux(z_aux + aux_prime)          # residual

        # Fuse: concat → project → L2-normalise
        z_cat   = torch.cat([z_eeg_fused, z_aux_fused], dim=-1)  # [B, 2D]
        z_joint = self.fusion_proj(z_cat)                         # [B, D]
        z_joint = F.normalize(z_joint, p=2, dim=-1)              # unit sphere

        return z_joint, z_eeg_fused, z_aux_fused


def fuse_branches(
    z_eeg:  torch.Tensor,
    z_aux:  torch.Tensor,
    fusion: CrossAttentionFusion,
) -> torch.Tensor:
    """
    Convenience wrapper: returns only z_joint (for use in training loop).

    Parameters
    ----------
    z_eeg   : [batch, hidden_dim]
    z_aux   : [batch, hidden_dim]
    fusion  : CrossAttentionFusion instance

    Returns
    -------
    z_joint : [batch, hidden_dim], L2-normalised
    """
    z_joint, _, _ = fusion(z_eeg, z_aux)
    return z_joint
