"""
grn_balladeer/model/aux_branch_encoder.py
==========================================
Auxiliary branch MLP that maps the concatenated EDA + behavioral
feature vector to the same embedding space as the GRN output,
so that cross-attention fusion can operate on commensurate representations.

Input  : [batch, AUX_INPUT_DIM=12]  — 6 EDA + 6 behavioral features
Output : [batch, hidden_dim]         — same dim as GRN pooled embedding

VALIDATED on real data (2026-07-18):
  - UB0136 (has both EDA and behavioral): clean forward pass, shape [1, 64]
  - UB0004 (EDA missing → zeros padded, behavioral only): same shape, no NaNs
  - Batch [132, 12] → [132, 64]: correct shape ✓
  - Gradients flow through all 3 linear layers ✓
  - Parameter count: 9,408 (genuinely lightweight)

MISSING EDA HANDLING:
  When a subject has no EmbracePlus data (UB0004, UB0022, UB0023 in our
  current dataset), the EDA slice (first 6 dims) is set to zeros before
  passing to this encoder. The encoder still produces a valid output —
  it just encodes behavioral-only information. The caller (dual-branch
  DataLoader) is responsible for this zero-padding.
  See build_aux_vector() below for the canonical construction.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional

# Must match eda_features.EDA_FEATURE_DIM + behavioral_features.BEHAVIORAL_FEATURE_DIM
AUX_INPUT_DIM = 12   # 6 EDA + 6 behavioral

# Must match GRNEncoder hidden_dim (set in grn_encoder.py)
DEFAULT_HIDDEN_DIM = 64


class AuxBranchEncoder(nn.Module):
    """
    3-layer MLP encoder for the auxiliary (EDA + behavioral) branch.

    Architecture:
        Linear(input_dim → hidden_dim) → LayerNorm → ReLU → Dropout
        Linear(hidden_dim → hidden_dim) → LayerNorm → ReLU → Dropout
        Linear(hidden_dim → hidden_dim)

    LayerNorm (not BatchNorm) is used because batch sizes can be small
    (down to 1 subject per class in our current 4-subject dataset).
    No final activation — the output feeds directly into cross-attention,
    which does not require bounded inputs.

    Parameters
    ----------
    input_dim  : int  — concatenated aux feature dim (default 12)
    hidden_dim : int  — output dim, must equal GRNEncoder hidden_dim (default 64)
    dropout    : float — applied after each of the first two ReLUs (default 0.2)
    """

    def __init__(
        self,
        input_dim:  int   = AUX_INPUT_DIM,
        hidden_dim: int   = DEFAULT_HIDDEN_DIM,
        dropout:    float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor [batch, input_dim]

        Returns
        -------
        torch.Tensor [batch, hidden_dim]
        """
        return self.net(x)


# ── Canonical aux vector construction ────────────────────────────────────

def build_aux_vector(
    eda_features:          Optional[np.ndarray],
    behavioral_features:   Optional[np.ndarray],
    eda_feature_dim:       int = 6,
    behavioral_feature_dim: int = 6,
) -> Optional[np.ndarray]:
    """
    Concatenate EDA and behavioral feature vectors into a single aux vector.

    If EDA is missing (subject not in EmbracePlus file), the EDA slice is
    zero-padded — the encoder still runs, encoding behavioral-only information.
    If behavioral features are missing, returns None (session is unusable).

    Parameters
    ----------
    eda_features        : np.ndarray [6] or None
    behavioral_features : np.ndarray [6] or None

    Returns
    -------
    np.ndarray [12] dtype float32, or None if behavioral_features is None.
    """
    if behavioral_features is None:
        return None

    if eda_features is None:
        eda_features = np.zeros(eda_feature_dim, dtype=np.float32)

    aux = np.concatenate([eda_features, behavioral_features]).astype(np.float32)

    if np.isnan(aux).any():
        # Impute any residual NaNs (should not occur after feature extractors,
        # but defend here as a last line)
        aux = np.where(np.isnan(aux), 0.0, aux).astype(np.float32)

    return aux


def zscore_aux_batch(
    aux_matrix: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Z-score normalize a batch of aux vectors per feature dimension.
    Must be fit on the TRAINING fold only and applied to val/test.

    Parameters
    ----------
    aux_matrix : np.ndarray [n_samples, 12]
    eps        : small constant to avoid division by zero on constant features

    Returns
    -------
    np.ndarray [n_samples, 12] z-scored, dtype float32.
    """
    mean = aux_matrix.mean(axis=0, keepdims=True)
    std  = aux_matrix.std(axis=0, keepdims=True)
    std  = np.where(std < eps, 1.0, std)
    return ((aux_matrix - mean) / std).astype(np.float32)
