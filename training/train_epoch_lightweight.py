"""
grn_balladeer.training.train_epoch_lightweight
==================================================
Training loop for the lightweight TCN + structural-graph + band-power-
fusion architecture. Mirrors train_epoch_batched.py's vectorized,
mini-batch design (this session's own earlier fix for the GRN training
loop's speed problem) rather than repeating the original per-sample
Python loop mistake.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from grn_balladeer.model.tcn_encoder import LightweightTCNEncoder, count_parameters


class LightweightClassifier(nn.Module):
    """Fuses the TCN-graph node embeddings (mean-pooled over nodes) with
    explicit band-power features (theta/beta ratio included) via
    concatenation, then a small MLP. The explicit features are what let
    a plain Random Forest reach AUC=0.668 this session -- fusing them
    here rather than relying solely on the TCN's learned representation
    is a deliberate hedge, not an admission the TCN alone is expected
    to underperform; ablating this fusion (TCN-only vs. TCN+features)
    is a natural next experiment once this architecture is validated
    end-to-end.
    """

    def __init__(self, tcn_hidden: int = 8, band_power_dim: int = 151, mlp_hidden: int = 32, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(tcn_hidden + band_power_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 2),
        )

    def forward(self, tcn_node_embeddings: torch.Tensor, band_power_features: torch.Tensor) -> torch.Tensor:
        """tcn_node_embeddings: (B, N, tcn_hidden). band_power_features:
        (B, band_power_dim). Returns (B, 2) logits."""
        pooled = tcn_node_embeddings.mean(dim=1)  # (B, tcn_hidden)
        fused = torch.cat([pooled, band_power_features], dim=-1)
        return self.net(fused)


def train_epoch_lightweight(
    encoder: LightweightTCNEncoder,
    classifier: LightweightClassifier,
    adjacency_norm: torch.Tensor,
    X_batch: torch.Tensor,
    band_power_batch: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    class_weights: "torch.Tensor | None" = None,
) -> dict:
    """One mini-batch training step. X_batch: (B, N, T) raw per-node
    time series. band_power_batch: (B, band_power_dim). Vectorized (one
    encoder call per batch), matching train_epoch_batched.py's design
    rather than looping per sample.
    """
    encoder.train()
    classifier.train()
    optimizer.zero_grad()

    node_embeddings = encoder(X_batch, adjacency_norm)
    logits = classifier(node_embeddings, band_power_batch)
    loss = nn.functional.cross_entropy(logits, labels, weight=class_weights)

    loss.backward()
    optimizer.step()

    return {"loss": loss.item()}


def report_model_size(encoder: LightweightTCNEncoder, classifier: LightweightClassifier) -> dict:
    """Reports total parameter count for the paper's efficiency-metrics
    table (params/FLOPs/latency, contextualized against BMI4DND's Table
    1 as an intentional non-deployment-claim comparison per this
    project's own Notion notes)."""
    n_encoder = count_parameters(encoder)
    n_classifier = count_parameters(classifier)
    return {"encoder_params": n_encoder, "classifier_params": n_classifier, "total_params": n_encoder + n_classifier}
