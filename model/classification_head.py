"""
grn_balladeer.model.classification_head
===========================================
Module 6 — turns per-node complex embeddings into a binary
ADHD/control prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def split_real_imag(h_complex: torch.Tensor) -> torch.Tensor:
    """Concatenates the real and imaginary parts of a complex tensor
    along the last dimension: (n_nodes, d) complex -> (n_nodes, 2*d)
    real. Same operation used inline in extract_resonance_frequency,
    formalized here as a reusable building block for the classification
    path."""
    return torch.cat([h_complex.real, h_complex.imag], dim=-1)


def global_pool(node_embeddings: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """Pools node-level embeddings into a graph-level embedding.

    Accepts EITHER a single graph (N, d) -> returns (d,), pooling over
    dim=0 (nodes) -- the original, still-default behavior -- OR a batch
    (B, N, d) -> returns (B, d), pooling over dim=1 (nodes), added this
    session alongside MagneticLaplacianConv's batch support. Auto-
    detected via node_embeddings.dim(); pooling over the wrong axis for
    the batched case would silently collapse the BATCH dimension
    instead of nodes, so this distinction matters and is not just
    cosmetic.

    method in {'mean', 'sum', 'max'}.
    """
    node_dim = 1 if node_embeddings.dim() == 3 else 0
    if method == "mean":
        return node_embeddings.mean(dim=node_dim)
    elif method == "sum":
        return node_embeddings.sum(dim=node_dim)
    elif method == "max":
        return node_embeddings.max(dim=node_dim).values
    else:
        raise ValueError(f"global_pool: unknown method '{method}', expected 'mean'/'sum'/'max'")


class ClassificationHead(nn.Module):
    """MLP on the pooled graph-level embedding, producing 2-class
    logits (ADHD/control). Softmax is left to the loss function
    (nn.CrossEntropyLoss expects raw logits) rather than applied here."""

    def __init__(self, in_features: int, hidden_features: int = 32, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, n_classes),
        )

    def forward(self, pooled_embedding: torch.Tensor) -> torch.Tensor:
        return self.net(pooled_embedding)
