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
    """Pools (n_nodes, d) real-valued node embeddings into a single
    (d,) graph-level embedding. method in {'mean', 'sum', 'max'}."""
    if method == "mean":
        return node_embeddings.mean(dim=0)
    elif method == "sum":
        return node_embeddings.sum(dim=0)
    elif method == "max":
        return node_embeddings.max(dim=0).values
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
