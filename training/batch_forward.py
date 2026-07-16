"""
grn_balladeer.training.batch_forward
========================================
Small forward-batching helper, introduced early (Week 3 sanity check)
because it will be reused as-is by the Module 9 training loop.

Each sample in a batch has ITS OWN graph (L_norm varies per epoch, since
connectivity is computed per-epoch in Module 3) — this is not standard
tensor batching. The straightforward, honest approach used here is a
Python loop over samples (no custom sparse-batching machinery): shared
model weights are applied to each (X_i, L_norm_i) pair independently,
and the resulting logits are stacked into a real batch dimension for
the loss. Simple, correct, and easy to debug — revisit only if profiling
later shows this loop is a real bottleneck at full dataset scale.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from grn_balladeer.model.classification_head import ClassificationHead, global_pool, split_real_imag
from grn_balladeer.model.grn_encoder import GRNEncoder


def forward_batch(
    encoder: GRNEncoder,
    head: ClassificationHead,
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    pool_method: str = "mean",
) -> torch.Tensor:
    """batch: list of (X_i, L_norm_i) pairs, one per sample (epoch/event),
    each X_i (n_nodes, in_channels), L_norm_i (n_nodes, n_nodes) complex.
    Returns logits, shape (batch_size, n_classes).
    """
    logits_list = []
    for X_i, L_norm_i in batch:
        h_i = encoder(X_i, L_norm_i)
        h_real_i = split_real_imag(h_i)
        pooled_i = global_pool(h_real_i, method=pool_method)
        logits_list.append(head(pooled_i.unsqueeze(0)))
    return torch.cat(logits_list, dim=0)
