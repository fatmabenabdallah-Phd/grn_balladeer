"""
grn_balladeer.training.train_epoch_batched_chbmit
=====================================================
Parametrized clone of training.train_epoch_batched, same convention as
training.train_epoch_batched_biciv2a: `symbolic_channels` is a real
argument (defaulting to losses.epilepsy_channels.EPILEPSY_CHANNELS,
i.e. FP1-F7/FP2-F8) rather than the original's hard-coded ADHD default.
Reused here rather than editing the biciv2a version because the two
domains' default clusters and channel lists differ (CHBMIT_CHANNELS,
18 channels, vs BICIV2A_CHANNELS, 22) -- no logic differs beyond that
default, only the import source and docstring.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from grn_balladeer.model.classification_head import ClassificationHead, global_pool, split_real_imag
from grn_balladeer.model.grn_encoder import GRNEncoder, extract_resonance_frequency
from grn_balladeer.losses.harmonic_loss import harmonic_loss, all_pairs_edge_index, compute_consonance_degree
from grn_balladeer.losses.symbolic_loss import get_frontal_pairs, symbolic_implication_loss
from grn_balladeer.losses.total_loss import total_loss
from grn_balladeer.losses.epilepsy_channels import EPILEPSY_CHANNELS


def train_epoch_batched_chbmit(
    encoder: GRNEncoder,
    head: ClassificationHead,
    resonance_head: nn.Module,
    X_batch: torch.Tensor,
    L_batch: torch.Tensor,
    labels: torch.Tensor,
    ch_names: List[str],
    optimizer: torch.optim.Optimizer,
    symbolic_channels: List[str] = EPILEPSY_CHANNELS,
    symbolic_direction: str = "direct",
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    pool_method: str = "mean",
    class_weights: "torch.Tensor | None" = None,
) -> dict:
    """Identical to train_epoch_batched, except get_frontal_pairs is
    called with `frontal_channels=symbolic_channels`. Set lambda2=0.0
    to run the L_symb-disabled ablation condition (see
    losses.epilepsy_channels for why both conditions are tested here,
    rather than only the anchored one as was eventually done for
    motor imagery's C3/C4 cluster).
    """
    encoder.train()
    head.train()
    resonance_head.train()
    optimizer.zero_grad()

    symbolic_pairs = get_frontal_pairs(ch_names, frontal_channels=symbolic_channels)
    all_pairs = all_pairs_edge_index(X_batch.shape[1])

    h_batch = encoder(X_batch, L_batch)
    z_eeg_batch = global_pool(split_real_imag(h_batch), method=pool_method)
    logits = head(z_eeg_batch)
    l_task = nn.functional.cross_entropy(logits, labels, weight=class_weights)

    omega_batch = extract_resonance_frequency(h_batch, resonance_head)
    l_harm = harmonic_loss(omega_batch, all_pairs).mean()

    probs = torch.softmax(logits, dim=-1)
    omega_sym_i = omega_batch[..., symbolic_pairs[:, 0]]
    omega_sym_j = omega_batch[..., symbolic_pairs[:, 1]]
    mu_ij = compute_consonance_degree(omega_sym_i, omega_sym_j)
    confidence = probs[torch.arange(probs.shape[0], device=probs.device), labels]
    l_symb = symbolic_implication_loss(mu_ij, confidence, direction=symbolic_direction)

    loss = total_loss(l_task, l_harm, l_symb, lambda1=lambda1, lambda2=lambda2)
    loss.backward()
    optimizer.step()

    last_omega = omega_batch[-1].detach()

    return {
        "loss_total": loss.item(),
        "loss_task": l_task.item(),
        "loss_harm": l_harm.item(),
        "loss_symb": l_symb.item(),
        "last_omega": last_omega,
    }
