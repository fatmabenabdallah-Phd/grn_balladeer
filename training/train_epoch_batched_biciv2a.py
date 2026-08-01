"""
grn_balladeer.training.train_epoch_batched_biciv2a
======================================================
Parametrized clone of training.train_epoch_batched, existing SOLELY
because the original hard-codes losses.symbolic_loss.get_frontal_pairs'
default FRONTAL_CHANNELS (the ADHD cluster) inside the function body --
it is not a caller-supplied argument there. Verified concretely: only
1 of the 5 ADHD frontal channels (Fz) exists in BICIV2A_CHANNELS,
so calling the original train_epoch_batched on BCI IV 2a data raises
(get_frontal_pairs requires >=2 channels to form a pair) rather than
silently running condition (a) ("same neurosymbolic constraints as
ADHD") as might be assumed. This closes off condition (a) in its
literal form for THIS montage -- there simply aren't enough overlapping
electrodes -- leaving condition (b) (re-anchor L_symb to a
task-appropriate cluster) as the only viable option here.

Not a general-purpose refactor of train_epoch_batched to take an
arbitrary cluster (that change belongs with the ADHD-validated file if
made at all, and is out of scope here) -- this is a domain-specific
copy, same convention as build_dataset_nasrabadi.py existing alongside
build_dataset.py rather than editing it.

Only change from the original: `symbolic_channels` is a real parameter
(defaulting to losses.motor_imagery_channels.MOTOR_CHANNELS, i.e.
C3/C4), passed through to get_frontal_pairs(ch_names,
frontal_channels=symbolic_channels) instead of relying on its own
internal default. Every other line -- loss composition, omega
extraction, class weights, return dict -- is unchanged.
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
from grn_balladeer.losses.motor_imagery_channels import MOTOR_CHANNELS


def train_epoch_batched_biciv2a(
    encoder: GRNEncoder,
    head: ClassificationHead,
    resonance_head: nn.Module,
    X_batch: torch.Tensor,
    L_batch: torch.Tensor,
    labels: torch.Tensor,
    ch_names: List[str],
    optimizer: torch.optim.Optimizer,
    symbolic_channels: List[str] = MOTOR_CHANNELS,
    symbolic_direction: str = "direct",
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    pool_method: str = "mean",
    class_weights: "torch.Tensor | None" = None,
) -> dict:
    """Identical to train_epoch_batched, except get_frontal_pairs is
    called with `frontal_channels=symbolic_channels` explicitly instead
    of its ADHD-specific default. See module docstring for why this
    exists as a separate file rather than a parameter added to the
    original.
    """
    encoder.train()
    head.train()
    resonance_head.train()
    optimizer.zero_grad()

    symbolic_pairs = get_frontal_pairs(ch_names, frontal_channels=symbolic_channels)
    all_pairs = all_pairs_edge_index(X_batch.shape[1])

    h_batch = encoder(X_batch, L_batch)                        # (B, N, d) complex, ONE call
    z_eeg_batch = global_pool(split_real_imag(h_batch), method=pool_method)  # (B, 2d)
    logits = head(z_eeg_batch)                                  # (B, n_classes)
    l_task = nn.functional.cross_entropy(logits, labels, weight=class_weights)

    omega_batch = extract_resonance_frequency(h_batch, resonance_head)  # (B, N)
    l_harm = harmonic_loss(omega_batch, all_pairs).mean()  # (B,) -> scalar over the fold

    probs = torch.softmax(logits, dim=-1)
    omega_sym_i = omega_batch[..., symbolic_pairs[:, 0]]
    omega_sym_j = omega_batch[..., symbolic_pairs[:, 1]]
    mu_ij = compute_consonance_degree(omega_sym_i, omega_sym_j)  # (B, n_pairs)
    confidence = probs[torch.arange(probs.shape[0], device=probs.device), labels]  # (B,)
    l_symb = symbolic_implication_loss(mu_ij, confidence, direction=symbolic_direction)

    loss = total_loss(l_task, l_harm, l_symb, lambda1=lambda1, lambda2=lambda2)
    loss.backward()
    optimizer.step()

    last_omega = omega_batch[-1].detach()  # matches train_epoch's "last sample" convention

    return {
        "loss_total": loss.item(),
        "loss_task": l_task.item(),
        "loss_harm": l_harm.item(),
        "loss_symb": l_symb.item(),
        "last_omega": last_omega,
    }
