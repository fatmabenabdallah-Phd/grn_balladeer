"""
grn_balladeer.training.train_epoch
======================================
Module 9 — one training epoch, EEG-only branch. Reuses losses.total_loss
to combine task/harmonic/symbolic terms. Triplet term stays at its
default 0.0 here (Module 7b/8) - EEG-only means no auxiliary branch, no
fusion, no triplet mining (see train_epoch_dual_branch.py for that).

OPTIMIZED this session (ahead of the full 138-subject Colab run): the
original version called encoder(X_i, L_norm_i) TWICE per sample - once
via forward_batch for l_task's logits, once again for l_harm/l_symb's
omega. Harmless at 33-66 samples, but a real 2x cost at full-dataset
scale. Now computes h_i ONCE per sample and derives both the pooled
z_eeg_i (for logits) and omega_i (for l_harm/l_symb) from that same
forward pass - mathematically identical result, roughly half the
encoder compute per epoch.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from grn_balladeer.model.classification_head import ClassificationHead, global_pool, split_real_imag
from grn_balladeer.model.grn_encoder import GRNEncoder, extract_resonance_frequency
from grn_balladeer.losses.harmonic_loss import harmonic_loss, all_pairs_edge_index, compute_consonance_degree
from grn_balladeer.losses.symbolic_loss import get_frontal_pairs, symbolic_implication_loss
from grn_balladeer.losses.total_loss import total_loss


def train_epoch(
    encoder: GRNEncoder,
    head: ClassificationHead,
    resonance_head: nn.Module,
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    labels: torch.Tensor,
    ch_names: List[str],
    optimizer: torch.optim.Optimizer,
    symbolic_direction: str = "direct",
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    pool_method: str = "mean",
) -> dict:
    """Runs ONE training epoch (one full pass over `batch`, single
    gradient step per call - call this in an outer loop over epochs).

    batch: list of (X_i, L_norm_i), one per sample, as in forward_batch.
    labels: (n_samples,) long tensor, class index per sample.
    ch_names: channel names for the graph nodes (needed to find frontal
        pairs for L_symb) - assumed IDENTICAL across all samples in the
        batch (same channel layout per epoch). If a future dataset mixes
        CGX/Emotiv subjects with different channel sets in one batch,
        this assumption breaks and per-sample frontal pairs would be
        needed instead - not handled here, flagging for later.

    Returns a dict of the four loss components (for logging/plotting)
    plus the resulting omega (from the LAST sample's embedding only -
    used for omega_diagnostics.check_omega_collapse - a full per-batch
    omega collapse check across all samples is not done here, since
    omega is per-node-per-sample and there's no single natural
    aggregate; revisit if this granularity turns out to matter).
    """
    encoder.train()
    head.train()
    resonance_head.train()
    optimizer.zero_grad()

    frontal_pairs = get_frontal_pairs(ch_names)
    all_pairs = all_pairs_edge_index(batch[0][0].shape[0])

    logits_list = []
    l_harm_terms = []
    omega_per_sample = []  # kept for the l_symb pass below (needs softmax(logits) first)
    last_omega = None

    for X_i, L_norm_i in batch:
        h_i = encoder(X_i, L_norm_i)  # SINGLE forward pass per sample now

        z_eeg_i = global_pool(split_real_imag(h_i), method=pool_method)
        logits_list.append(head(z_eeg_i.unsqueeze(0)))

        omega_i = extract_resonance_frequency(h_i, resonance_head)
        omega_per_sample.append(omega_i)
        last_omega = omega_i
        l_harm_terms.append(harmonic_loss(omega_i, all_pairs))

    logits = torch.cat(logits_list, dim=0)  # (n_samples, n_classes)
    l_task = nn.functional.cross_entropy(logits, labels)

    # l_symb needs softmax(logits) as "confidence", which needs ALL logits computed
    # first - hence this second (cheap, no encoder call) pass over the already-
    # computed omega_per_sample list, not a second encoder forward.
    probs = torch.softmax(logits, dim=-1)
    l_symb_terms = []
    for sample_idx, omega_i in enumerate(omega_per_sample):
        omega_frontal_i = omega_i[frontal_pairs[:, 0]]
        omega_frontal_j = omega_i[frontal_pairs[:, 1]]
        mu_ij = compute_consonance_degree(omega_frontal_i, omega_frontal_j)
        confidence_i = probs[sample_idx, labels[sample_idx]]
        l_symb_terms.append(symbolic_implication_loss(mu_ij, confidence_i, direction=symbolic_direction))

    l_harm = torch.stack(l_harm_terms).mean()
    l_symb = torch.stack(l_symb_terms).mean()

    loss = total_loss(l_task, l_harm, l_symb, lambda1=lambda1, lambda2=lambda2)
    loss.backward()
    optimizer.step()

    return {
        "loss_total": loss.item(),
        "loss_task": l_task.item(),
        "loss_harm": l_harm.item(),
        "loss_symb": l_symb.item(),
        "last_omega": last_omega.detach(),
    }
