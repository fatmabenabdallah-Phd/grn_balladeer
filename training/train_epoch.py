"""
grn_balladeer.training.train_epoch
======================================
Module 9 — one training epoch, EEG-only branch. Reuses forward_batch
(Week 3 sanity check) for the per-sample graph-forward loop, and
losses.total_loss to combine task/harmonic/symbolic terms. Triplet term
stays at its default 0.0 here (Module 7b/8, Week 5) - EEG-only means no
auxiliary branch, no fusion, no triplet mining yet.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import torch
import torch.nn as nn

from grn_balladeer.model.classification_head import ClassificationHead, global_pool, split_real_imag
from grn_balladeer.model.grn_encoder import GRNEncoder, extract_resonance_frequency
from grn_balladeer.training.batch_forward import forward_batch
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

    # NOTE: encoder(X_i, L_norm_i) is computed twice per sample - once inside
    # forward_batch for l_task's logits, once again in the loop below for
    # l_harm/l_symb's omega. Mathematically fine (same weights, gradients from
    # both computational graphs correctly accumulate on .backward()), just not
    # compute-efficient. Kept simple/readable for now given the tiny batch
    # sizes in play (33 real epochs total) - revisit if this becomes a real
    # bottleneck at full-dataset scale (same tradeoff already made explicitly
    # in batch_forward.py's per-sample Python loop).
    logits = forward_batch(encoder, head, batch, pool_method=pool_method)  # (n_samples, n_classes)
    l_task = nn.functional.cross_entropy(logits, labels)

    frontal_pairs = get_frontal_pairs(ch_names)
    all_pairs = all_pairs_edge_index(batch[0][0].shape[0])

    l_harm_terms = []
    l_symb_terms = []
    last_omega = None
    probs = torch.softmax(logits, dim=-1)
    for sample_idx, (X_i, L_norm_i) in enumerate(batch):
        h_i = encoder(X_i, L_norm_i)
        omega_i = extract_resonance_frequency(h_i, resonance_head)
        last_omega = omega_i

        l_harm_terms.append(harmonic_loss(omega_i, all_pairs))

        omega_frontal_i = omega_i[frontal_pairs[:, 0]]
        omega_frontal_j = omega_i[frontal_pairs[:, 1]]
        mu_ij = compute_consonance_degree(omega_frontal_i, omega_frontal_j)
        confidence_i = probs[sample_idx, labels[sample_idx]]  # P(true class) for this sample
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
