"""
grn_balladeer.training.train_epoch
======================================
Module 9 — one training epoch, EEG-only (no dual-branch/triplet yet —
those land Week 5+ once EmbracePlus data is available).

Per-sample forward (not batched matrix ops — see training/batch_forward.py's
docstring for why: each sample has its own graph/L_norm) computes both
the classification logits AND the per-node omega needed for the harmonic
and symbolic losses, then averages loss components across the batch
before a single optimizer step.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from grn_balladeer.losses.harmonic_loss import all_pairs_edge_index, compute_consonance_degree, harmonic_loss
from grn_balladeer.losses.symbolic_loss import get_frontal_pairs, symbolic_implication_loss
from grn_balladeer.losses.total_loss import total_loss
from grn_balladeer.model.classification_head import ClassificationHead, global_pool, split_real_imag
from grn_balladeer.model.grn_encoder import GRNEncoder, extract_resonance_frequency


def forward_sample(
    encoder: GRNEncoder,
    resonance_head: nn.Module,
    cls_head: ClassificationHead,
    X: torch.Tensor,
    L_norm: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One sample's full forward pass. Returns (logits, omega) — both
    needed downstream: logits for L_task, omega for L_harm/L_symb."""
    h = encoder(X, L_norm)
    omega = extract_resonance_frequency(h, resonance_head)
    h_real = split_real_imag(h)
    pooled = global_pool(h_real)
    logits = cls_head(pooled.unsqueeze(0))
    return logits, omega


def train_epoch(
    encoder: GRNEncoder,
    resonance_head: nn.Module,
    cls_head: ClassificationHead,
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    ch_names: List[str],
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    direction: str = "direct",
    grad_clip_max_norm: float = 1.0,
) -> dict:
    """One training epoch over `batch` (list of (X_i, L_norm_i) pairs,
    same convention as batch_forward.forward_batch) with matching
    `labels` (n_samples,) long tensor.

    direction='direct' is a PLACEHOLDER — determine_rule_direction has
    only been validated on synthetic data so far (needs multiple real
    ADHD+control subjects, not yet available — see Week 4 Notion notes).
    Do not treat this default as an empirically justified choice yet.

    grad_clip_max_norm=1.0: EMPIRICALLY MOTIVATED, not an arbitrary
    default. A real 30-epoch run on UB0136 without clipping showed
    harmonic_loss spike to 261 at epoch 10 (omega presumably shooting far
    from any consonance ratio for one unlucky gradient step) before
    recovering; with clip_grad_norm_(max_norm=1.0), the same run's worst
    spike was only 1.92, and the final loss was slightly better overall
    (0.872 vs 0.935 after 30 epochs). Nothing currently bounds omega's
    range (no activation constrains extract_resonance_frequency's linear
    output), so this instability is expected to recur without clipping;
    revisit if a bounded omega activation (e.g. scaled tanh to a
    physically plausible 1-45 Hz range) turns out to be a more principled
    fix once more training data is available to study this properly.

    Returns a dict of the epoch's mean loss components, for logging.
    """
    frontal_pairs = get_frontal_pairs(ch_names)
    task_loss_fn = nn.CrossEntropyLoss()

    optimizer.zero_grad()

    logits_list = []
    l_harm_list = []
    l_symb_list = []

    for (X_i, L_norm_i), label_i in zip(batch, labels):
        logits_i, omega_i = forward_sample(encoder, resonance_head, cls_head, X_i, L_norm_i)
        logits_list.append(logits_i)

        edges_i = all_pairs_edge_index(omega_i.shape[0])
        l_harm_list.append(harmonic_loss(omega_i, edges_i))

        mu_frontal_i = compute_consonance_degree(omega_i[frontal_pairs[:, 0]], omega_i[frontal_pairs[:, 1]])
        confidence_i = torch.softmax(logits_i, dim=-1)[0, label_i.item()]
        l_symb_list.append(symbolic_implication_loss(mu_frontal_i, confidence_i, direction=direction))

    logits_batch = torch.cat(logits_list, dim=0)
    l_task = task_loss_fn(logits_batch, labels)
    l_harm = torch.stack(l_harm_list).mean()
    l_symb = torch.stack(l_symb_list).mean()

    l_total = total_loss(l_task, l_harm, l_symb, lambda1=lambda1, lambda2=lambda2)
    l_total.backward()

    all_params = list(encoder.parameters()) + list(resonance_head.parameters()) + list(cls_head.parameters())
    torch.nn.utils.clip_grad_norm_(all_params, max_norm=grad_clip_max_norm)

    optimizer.step()

    return {
        "loss_total": l_total.item(),
        "loss_task": l_task.item(),
        "loss_harm": l_harm.item(),
        "loss_symb": l_symb.item(),
    }
