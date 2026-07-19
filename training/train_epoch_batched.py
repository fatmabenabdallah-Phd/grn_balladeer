"""
grn_balladeer.training.train_epoch_batched
=============================================
Vectorized replacement for train_epoch.py's per-sample Python loop.
Verified numerically identical to the original (logits, l_task, l_harm,
l_symb all matched to float rounding precision, see this session's test)
before being introduced -- NOT a drop-in silent behavior change.

WHY THIS EXISTS: the original train_epoch() calls encoder(X_i, L_norm_i)
once per sample inside a Python for-loop -- e.g. ~4600 separate small
GPU kernel launches per epoch for a 91-subject training fold (91 x ~51
EEG epochs each). On the first full 114-subject Colab run, a single
fold took 45+ minutes on a T4 GPU despite the actual FLOP count being
small -- overwhelmingly kernel-launch/Python-loop overhead, not real
compute. This version stacks every sample into one batch dimension and
calls the encoder/heads ONCE per training epoch instead of once per
sample, which is what actually uses the GPU's parallelism.

REQUIRES: model.magnetic_laplacian_conv.MagneticLaplacianConv,
model.classification_head.global_pool, losses.harmonic_loss.
harmonic_loss, and losses.symbolic_loss.symbolic_implication_loss all
updated this session to auto-detect and correctly handle a leading
batch dimension (previously they would have silently indexed/pooled/
broadcast over the wrong axis for batched input -- see each file's
own docstring for the specific fix).

LIMITATION (inherited from the batching approach, not this file's own
bug): every sample in the batch must have the SAME number of nodes
(true here -- all real subjects share the 30-channel CGX montage), but
L_norm differs per sample (real per-epoch connectivity), which is
exactly the case torch.matmul's batched broadcasting handles.
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


def train_epoch_batched(
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
    class_weights: "torch.Tensor | None" = None,
) -> dict:
    """Same signature, same return dict, same loss semantics as
    train_epoch() -- ONLY the internal computation is vectorized (one
    encoder forward call over a stacked batch, instead of one call per
    sample in a Python loop). Verified numerically equivalent to
    train_epoch() on identical inputs/weights (see this session's test:
    logits, l_task, l_harm, l_symb all matched).

    batch: list of (X_i, L_norm_i), one per sample -- same input format
    as train_epoch(), stacked internally into (B,N,Cin)/(B,N,N) tensors.
    """
    encoder.train()
    head.train()
    resonance_head.train()
    optimizer.zero_grad()

    frontal_pairs = get_frontal_pairs(ch_names)
    all_pairs = all_pairs_edge_index(batch[0][0].shape[0])

    # Stack the whole fold into one batch -- this is the key change.
    X_batch = torch.stack([X_i for X_i, _ in batch])          # (B, N, Cin)
    L_batch = torch.stack([L_i for _, L_i in batch])          # (B, N, N)

    h_batch = encoder(X_batch, L_batch)                        # (B, N, d) complex, ONE call
    z_eeg_batch = global_pool(split_real_imag(h_batch), method=pool_method)  # (B, 2d)
    logits = head(z_eeg_batch)                                  # (B, n_classes)
    l_task = nn.functional.cross_entropy(logits, labels, weight=class_weights)

    omega_batch = extract_resonance_frequency(h_batch, resonance_head)  # (B, N)
    l_harm = harmonic_loss(omega_batch, all_pairs).mean()  # (B,) -> scalar over the fold

    probs = torch.softmax(logits, dim=-1)
    omega_frontal_i = omega_batch[..., frontal_pairs[:, 0]]
    omega_frontal_j = omega_batch[..., frontal_pairs[:, 1]]
    mu_ij = compute_consonance_degree(omega_frontal_i, omega_frontal_j)  # (B, n_pairs)
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
