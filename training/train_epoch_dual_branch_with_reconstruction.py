"""
grn_balladeer/training/train_epoch_dual_branch_with_reconstruction.py
=========================================================================
Extends train_epoch_dual_branch.py (Module 9) with the auxiliary-branch
reconstruction loss from the user's hand-sketched schema:

    Modality1 (EEG) -> Input1 -> Encoder1 --------\
                                                     -> Fit -> Classify -> L_entropy
    Modality2 (aux) -> Input2 -> Encoder2 ---+------/
                                              |
                                              +--> Decoder -> L_reconstruction
    Triplet loss: touches Encoder2's embedding AND the joint (Fit) embedding

This was never implemented in the original train_epoch_dual_branch.py
(confirmed by direct inspection this session) -- that version has no
decoder/reconstruction path at all. This variant adds:
  - AuxBranchDecoder reconstructing the raw 12-dim aux vector from
    Encoder2's embedding (z_aux), with an MSE reconstruction loss
    (lambda4 * L_recon), added to the existing total_loss terms.
  - Everything else (task/harmonic/symbolic/triplet losses, fusion,
    classification) is UNCHANGED from train_epoch_dual_branch.py -- only
    the reconstruction path is new, to isolate its effect cleanly rather
    than changing multiple things at once.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from grn_balladeer.model.classification_head import ClassificationHead, global_pool, split_real_imag
from grn_balladeer.model.grn_encoder import GRNEncoder, extract_resonance_frequency
from grn_balladeer.model.aux_branch_encoder import AuxBranchEncoder
from grn_balladeer.model.aux_branch_decoder import AuxBranchDecoder
from grn_balladeer.model.cross_attention_fusion import CrossAttentionFusion
from grn_balladeer.losses.harmonic_loss import harmonic_loss, all_pairs_edge_index, compute_consonance_degree
from grn_balladeer.losses.symbolic_loss import get_frontal_pairs, symbolic_implication_loss
from grn_balladeer.losses.triplet_loss import mine_batch_hard_triplets, triplet_loss
from grn_balladeer.losses.total_loss import total_loss


def train_epoch_dual_branch_with_reconstruction(
    encoder: GRNEncoder,
    resonance_head: nn.Module,
    aux_encoder: AuxBranchEncoder,
    aux_decoder: AuxBranchDecoder,
    fusion: CrossAttentionFusion,
    head: ClassificationHead,
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    labels: torch.Tensor,
    subject_ids: List[str],
    aux_vectors_by_subject: Dict[str, np.ndarray],
    ch_names: List[str],
    optimizer: torch.optim.Optimizer,
    symbolic_direction: str = "direct",
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    lambda3: float = 1.0,
    lambda4: float = 1.0,
    triplet_margin: float = 1.0,
    pool_method: str = "mean",
    class_weights: "torch.Tensor | None" = None,
) -> dict:
    """One training epoch, dual-branch with auxiliary reconstruction.
    Identical signature/behavior to train_epoch_dual_branch, plus
    aux_decoder and lambda4 (reconstruction loss weight, default 1.0
    matching the other untuned lambdas' convention throughout this
    project).

    Returns the same loss dict as train_epoch_dual_branch, plus
    'loss_recon'.
    """
    encoder.train(); resonance_head.train(); aux_encoder.train()
    aux_decoder.train(); fusion.train(); head.train()
    optimizer.zero_grad()

    frontal_pairs = get_frontal_pairs(ch_names)
    all_pairs = all_pairs_edge_index(batch[0][0].shape[0])

    z_joint_list = []
    z_aux_list = []
    aux_targets_list = []
    l_harm_terms = []
    l_symb_terms = []
    omega_per_sample = []
    last_omega = None

    for sample_idx in range(len(batch)):
        X_i, L_norm_i = batch[sample_idx]
        subject_id = subject_ids[sample_idx]

        h_i = encoder(X_i, L_norm_i)
        z_eeg_i = global_pool(split_real_imag(h_i), method=pool_method)

        omega_i = extract_resonance_frequency(h_i, resonance_head)
        omega_per_sample.append(omega_i)
        last_omega = omega_i
        l_harm_terms.append(harmonic_loss(omega_i, all_pairs))

        aux_vec_np = aux_vectors_by_subject[subject_id]
        aux_vec = torch.tensor(aux_vec_np, dtype=torch.float32, device=X_i.device).unsqueeze(0)
        z_aux_i = aux_encoder(aux_vec)
        z_aux_list.append(z_aux_i.squeeze(0))
        aux_targets_list.append(aux_vec.squeeze(0))

        z_joint_i, _, _ = fusion(z_eeg_i.unsqueeze(0), z_aux_i)
        z_joint_list.append(z_joint_i.squeeze(0))

    z_joint_batch = torch.stack(z_joint_list, dim=0)
    z_aux_batch = torch.stack(z_aux_list, dim=0)
    aux_targets_batch = torch.stack(aux_targets_list, dim=0)

    logits = head(z_joint_batch)
    l_task = nn.functional.cross_entropy(logits, labels, weight=class_weights)

    probs = torch.softmax(logits, dim=-1)
    for sample_idx, omega_i in enumerate(omega_per_sample):
        omega_frontal_i = omega_i[frontal_pairs[:, 0]]
        omega_frontal_j = omega_i[frontal_pairs[:, 1]]
        mu_ij = compute_consonance_degree(omega_frontal_i, omega_frontal_j)
        confidence_i = probs[sample_idx, labels[sample_idx]]
        l_symb_terms.append(symbolic_implication_loss(mu_ij, confidence_i, direction=symbolic_direction))

    l_harm = torch.stack(l_harm_terms).mean()
    l_symb = torch.stack(l_symb_terms).mean()

    triplets = mine_batch_hard_triplets(z_joint_batch, labels.cpu().numpy(), subject_ids)
    if triplets:
        l_triplet = triplet_loss(z_joint_batch, triplets, margin=triplet_margin)
    else:
        l_triplet = torch.tensor(0.0, device=z_joint_batch.device)

    # NEW: reconstruction loss on the auxiliary branch (Decoder from the
    # user's schema) -- reconstructs the raw aux vector from z_aux,
    # independent of the fused/classification path.
    aux_recon = aux_decoder(z_aux_batch)
    l_recon = nn.functional.mse_loss(aux_recon, aux_targets_batch)

    loss = total_loss(l_task, l_harm, l_symb, l_triplet=l_triplet,
                       lambda1=lambda1, lambda2=lambda2, lambda3=lambda3)
    loss = loss + lambda4 * l_recon
    loss.backward()
    optimizer.step()

    return {
        "loss_total": loss.item(),
        "loss_task": l_task.item(),
        "loss_harm": l_harm.item(),
        "loss_symb": l_symb.item(),
        "loss_triplet": l_triplet.item(),
        "loss_recon": l_recon.item(),
        "n_triplets_mined": len(triplets),
        "last_omega": last_omega.detach(),
    }
