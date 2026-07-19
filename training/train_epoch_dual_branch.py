"""
grn_balladeer.training.train_epoch_dual_branch
===================================================
Module 9 — "Integrate dual-branch into training loop". Extends
train_epoch.py (EEG-only) to fuse the auxiliary (behavioral + EDA)
branch via CrossAttentionFusion before classification, and adds the
triplet loss (Module 7b/8) on the fused embedding z_joint.

IMPORTANT DESIGN NOTE, not obvious from the individual modules: aux
features (behavioral, EDA) are extracted at SUBJECT/SESSION level
(one vector per subject - see training.behavioral_features and
training.eda_features), while the EEG branch is per-EPOCH (multiple
samples per subject). This loop therefore repeats the SAME aux vector
for every epoch belonging to a given subject - the aux branch currently
has no within-session temporal resolution, unlike the epoch-locked EEG
branch. This is a real architectural asymmetry, not a bug - flagging it
explicitly since it could matter for interpretation (the aux branch
cannot explain epoch-to-epoch variation, only subject-to-subject).

l_harm/l_symb are still computed from the EEG-only omega (extract_
resonance_frequency on the pre-fusion h_i) - the harmonic/symbolic
losses target the EEG branch's resonance structure specifically, per
the original architecture; fusion only changes what feeds the
classification head and the triplet loss.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from grn_balladeer.model.classification_head import ClassificationHead, global_pool, split_real_imag
from grn_balladeer.model.grn_encoder import GRNEncoder, extract_resonance_frequency
from grn_balladeer.model.aux_branch_encoder import AuxBranchEncoder
from grn_balladeer.model.cross_attention_fusion import CrossAttentionFusion
from grn_balladeer.losses.harmonic_loss import harmonic_loss, all_pairs_edge_index, compute_consonance_degree
from grn_balladeer.losses.symbolic_loss import get_frontal_pairs, symbolic_implication_loss
from grn_balladeer.losses.triplet_loss import mine_batch_hard_triplets, triplet_loss
from grn_balladeer.losses.total_loss import total_loss


def train_epoch_dual_branch(
    encoder: GRNEncoder,
    resonance_head: nn.Module,
    aux_encoder: AuxBranchEncoder,
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
    triplet_margin: float = 1.0,
    pool_method: str = "mean",
) -> dict:
    """One training epoch, dual-branch (EEG + aux, fused). batch/labels/
    ch_names as in train_epoch.py. subject_ids: (n_samples,) list, same
    length/order as batch - which subject each epoch belongs to (needed
    both to look up the right aux vector and for the triplet loss's
    anti-identity-leak rule). aux_vectors_by_subject: {subject_id:
    np.ndarray[12]} - precomputed via model.aux_branch_encoder.
    build_aux_vector, one entry per DISTINCT subject in this batch.

    Returns the same loss dict as train_epoch, plus 'loss_triplet' and
    'n_triplets_mined' (worth watching - if this drops to 0, the batch's
    subject/class composition doesn't support the anti-leak rule, see
    losses.triplet_loss.make_pk_batches for a sampler that avoids this).
    """
    encoder.train(); resonance_head.train(); aux_encoder.train(); fusion.train(); head.train()
    optimizer.zero_grad()

    frontal_pairs = get_frontal_pairs(ch_names)
    all_pairs = all_pairs_edge_index(batch[0][0].shape[0])

    z_joint_list = []
    l_harm_terms = []
    l_symb_terms = []
    omega_per_sample = []
    last_omega = None

    for sample_idx in range(len(batch)):
        X_i, L_norm_i = batch[sample_idx]
        subject_id = subject_ids[sample_idx]

        h_i = encoder(X_i, L_norm_i)  # SINGLE forward pass per sample (was computed twice before this fix)
        z_eeg_i = global_pool(split_real_imag(h_i), method=pool_method)

        omega_i = extract_resonance_frequency(h_i, resonance_head)
        omega_per_sample.append(omega_i)
        last_omega = omega_i
        l_harm_terms.append(harmonic_loss(omega_i, all_pairs))

        aux_vec = aux_vectors_by_subject[subject_id]
        z_aux_i = aux_encoder(torch.tensor(aux_vec, dtype=torch.float32, device=X_i.device).unsqueeze(0))

        z_joint_i, _, _ = fusion(z_eeg_i.unsqueeze(0), z_aux_i)
        z_joint_list.append(z_joint_i.squeeze(0))

    z_joint_batch = torch.stack(z_joint_list, dim=0)  # (n_samples, hidden_dim)
    logits = head(z_joint_batch)
    l_task = nn.functional.cross_entropy(logits, labels)

    # l_symb needs softmax(logits), which needs the full batch's logits first - this
    # second pass reuses omega_per_sample (no encoder call), not a fresh forward pass.
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
        l_triplet = torch.tensor(0.0, device=z_joint_batch.device)  # no valid anti-leak triplet in this batch - see docstring

    loss = total_loss(l_task, l_harm, l_symb, l_triplet=l_triplet, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3)
    loss.backward()
    optimizer.step()

    return {
        "loss_total": loss.item(),
        "loss_task": l_task.item(),
        "loss_harm": l_harm.item(),
        "loss_symb": l_symb.item(),
        "loss_triplet": l_triplet.item(),
        "n_triplets_mined": len(triplets),
        "last_omega": last_omega.detach(),
    }
