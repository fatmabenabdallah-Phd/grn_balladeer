"""
grn_balladeer.losses.triplet_loss
=====================================
Module 7b / 8 — triplet loss on z_fused (pooled, real-valued graph-level
embeddings), with batch-hard mining under an explicit anti-identity-leak
rule: anchor and positive MUST come from different subjects (same
class, different subject). Without this rule, the easiest "positive"
match is trivially the same subject's other epochs (shared EEG
idiosyncrasies), which the triplet loss would then reinforce as if it
were ADHD-relevant signal - literally the same subject-identity
confound flagged repeatedly this session, but baked into the loss
function instead of just the train/val split.

First real test of this rule needed >= 2 subjects/class - available
this session (UB0004, UB0022 = Control; UB0136, UB0023 = ADHD).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def make_pk_batches(
    subject_ids: List[str], labels: np.ndarray, P: int, K: int, seed: int = 42
) -> List[List[int]]:
    """P-K batch sampler: each batch contains P subjects x K samples/
    subject (indices into the original arrays), so every batch has
    enough same-subject AND cross-subject same-class pairs for
    mine_batch_hard_triplets to find valid anchor/positive/negative
    triplets without the anti-leak rule starving the batch.

    subject_ids: (n_samples,) subject id per sample (multiple epochs per
        subject expected). labels: (n_samples,) class label per sample.
    P: number of DISTINCT subjects per batch (not classes - a batch can
        mix subjects from different classes, mine_batch_hard_triplets
        handles same-class negatives itself).
    K: number of samples per chosen subject, drawn WITHOUT replacement if
        the subject has >= K samples, WITH replacement otherwise.

    Returns a list of batches, each a list of sample indices (length
    P*K).
    """
    rng = np.random.default_rng(seed)
    subject_ids_arr = np.array(subject_ids)
    unique_subjects = sorted(set(subject_ids))

    if len(unique_subjects) < P:
        raise ValueError(
            f"make_pk_batches: only {len(unique_subjects)} distinct subjects available, "
            f"need at least P={P}."
        )

    subject_to_indices = {s: np.where(subject_ids_arr == s)[0] for s in unique_subjects}

    shuffled_subjects = list(unique_subjects)
    rng.shuffle(shuffled_subjects)

    batches = []
    for i in range(0, len(shuffled_subjects), P):
        chosen_subjects = shuffled_subjects[i : i + P]
        if len(chosen_subjects) < P:
            break
        batch_idx = []
        for s in chosen_subjects:
            idx_pool = subject_to_indices[s]
            replace = len(idx_pool) < K
            chosen = rng.choice(idx_pool, size=K, replace=replace)
            batch_idx.extend(chosen.tolist())
        batches.append(batch_idx)

    return batches


def mine_batch_hard_triplets(
    embeddings: torch.Tensor, labels: np.ndarray, subject_ids: List[str]
) -> List[Tuple[int, int, int]]:
    """For each sample i (as anchor), finds the HARDEST valid positive
    (same class, DIFFERENT subject, maximum distance) and the HARDEST
    valid negative (different class, minimum distance), within the given
    batch. Skips an anchor if no valid positive exists.

    ANTI-IDENTITY-LEAK RULE: a candidate positive j is valid only if
    labels[j] == labels[i] AND subject_ids[j] != subject_ids[i].

    Returns a list of (anchor_idx, positive_idx, negative_idx) triplets,
    at most one per anchor that has a valid positive.
    """
    n = embeddings.shape[0]
    labels_arr = np.array(labels)
    subject_ids_arr = np.array(subject_ids)

    with torch.no_grad():
        dists = torch.cdist(embeddings, embeddings, p=2)

    triplets = []
    for i in range(n):
        same_class = labels_arr == labels_arr[i]
        diff_subject = subject_ids_arr != subject_ids_arr[i]
        valid_pos_mask = same_class & diff_subject
        valid_pos_mask[i] = False

        diff_class = labels_arr != labels_arr[i]
        valid_neg_mask = diff_class

        if not valid_pos_mask.any() or not valid_neg_mask.any():
            continue

        pos_candidates = np.where(valid_pos_mask)[0]
        neg_candidates = np.where(valid_neg_mask)[0]

        pos_dists = dists[i, pos_candidates]
        neg_dists = dists[i, neg_candidates]

        hardest_pos = pos_candidates[torch.argmax(pos_dists).item()]
        hardest_neg = neg_candidates[torch.argmin(neg_dists).item()]

        triplets.append((i, int(hardest_pos), int(hardest_neg)))

    return triplets


def triplet_loss(
    embeddings: torch.Tensor, triplets: List[Tuple[int, int, int]], margin: float = 1.0
) -> torch.Tensor:
    """Standard margin triplet loss: mean(relu(d(a,p) - d(a,n) + margin))
    over the given (anchor, positive, negative) index triplets.
    """
    if not triplets:
        raise ValueError(
            "triplet_loss: no triplets given - mine_batch_hard_triplets likely found no "
            "valid (same-class, different-subject) positive for any anchor in this batch. "
            "Check P/K batch composition (need >= 2 subjects per class in the batch)."
        )

    anchor_idx = torch.tensor([t[0] for t in triplets], dtype=torch.long)
    pos_idx = torch.tensor([t[1] for t in triplets], dtype=torch.long)
    neg_idx = torch.tensor([t[2] for t in triplets], dtype=torch.long)

    d_ap = F.pairwise_distance(embeddings[anchor_idx], embeddings[pos_idx], p=2)
    d_an = F.pairwise_distance(embeddings[anchor_idx], embeddings[neg_idx], p=2)

    return F.relu(d_ap - d_an + margin).mean()
