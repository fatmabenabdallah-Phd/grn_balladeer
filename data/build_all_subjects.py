"""
grn_balladeer.data.build_all_subjects
=========================================
Week 6 extension — "extend to 138 subjects" (Level1-only coverage gave
114 usable subjects; some of the remaining label-usable subjects have
CGX only at Level6 or Level11, not Level1). Builds on top of
subject_files.py's multi-level discovery (already covered all 3 levels)
and build_dataset.py's build_subject_dataset (already level-agnostic,
takes `level` as a parameter) — the piece that was missing is deciding,
PER SUBJECT, which single level to actually use.

DESIGN CHOICE: exactly ONE level per subject, in priority order
Level1 > Level6 > Level11 (first one with a real CGX file wins). NOT
combining multiple levels' epochs for subjects that have CGX at more
than one level (e.g. SUBJ_C-type cases below) — mixing epochs from
different task-difficulty levels within one subject would introduce
within-subject heterogeneity (different flag-type distributions, see
Week 1-2 findings: Level1 is circle-heavy [15,6,7,5], Level6/11 are
more balanced) on top of the existing subject-level aux-feature
asymmetry already flagged in train_epoch_dual_branch.py. Priority order
matches what's already validated on real data (all 4 hand-picked
subjects so far are Level1) rather than introducing an unvalidated new
level as the default for subjects that already work. Revisit as a
deliberate ablation (does per-subject level choice matter?) rather than
silently mixing levels now.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from grn_balladeer.data.subject_files import find_slackline_sessions, SlacklineSessionFiles
from grn_balladeer.data.build_dataset import build_subject_dataset

LEVEL_PRIORITY = ("1", "6", "11")


@dataclass
class SubjectLevelResolution:
    subject_id: str
    level: Optional[str]          # None if no level had a usable CGX+TAGS pair
    cgx_path: Optional[str]
    tags_path: Optional[str]


def resolve_subject_level(
    dataset_root: str, subject_id: str, level_priority: Tuple[str, ...] = LEVEL_PRIORITY
) -> SubjectLevelResolution:
    """Picks the first level (in priority order) for which this subject
    has BOTH a CGX and a TAGS file. Returns level=None (not an error) if
    no level qualifies - some subjects genuinely have TAGS-only coverage
    at every level (expected per the dataset's own CGX/TAGS coverage gap,
    see subject_files.py's docstring), and this must be filterable
    downstream rather than crashing the whole batch build.
    """
    for level in level_priority:
        sessions = find_slackline_sessions(dataset_root, subject_id, level)
        for session in sessions:
            if session.cgx_path is not None and session.tags_path is not None:
                return SubjectLevelResolution(subject_id, level, session.cgx_path, session.tags_path)
    return SubjectLevelResolution(subject_id, None, None, None)


def build_all_subjects_datasets(
    dataset_root: str,
    subject_ids: List[str],
    flags_path: str,
    level_priority: Tuple[str, ...] = LEVEL_PRIORITY,
    verbose: bool = True,
    checkpoint_dir: Optional[str] = None,
) -> Tuple[Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]], Dict[str, str], List[str]]:
    """Builds (X_i, L_norm_i) graph datasets for as many of subject_ids
    as have a usable CGX+TAGS pair at SOME level, using resolve_subject_
    level's priority order.

    checkpoint_dir: NEW this session -- if given, checks for an existing
    checkpoint before calling build_subject_dataset, and saves a new one
    after building. Uses the EXACT SAME filename convention as the
    Colab driver's Stage 3 (real_dataset_{subject_id}_L{level}.pt), so
    if this points at the same
    /content/drive/MyDrive/BALLADEER_GRN_checkpoints/datasets directory
    already used for the 114 Level1-only subjects, those are genuinely
    reloaded instantly rather than rebuilt -- this only matters for
    subjects resolve_subject_level assigns to Level1 (the 114 already
    covered); subjects newly reached via the Level6/11 fallback have no
    prior checkpoint and will be built fresh regardless.
    If None (default), no caching -- behavior identical to before this
    change.

    Returns (dataset_by_subject, level_used_by_subject, unusable_subject_ids):
    - dataset_by_subject: ready for training.cross_validation.run_cross_
      validation, exactly as before, just covering more subjects now.
    - level_used_by_subject: which level each usable subject came from -
      worth logging/reporting (e.g. as a column alongside sex/age_bin in
      any disaggregated evaluation, in case level choice itself turns out
      to correlate with anything) rather than discarding this information.
    - unusable_subject_ids: subjects with NO CGX at any level - EXPECTED
      for some (per the dataset's documented CGX/TAGS coverage gap), not
      a bug. Report this count rather than silently dropping them.
    """
    import os

    dataset_by_subject: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
    level_used_by_subject: Dict[str, str] = {}
    unusable_subject_ids: List[str] = []
    loaded_from_checkpoint = 0
    newly_built = 0

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    for subject_id in subject_ids:
        resolution = resolve_subject_level(dataset_root, subject_id, level_priority)
        if resolution.level is None:
            unusable_subject_ids.append(subject_id)
            continue

        ckpt_path = None
        if checkpoint_dir is not None:
            ckpt_path = os.path.join(
                checkpoint_dir, f"real_dataset_{subject_id}_L{resolution.level}.pt"
            )
            if os.path.exists(ckpt_path):
                # weights_only=False: PyTorch 2.6+ changed the default to True,
                # which rejects numpy arrays embedded in the checkpoint (our
                # complex-valued graph tensors were built via numpy before
                # converting to torch). These are our own trusted checkpoints
                # written by this same codebase, not third-party files, so
                # explicitly opting into the pre-2.6 behavior is appropriate here.
                dataset_by_subject[subject_id] = torch.load(ckpt_path, weights_only=False)
                level_used_by_subject[subject_id] = resolution.level
                loaded_from_checkpoint += 1
                continue

        try:
            dataset = build_subject_dataset(
                resolution.cgx_path, flags_path, level=f"Level{resolution.level}"
            )
        except Exception as e:
            # A real per-subject preprocessing failure (e.g. a corrupt file) should
            # not silently vanish OR crash the whole 138-subject batch - surface it.
            if verbose:
                print(f"[build_all_subjects_datasets] FAILED for {subject_id} "
                      f"(Level{resolution.level}): {e}")
            unusable_subject_ids.append(subject_id)
            continue

        if ckpt_path is not None:
            torch.save(dataset, ckpt_path)

        dataset_by_subject[subject_id] = dataset
        level_used_by_subject[subject_id] = resolution.level
        newly_built += 1

    if verbose:
        from collections import Counter
        level_counts = Counter(level_used_by_subject.values())
        cache_note = (
            f" ({loaded_from_checkpoint} depuis le cache Drive, {newly_built} nouveaux)"
            if checkpoint_dir is not None else ""
        )
        print(f"[build_all_subjects_datasets] {len(dataset_by_subject)}/{len(subject_ids)} "
              f"subjects usable{cache_note}. Level breakdown: {dict(level_counts)}. "
              f"{len(unusable_subject_ids)} subjects with no usable CGX at any level.")

    return dataset_by_subject, level_used_by_subject, unusable_subject_ids
