"""
grn_balladeer.data.subject_files
===================================
Generalizes subject-file discovery from the 4 hand-picked subjects
(UB0004/UB0022/UB0136/UB0023, hardcoded paths in earlier sessions) to
the full 158-folder Drive dataset, based on the real directory tree
inspected this session (`balladeer_tree.csv`, Drive-side listing).

Why this exists: `data.build_dataset.build_subject_dataset` takes exact
file paths as arguments -- it does not search for them. Something has
to turn (subject_id, level) into those paths across 158 subjects, and
the real filenames are NOT fully regular. Confirmed irregularities
(counted directly on the real tree, not assumed):

  - TAGS files: 3 filename shapes --
      UBxxxx_TAGS_<date>+<tz>.csv        (majority)
      UBxxxx_TAGS_<date>.csv             (no timezone suffix)
      UBxxxx_TAGS_CGX_<date>.csv         (extra "_CGX_" token, 14 files)
    A rigid f-string reconstruction of the filename would silently miss
    the third shape. Fixed here with a tolerant glob
    (`{subject_id}_TAGS*.csv`) instead of an exact name guess.
  - EEG_CGX files: single consistent shape, no irregularity found.
  - Not every subject/level has a CGX file -- TAGS coverage (151-154
    per level) exceeds CGX coverage (140 per level, all 3 levels) per
    the dataset paper's own Table 1. Missing CGX for a given
    subject/level is EXPECTED, not an error -- must return None, not
    raise.
  - Verified (this session, via glob against a full mirror of the real
    158-subject tree): no case of >1 CGX or >1 TAGS file matching for
    the same subject/level/session -- the tolerant glob does not
    introduce ambiguity, at least on the tree inspected.

Only Slackline (CGX + TAGS) is covered here. AttentionRobotsDesktop
(Emotiv, eye-tracking) and Cognifit (Emotiv only) use a different
naming convention (`EPOCX`/`EPOCPLUS`, `EYE_TRACKING_DATA`, `GAME_DATA`)
and are NOT needed yet -- current focus is session 2 / Slackline / CGX
first, per this session's plan (session 1 / Emotiv / Robots+CogniFit
comes later, for cross-session generalization). Extend this module
with `find_robots_session_files` / `find_cognifit_session_files` when
that phase starts, rather than guessing their shape now.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List, Optional

# Slackline level names as they appear in folder names (SlacklineLvl<LEVEL>)
# and in slackline_flags_info.json's "level" field (Level<LEVEL>).
SLACKLINE_LEVELS = ("1", "6", "11")


@dataclass
class SlacklineSessionFiles:
    """Resolved file paths for one subject/level/session.

    cgx_path is None when no CGX file exists for this session (expected
    for some subjects per the dataset's own documented coverage gaps --
    NOT an error, do not raise on this alone).
    """
    subject_id: str
    level: str                  # "1", "6", or "11" (matches SlacklineLvl<level>)
    session_dir: str             # the UnixSessionDate folder name
    cgx_path: Optional[str]
    tags_path: Optional[str]


class SubjectFileDiscoveryError(RuntimeError):
    """Raised only on genuine ambiguity (>1 matching file) -- never for
    a merely absent file, which is expected and handled by returning
    None instead."""


def _unique_or_raise(matches: List[str], subject_id: str, level: str,
                      session_dir: str, kind: str) -> Optional[str]:
    if len(matches) == 0:
        return None
    if len(matches) == 1:
        return matches[0]
    raise SubjectFileDiscoveryError(
        f"Ambiguous {kind} match for {subject_id} SlacklineLvl{level} "
        f"session {session_dir}: {matches}"
    )


def find_slackline_sessions(
    dataset_root: str,
    subject_id: str,
    level: str,
) -> List[SlacklineSessionFiles]:
    """Finds all Slackline sessions for one subject/level under
    dataset_root/<subject_id>/SlacklineLvl<level>/<UnixSessionDate>/.

    Returns an empty list if the subject has no SlacklineLvl<level>
    folder at all (subject didn't do this level -- expected, not an
    error). Normally returns exactly one session (one UnixSessionDate
    folder per subject/level in the tree inspected this session), but
    returns a list rather than assuming that, since nothing in the
    dataset paper guarantees exactly one session per subject/level.
    """
    task_dir = os.path.join(dataset_root, subject_id, f"SlacklineLvl{level}")
    if not os.path.isdir(task_dir):
        return []

    session_dirs = sorted(
        d for d in os.listdir(task_dir)
        if os.path.isdir(os.path.join(task_dir, d))
    )

    results = []
    for session_dir in session_dirs:
        sdir = os.path.join(task_dir, session_dir)

        # Tolerant globs -- catch all 3 real TAGS filename shapes and
        # the single consistent EEG_CGX shape, without over-matching
        # another subject's files (subject_id prefix anchors the match).
        cgx_matches = glob.glob(os.path.join(sdir, f"{subject_id}_EEG_CGX_*.csv"))
        tags_matches = glob.glob(os.path.join(sdir, f"{subject_id}_TAGS*.csv"))

        cgx_path = _unique_or_raise(cgx_matches, subject_id, level, session_dir, "CGX")
        tags_path = _unique_or_raise(tags_matches, subject_id, level, session_dir, "TAGS")

        results.append(SlacklineSessionFiles(
            subject_id=subject_id,
            level=level,
            session_dir=session_dir,
            cgx_path=cgx_path,
            tags_path=tags_path,
        ))

    return results


def build_dataset_file_index(
    dataset_root: str,
    subject_ids: List[str],
    levels: List[str] = SLACKLINE_LEVELS,
) -> List[SlacklineSessionFiles]:
    """Runs find_slackline_sessions across many subjects/levels and
    returns a flat list -- the natural input to a "build once per
    subject" pass before cross_validation.train_fold (per cross_
    validation.py's own docstring: build each subject's dataset ONCE,
    not once per fold).

    Sessions with cgx_path=None ARE included in the returned list (not
    silently dropped) -- callers building a CGX-only dataset must filter
    on `.cgx_path is not None` themselves and should log/report how many
    were dropped, so a coverage gap is visible rather than silent.
    """
    index = []
    for subject_id in subject_ids:
        for level in levels:
            index.extend(find_slackline_sessions(dataset_root, subject_id, level))
    return index


def summarize_coverage(index: List[SlacklineSessionFiles]) -> dict:
    """Quick coverage report: how many sessions have CGX vs TAGS-only,
    broken down by level. Meant to be printed once after building the
    index, so a coverage gap (expected per the dataset's own Table 1,
    ~140/151-154 CGX/TAGS ratio) is visible and not just silently
    dropped downstream."""
    by_level: dict = {}
    for entry in index:
        d = by_level.setdefault(entry.level, {"total": 0, "with_cgx": 0, "with_tags": 0})
        d["total"] += 1
        if entry.cgx_path is not None:
            d["with_cgx"] += 1
        if entry.tags_path is not None:
            d["with_tags"] += 1
    return by_level
