"""
grn_balladeer.preprocessing.event_alignment
==============================================
Module 2b (part 2, step 4) — TAGS parsing and event-to-EEG-sample
alignment.

CRITICAL FINDING (verified on real UB0136 TAGS file + slackline_flags_info.json):
the raw 'timestamp' column in a TAGS csv is a Unix millisecond timestamp
from the web/game client's own clock — it is NOT in the same clock
domain as the CGX device's internal 'timestamps' column (the CGX has no
hardware trigger and its clock is not confirmed to be wall-clock synced).

The field 'generalTime' inside the parsed 'value' dict IS in the correct
domain: empirically, generalTime - reactionTime lines up closely with
the flag_spawn_time values in slackline_flags_info.json (e.g. row 0:
generalTime=2.551, reactionTime=0.900 -> estimated spawn=1.651, closest
real flag at t=2s; row 5: generalTime=65.983, reactionTime=6.322 ->
estimated spawn=59.66, closest real flag at t=60s). This confirms
generalTime is session/level-relative time (seconds since the level
started), the same domain used by the game engine to schedule flags.

WORKING ASSUMPTION (NOT YET CROSS-VALIDATED — flagged explicitly):
this module assumes the EEG recording's own t=0 (raw.times[0]) coincides
with generalTime=0 (session start). This has NOT been verified against
a matched CGX+TAGS pair for the same subject/session in this codebase —
the CGX file available so far (UB0004) and this TAGS file (UB0136) are
from different subjects and cannot be cross-checked against each other.
Confirm this assumption on a same-subject CGX+TAGS pair before trusting
epoch boundaries produced from this alignment in any real analysis.
"""

from __future__ import annotations

import ast
from typing import List

import numpy as np
import pandas as pd


def parse_tags_file(filepath: str) -> pd.DataFrame:
    """Parses a raw TAGS csv export. The 'value' column holds a Python
    dict LITERAL (single-quoted, e.g. {'reacted': 'True', ...}) — this is
    NOT valid JSON, so json.loads would fail; ast.literal_eval is
    required. Flattens the first entry of 'reactionOrOmission' into
    columns alongside the original 'timestamp' (Unix ms, game-client
    clock — kept for reference, not used for EEG alignment).

    Returns a DataFrame with columns: timestamp_ms, reacted, reaction_time,
    correct, duplicated, flag_type, general_time, focus.
    """
    df = pd.read_csv(filepath)
    parsed = df["value"].apply(ast.literal_eval)
    events = [p["reactionOrOmission"][0] for p in parsed]
    events_df = pd.DataFrame(events)

    events_df = events_df.rename(
        columns={"reactionTime": "reaction_time", "generalTime": "general_time", "flagType": "flag_type"}
    )
    events_df["timestamp_ms"] = df["timestamp"].values
    events_df["reacted"] = events_df["reacted"].map({"True": True, "False": False})
    events_df["correct"] = events_df["correct"].map({"True": True, "False": False})
    events_df["duplicated"] = events_df["duplicated"].map({"True": True, "False": False})

    return events_df[
        ["timestamp_ms", "general_time", "reaction_time", "reacted", "correct", "duplicated", "flag_type", "focus"]
    ]


def align_events_to_eeg(
    tags_df: pd.DataFrame, sfreq: float, session_start_general_time: float = 0.0
) -> np.ndarray:
    """Converts each event's 'general_time' (seconds, session-relative —
    see module docstring for why this field and not the raw Unix
    'timestamp') into an EEG sample index, assuming raw.times[0]
    corresponds to session_start_general_time (default 0.0, i.e. the
    session/level start).

    Returns an integer array of sample indices, one per row of tags_df,
    in the same order. Raises if any resulting index is negative (event
    timestamped before the assumed session start — likely means
    session_start_general_time is wrong for this file).
    """
    elapsed = tags_df["general_time"].to_numpy() - session_start_general_time
    sample_indices = np.round(elapsed * sfreq).astype(int)

    if (sample_indices < 0).any():
        n_bad = int((sample_indices < 0).sum())
        raise ValueError(
            f"{n_bad} event(s) map to a negative EEG sample index — "
            "session_start_general_time is likely incorrect for this "
            "file/session. Do not silently clip; investigate first."
        )

    return sample_indices


def estimate_flag_spawn_time(tags_df: pd.DataFrame) -> np.ndarray:
    """Convenience helper used to CROSS-CHECK general_time against
    slackline_flags_info.json: estimated flag spawn time = general_time
    - reaction_time. Not used in the main alignment path, only for
    validating the generalTime hypothesis against a known flags file."""
    return tags_df["general_time"].to_numpy() - tags_df["reaction_time"].to_numpy()
