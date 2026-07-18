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

WORKING ASSUMPTION (partially checked, not fully proven — flagged explicitly):
this module assumes the EEG recording's own t=0 (raw.times[0]) coincides
with generalTime=0 (session start). A same-subject check was run on
UB0136 (first 20s of real CGX data + real TAGS): the 4 events whose
general_time falls within that 20s window map to sample indices well
within [0, n_samples), consistent with the assumption — but this only
checks that indices land in-range, not that the alignment is precisely
correct sample-for-sample (e.g. no independent ERP-based or
protocol-documentation confirmation yet). Treat as "not contradicted",
not as "proven" — a stronger check (visible ERP pattern, or explicit
acquisition-protocol documentation of the trigger) is still worth doing
before relying on exact epoch boundaries in the final analysis.
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

    THIS IS THE CANONICAL parse_tags_file FOR THE PACKAGE. A second,
    near-duplicate implementation previously lived in
    training/behavioral_features.py (camelCase columns, e.g. 'flagType'
    instead of 'flag_type') — that copy has been removed; import this
    function from there instead. See docstring note in
    training/behavioral_features.py.

    Per-row parsing errors are skipped (not raised), matching the more
    defensive behavior of the former training/behavioral_features.py copy
    — a single malformed TAGS row should not abort loading an otherwise
    valid session.

    Returns a DataFrame with columns: timestamp_ms, general_time,
    reaction_time, reacted, correct, duplicated, flag_type, focus.
    flag_type=-1 rows are NOT filtered here (that is caller-specific
    business logic, e.g. align_events_to_eeg / extract_behavioral_features
    each decide independently whether/how to exclude them).
    """
    df = pd.read_csv(filepath)
    records: list[dict] = []

    for _, row in df.iterrows():
        try:
            v = ast.literal_eval(row["value"])
            r = v["reactionOrOmission"][0]
        except (KeyError, IndexError, ValueError, SyntaxError, TypeError):
            continue

        _rt = r.get("reactionTime", None)
        records.append({
            "timestamp_ms":    float(row["timestamp"]),
            "general_time":    float(r["generalTime"]),
            "reaction_time":   float(_rt) if _rt is not None else np.nan,
            "reacted":         r["reacted"] == "True",
            "correct":         r["correct"] == "True",
            "duplicated":      r["duplicated"] == "True",
            "flag_type":       int(r["flagType"][0]),
            "focus":           r["focus"],
        })

    return pd.DataFrame(
        records,
        columns=["timestamp_ms", "general_time", "reaction_time", "reacted",
                 "correct", "duplicated", "flag_type", "focus"],
    )


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
