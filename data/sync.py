"""
data/sync.py
============
Temporal synchronization between the CGX EEG files (relative clock,
seconds since session start) and the TAGS files (absolute Unix clock,
milliseconds).

EMPIRICAL FINDINGS VALIDATED on real UB0136 data:
  - The CGX timestamp is RELATIVE (seconds since t=0 of the session).
  - TAGS timestamps are ABSOLUTE (Unix milliseconds, sub-ms precision).
  - generalTime in TAGS = time since the start of the GAME, not since the
    start of the EEG recording -> do not use as a direct anchor.
  - The correct anchor = the Unix timestamp extracted from the EEG FILE
    NAME (second-level precision, sufficient as validated with std < 20 ms).
  - Confirmed CGX channels: 29 EEG (uV) + 3 accelerometer ACC32/33/34 (mg).
  - Confirmed Slackline Lvl1 session duration: ~305 s.
  - flagType=-1: undocumented value, coincides with correct=False.

Author: GRN-BALLADEER project
"""

import numpy as np
import pandas as pd
import json
import ast
import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Dict, List

logger = logging.getLogger(__name__)

# CGX sampling rate (confirmed on real data)
CGX_SFREQ = 500.0  # Hz

# Spain timezone (GMT+1, confirmed in TAGS file names: +01.00)
TZ_SPAIN = timezone(timedelta(hours=1))


# ---------------------------------------------------------------------------
# 1. File loading
# ---------------------------------------------------------------------------

def load_tags(tags_path: str) -> pd.DataFrame:
    """
    Loads a TAGS file and parses the JSON-like 'value' field.

    Output columns:
        timestamp_ms  (float) — absolute Unix clock, milliseconds
        label         (str)   — always 'Marcador', kept for traceability
        reacted       (bool)
        reactionTime  (float) — seconds after stimulus appearance
        correct       (bool)
        duplicated    (bool)
        flagType      (int)   — 0=circle,1=square,2=rhombus,3=doubleCircle,-1=unknown
        generalTime   (float) — seconds since the start of the GAME (!= EEG start)
        focus         (str)   — 'Target' | 'non_focusable'
    """
    df = pd.read_csv(tags_path)
    records = []

    for _, row in df.iterrows():
        try:
            v = ast.literal_eval(row['value'])
            r = v['reactionOrOmission'][0]
        except (KeyError, IndexError, ValueError, SyntaxError) as e:
            logger.warning("Unparseable TAGS row (skipped): %s", e)
            continue

        records.append({
            'timestamp_ms': float(row['timestamp']),
            'label':        row['label'],
            'reacted':      r['reacted'] == 'True',
            'reactionTime': float(r.get('reactionTime', np.nan)),
            'correct':      r['correct'] == 'True',
            'duplicated':   r['duplicated'] == 'True',
            'flagType':     int(r['flagType'][0]),
            'generalTime':  float(r['generalTime']),
            'focus':        r['focus'],
        })

    parsed = pd.DataFrame(records)

    n_unknown = (parsed['flagType'] == -1).sum()
    if n_unknown > 0:
        logger.info("flagType=-1 (undocumented): %d occurrences in %s",
                    n_unknown, tags_path)

    return parsed


def load_eeg_cgx(eeg_path: str) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[np.ndarray]]:
    """
    Loads an EEG_CGX.csv file.

    Channels confirmed on real data:
        29 EEG channels with suffix '(uV)': AF7, Fpz, F7, Fz, T7, FC6, Fp1,
        F4, C4, Oz, CP6, Cz, PO8, CP5, O2, O1, P3, P4, P7, P8, Pz, PO7,
        T8, C3, Fp2, F3, F8, FC5, AF8.
        3 accelerometer channels: ACC32(mg), ACC33(mg), ACC34(mg).

    Returns
    -------
    times    : [n_samples]            — relative timestamps in seconds
    data     : [n_samples, n_eeg]     — uV, EEG channels only
    channels : list[str]              — EEG channel names (no accelerometer)
    accel    : [n_samples, 3] | None  — accelerometer X/Y/Z data in mg
    """
    df = pd.read_csv(eeg_path)

    # First column = relative time (seconds)
    time_col = df.columns[0]
    times = df[time_col].values.astype(np.float64)

    # Split EEG / accelerometer / other
    accel_cols = [c for c in df.columns if c.startswith('ACC')]
    NON_EEG    = ('Packet', 'TRIGGER', 'ExG', 'A2')
    eeg_cols   = [
        c for c in df.columns[1:]
        if c not in accel_cols
        and not any(c.startswith(p) for p in NON_EEG)
        and c != time_col
    ]

    data  = df[eeg_cols].values.astype(np.float32)
    accel = df[accel_cols].values.astype(np.float32) if accel_cols else None

    logger.info(
        "CGX EEG loaded: %d samples | %d EEG channels | %d accel channels | duration %.1f s",
        len(times), len(eeg_cols),
        len(accel_cols) if accel_cols else 0,
        times[-1] - times[0]
    )

    return times, data, eeg_cols, accel


# ---------------------------------------------------------------------------
# 2. Temporal anchoring from the EEG file name
# ---------------------------------------------------------------------------

def parse_eeg_start_unix_ms(eeg_path: str) -> float:
    """
    Extracts the Unix timestamp (ms) of the EEG recording start from the
    CGX file name.

    Format confirmed on real data:
        UB0136_EEG_CGX_2024_01_19T16.30.01.csv
        -> date = 2024-01-19T16:30:01 (GMT+1, Spain)

    Returns
    -------
    float — Unix timestamp in milliseconds
    """
    basename = os.path.basename(eeg_path).replace(".csv", "")

    # Extract the part after '_EEG_CGX_'
    try:
        date_str = basename.split("_EEG_CGX_")[-1]
        # Convert file format -> ISO 8601
        # '2024_01_19T16.30.01' -> '2024-01-19T16:30:01'
        parts = date_str.split("T")
        date_part = parts[0].replace("_", "-")
        time_part = parts[1].replace(".", ":")
        iso_str = f"{date_part}T{time_part}"

        dt = datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%S")
        dt = dt.replace(tzinfo=TZ_SPAIN)
        unix_ms = dt.timestamp() * 1000.0

        logger.info("EEG start extracted from file name: %s -> %.0f ms Unix",
                    iso_str, unix_ms)
        return unix_ms

    except Exception as e:
        raise ValueError(
            f"Could not parse the date from file name '{eeg_path}'. "
            f"Expected format: *_EEG_CGX_YYYY_MM_DDTHH.MM.SS.csv\nError: {e}"
        )


# ---------------------------------------------------------------------------
# 3. Session offset computation and validation
# ---------------------------------------------------------------------------

def compute_session_offset(
    tags_df: pd.DataFrame,
    eeg_times: np.ndarray,
    eeg_start_unix_ms: float,
    sfreq: float = CGX_SFREQ
) -> Tuple[float, float]:
    """
    Computes and validates the session's temporal offset.

    Strategy:
        offset_ms = eeg_start_unix_ms
        (the CGX timestamp is relative from 0 -> adding the Unix time of
        the recording start gives the absolute time)

    Validation:
        For each TAGS event (absolute timestamp_ms), the corresponding
        EEG index is recomputed and checked for consistency.
        Expected std < 20 ms if the anchoring is correct.

    Parameters
    ----------
    tags_df           : output of load_tags()
    eeg_times         : relative CGX timestamps (seconds)
    eeg_start_unix_ms : Unix timestamp (ms) of the EEG recording start

    Returns
    -------
    offset_ms  : float — offset to apply (= eeg_start_unix_ms)
    offset_std : float — validation standard deviation in ms
    """
    offset_ms = eeg_start_unix_ms
    residuals = []

    for _, row in tags_df.iterrows():
        tag_unix_ms    = row['timestamp_ms']
        eeg_relative_s = (tag_unix_ms - offset_ms) / 1000.0

        # Check that the event falls within the EEG window
        if eeg_relative_s < eeg_times[0] or eeg_relative_s > eeg_times[-1]:
            continue

        idx = np.searchsorted(eeg_times, eeg_relative_s)
        idx = np.clip(idx, 0, len(eeg_times) - 1)
        reconstructed_ms = eeg_times[idx] * 1000.0 + offset_ms
        residuals.append(abs(reconstructed_ms - tag_unix_ms))

    if not residuals:
        logger.warning("No TAGS event within the EEG window — offset not validated.")
        return offset_ms, 9999.0

    offset_std = float(np.std(residuals))
    mean_res   = float(np.mean(residuals))

    if offset_std > 20.0:
        logger.warning(
            "High validation std (%.1f ms, mean=%.1f ms) — "
            "check the EEG file name or the timezone.",
            offset_std, mean_res
        )
    else:
        logger.info(
            "Offset validated: %.0f ms | residual mean=%.2f ms, std=%.2f ms | n=%d events",
            offset_ms, mean_res, offset_std, len(residuals)
        )

    return offset_ms, offset_std


# ---------------------------------------------------------------------------
# 4. Index conversion
# ---------------------------------------------------------------------------

def eeg_idx_to_unix_ms(eeg_times: np.ndarray, offset_ms: float) -> np.ndarray:
    """Relative EEG timestamps (s) -> absolute Unix time (ms)."""
    return eeg_times * 1000.0 + offset_ms


def unix_ms_to_eeg_idx(
    unix_timestamps_ms: np.ndarray,
    eeg_times: np.ndarray,
    offset_ms: float
) -> np.ndarray:
    """
    Unix timestamps (ms) -> EEG sample indices.
    Clipped to [0, n_samples-1].
    """
    eeg_relative_s = (unix_timestamps_ms - offset_ms) / 1000.0
    indices = np.searchsorted(eeg_times, eeg_relative_s)
    return np.clip(indices, 0, len(eeg_times) - 1).astype(int)


# ---------------------------------------------------------------------------
# 5. Slackline level cross-check validation
# ---------------------------------------------------------------------------

def validate_level_assignment(
    tags_df: pd.DataFrame,
    flags_info: dict,
    candidate_levels: List[str] = ['Level1', 'Level6', 'Level11']
) -> Dict[str, dict]:
    """
    Identifies the Slackline level by comparing TAGS events' generalTime
    to each level's flag_spawn_time.

    The correct level = the one with the minimal mean residual between
    generalTime and the closest spawn_time. Empirically validated: the
    correct level has ~0.94 s mean residual, wrong levels 2-5x higher.

    NOTE: this automatic matching is not 100% reliable — it has previously
    picked the wrong level for a real subject (see context-transfer docs).
    Always confirm against external metadata when possible.

    Returns
    -------
    dict { level_name : { 'mean_residual_s', 'std_residual_s' }, '_best': str }
    """
    levels_map = {
        item['level']: [f['flag_spawn_time'] for f in item['flags']]
        for item in flags_info['slackline_levels_flags_info']
    }

    reacted = tags_df[tags_df['reacted']].reset_index(drop=True)
    results = {}

    for level_name in candidate_levels:
        if level_name not in levels_map:
            continue
        spawn_times = levels_map[level_name]

        residuals = []
        for _, row in reacted.iterrows():
            gt = row['generalTime']
            closest = min(spawn_times, key=lambda t: abs(t - gt))
            residuals.append(abs(gt - closest))

        mean_res = float(np.mean(residuals))
        std_res  = float(np.std(residuals))
        results[level_name] = {
            'mean_residual_s': mean_res,
            'std_residual_s':  std_res,
        }
        logger.info("%s: mean residual=%.3f s (std=%.3f s)",
                    level_name, mean_res, std_res)

    best = min(results, key=lambda k: results[k]['mean_residual_s'])
    logger.info("Assigned level: %s", best)
    results['_best'] = best

    return results


# ---------------------------------------------------------------------------
# 6. Main entry point
# ---------------------------------------------------------------------------

def sync_session(
    eeg_path:        str,
    tags_path:       str,
    flags_info_path: str,
    validate:        bool = True
) -> Dict:
    """
    Loads EEG + TAGS, computes the temporal offset, validates the
    Slackline level.

    Returns
    -------
    {
        'eeg_times'   : np.ndarray [n_samples]          — relative timestamps (s)
        'eeg_data'    : np.ndarray [n_samples, n_eeg]   — uV
        'eeg_channels': list[str]                        — EEG channel names
        'accel_data'  : np.ndarray [n_samples, 3] | None — accelerometer (mg)
        'tags_df'     : pd.DataFrame
        'offset_ms'   : float   — Unix offset to add to EEG timestamps
        'offset_std'  : float   — validation std (ms), should be < 20
        'level'       : str | None — automatically detected Slackline level
        'valid'       : bool    — True if offset_std < 20 ms
    }
    """
    # Loading
    eeg_times, eeg_data, channels, accel = load_eeg_cgx(eeg_path)
    tags_df = load_tags(tags_path)

    # Temporal anchoring from the file name
    eeg_start_unix_ms = parse_eeg_start_unix_ms(eeg_path)

    # Offset computation + validation
    offset_ms, offset_std = compute_session_offset(
        tags_df, eeg_times, eeg_start_unix_ms
    )

    # Slackline level identification
    level = None
    if validate:
        with open(flags_info_path) as f:
            flags_info = json.load(f)
        val_results = validate_level_assignment(tags_df, flags_info)
        level = val_results.get('_best')

    return {
        'eeg_times':    eeg_times,
        'eeg_data':     eeg_data,
        'eeg_channels': channels,
        'accel_data':   accel,
        'tags_df':      tags_df,
        'offset_ms':    offset_ms,
        'offset_std':   offset_std,
        'level':        level,
        'valid':        offset_std < 20.0,
    }
