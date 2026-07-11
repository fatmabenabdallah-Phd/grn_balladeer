"""
data/sync.py
============
Synchronisation temporelle entre les fichiers EEG CGX (horloge relative,
secondes depuis le début de session) et les fichiers TAGS (horloge Unix
absolue, millisecondes).

DÉCOUVERTES EMPIRIQUES VALIDÉES (à ne pas modifier sans re-valider sur données réelles) :
  - Le timestamp CGX est RELATIF (secondes depuis t=0 de la session).
  - Les timestamps TAGS sont ABSOLUS (Unix millisecondes, précision sous-ms).
  - L'offset session-to-Unix est quasi-constant PAR session (std ~3-4 ms)
    mais DIFFÈRE entre sessions (exemples observés : 1.55s / 1.65s / 1.83s).
  - Le nom de fichier n'a qu'une précision à la seconde → TROP IMPRÉCIS pour
    servir d'ancre. L'ancre doit venir du fichier TAGS lui-même.
  - Validation : le bon niveau de Slackline donne un écart moyen ~0.94s,
    max ~0.95s, très stable. Un mauvais niveau donne 2-5x plus d'écart
    avec forte dispersion.

Auteur : GRN-BALLADEER project
"""

import numpy as np
import pandas as pd
import json
import ast
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)

# Fréquence d'échantillonnage CGX (confirmée sur données réelles)
CGX_SFREQ = 500.0  # Hz


# ---------------------------------------------------------------------------
# 1. Chargement des fichiers
# ---------------------------------------------------------------------------

def load_tags(tags_path: str) -> pd.DataFrame:
    """
    Charge un fichier TAGS et parse le champ JSON 'value'.

    Colonnes en sortie :
        timestamp_ms (float) — horloge Unix absolue, millisecondes
        label (str)          — toujours 'Marcador', conservé pour traçabilité
        reacted (bool)
        reactionTime (float) — secondes après apparition du stimulus
        correct (bool)
        duplicated (bool)
        flagType (int)       — 0=circle,1=square,2=rhombus,3=doubleCircle,-1=inconnu
        generalTime (float)  — secondes depuis le début du JEU (pas de la session EEG)
        focus (str)          — 'Target' | 'non_focusable'
    """
    df = pd.read_csv(tags_path)

    # Parsing du champ JSON 'value'
    records = []
    for _, row in df.iterrows():
        try:
            v = ast.literal_eval(row['value'])
            r = v['reactionOrOmission'][0]
        except (KeyError, IndexError, ValueError, SyntaxError) as e:
            logger.warning("Ligne TAGS non parseable (ignorée) : %s", e)
            continue

        records.append({
            'timestamp_ms':  float(row['timestamp']),
            'label':         row['label'],
            'reacted':       r['reacted'] == 'True',
            'reactionTime':  float(r.get('reactionTime', np.nan)),
            'correct':       r['correct'] == 'True',
            'duplicated':    r['duplicated'] == 'True',
            'flagType':      int(r['flagType'][0]),   # liste d'un élément
            'generalTime':   float(r['generalTime']),
            'focus':         r['focus'],
        })

    parsed = pd.DataFrame(records)

    # flagType == -1 : valeur non documentée dans le README, coïncide avec
    # correct=False → code d'erreur de commission probable. Conservé mais signalé.
    n_unknown = (parsed['flagType'] == -1).sum()
    if n_unknown > 0:
        logger.info("flagType=-1 (non documenté) : %d occurrences dans %s",
                    n_unknown, tags_path)

    return parsed


def load_eeg_cgx(eeg_path: str) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Charge un fichier EEG_CGX.csv.

    Retourne :
        times   (np.ndarray [n_samples])   — timestamps relatifs en secondes
        data    (np.ndarray [n_samples, n_channels]) — µV
        channels (list[str])               — noms des canaux EEG uniquement
    """
    df = pd.read_csv(eeg_path)

    # La colonne de temps est toujours la première
    time_col = df.columns[0]
    times = df[time_col].values.astype(np.float64)

    # Colonnes EEG : on exclut les colonnes non-EEG connues
    NON_EEG_PREFIXES = ('Accel', 'Packet', 'TRIGGER', 'ExG', 'A2')
    eeg_cols = [c for c in df.columns[1:]
                if not any(c.startswith(p) for p in NON_EEG_PREFIXES)
                and c != time_col]

    data = df[eeg_cols].values.astype(np.float32)

    logger.info("EEG CGX chargé : %d échantillons, %d canaux, durée %.1f s",
                len(times), len(eeg_cols), times[-1] - times[0])

    return times, data, eeg_cols


# ---------------------------------------------------------------------------
# 2. Calcul de l'offset session
# ---------------------------------------------------------------------------

def compute_session_offset(
    tags_df: pd.DataFrame,
    eeg_times: np.ndarray,
    sfreq: float = CGX_SFREQ
) -> Tuple[float, float]:
    """
    Calcule l'offset (ms) à ajouter aux timestamps EEG relatifs pour les
    convertir en temps Unix absolu.

    Stratégie :
        offset_ms = tags_timestamp_ms - (eeg_relative_time_s * 1000)
    calculé sur TOUS les événements TAGS (pas seulement le premier), en cherchant
    pour chaque événement l'échantillon EEG le plus proche par generalTime.

    L'offset median est retourné (robuste aux outliers), ainsi que son écart-type
    (doit être < 20 ms pour une session valide — std observée ~3-4 ms).

    Paramètres
    ----------
    tags_df   : sortie de load_tags()
    eeg_times : tableau des timestamps relatifs CGX (secondes)
    sfreq     : fréquence d'échantillonnage CGX (default 500 Hz)

    Retourne
    --------
    offset_ms  : float — offset médian à appliquer
    offset_std : float — écart-type en ms (indicateur de qualité)
    """
    offsets = []

    for _, row in tags_df.iterrows():
        # generalTime = secondes depuis le début du JEU, qui correspond
        # approximativement au temps relatif EEG (même référence de début)
        game_time_s = row['generalTime']
        unix_time_ms = row['timestamp_ms']

        # Échantillon EEG le plus proche
        idx = np.searchsorted(eeg_times, game_time_s)
        idx = np.clip(idx, 0, len(eeg_times) - 1)
        eeg_time_ms = eeg_times[idx] * 1000.0

        offsets.append(unix_time_ms - eeg_time_ms)

    offsets = np.array(offsets)
    offset_ms = float(np.median(offsets))
    offset_std = float(np.std(offsets))

    if offset_std > 20.0:
        logger.warning(
            "Std de l'offset élevée (%.1f ms) — vérifier l'alignement TAGS/EEG.",
            offset_std
        )
    else:
        logger.info(
            "Offset session : %.2f ms (std=%.2f ms, n=%d événements)",
            offset_ms, offset_std, len(offsets)
        )

    return offset_ms, offset_std


def eeg_idx_to_unix_ms(
    eeg_times: np.ndarray,
    offset_ms: float
) -> np.ndarray:
    """
    Convertit les timestamps EEG relatifs (secondes) en temps Unix absolu (ms).
    """
    return eeg_times * 1000.0 + offset_ms


def unix_ms_to_eeg_idx(
    unix_timestamps_ms: np.ndarray,
    eeg_times: np.ndarray,
    offset_ms: float
) -> np.ndarray:
    """
    Convertit des timestamps Unix (ms) en indices d'échantillons EEG.

    Retourne
    --------
    indices (np.ndarray[int]) — indices dans le tableau EEG, clippés aux bornes.
    """
    eeg_relative_ms = unix_timestamps_ms - offset_ms
    eeg_relative_s = eeg_relative_ms / 1000.0
    indices = np.searchsorted(eeg_times, eeg_relative_s)
    return np.clip(indices, 0, len(eeg_times) - 1).astype(int)


# ---------------------------------------------------------------------------
# 3. Validation cross-check niveau Slackline
# ---------------------------------------------------------------------------

def validate_level_assignment(
    tags_df: pd.DataFrame,
    eeg_times: np.ndarray,
    flags_info: dict,
    candidate_levels: list = ['Level1', 'Level6', 'Level11']
) -> Dict[str, dict]:
    """
    Teste chaque niveau candidat et retourne les statistiques d'alignement.
    Le bon niveau donne : offset_std < 20 ms, écart_moyen ~0.94 s, stable.
    Les mauvais niveaux : écart 2-5x plus grand, dispersion élevée.

    Paramètres
    ----------
    tags_df       : événements TAGS de la session (generalTime en secondes)
    eeg_times     : timestamps relatifs CGX (secondes)
    flags_info    : contenu de slackline_flags_info.json
    candidate_levels : niveaux à tester

    Retourne
    --------
    dict { level_name : { 'offset_ms', 'offset_std', 'mean_abs_residual_s' } }
    """
    levels_map = {
        item['level']: [f['flag_spawn_time'] for f in item['flags']]
        for item in flags_info['slackline_levels_flags_info']
    }

    results = {}
    reacted_events = tags_df[tags_df['reacted']].reset_index(drop=True)

    for level_name in candidate_levels:
        if level_name not in levels_map:
            continue
        spawn_times = levels_map[level_name]

        # Pour chaque événement TAGS, trouver le spawn_time le plus proche
        residuals = []
        for _, row in reacted_events.iterrows():
            gt = row['generalTime']
            closest = min(spawn_times, key=lambda t: abs(t - gt))
            residuals.append(abs(gt - closest))

        mean_res = float(np.mean(residuals))
        std_res = float(np.std(residuals))
        offset_ms, offset_std = compute_session_offset(
            reacted_events, eeg_times
        )

        results[level_name] = {
            'offset_ms': offset_ms,
            'offset_std': offset_std,
            'mean_abs_residual_s': mean_res,
            'std_residual_s': std_res,
        }
        logger.info(
            "%s : résidu moyen=%.3f s (std=%.3f s), offset=%.1f ms (std=%.1f ms)",
            level_name, mean_res, std_res, offset_ms, offset_std
        )

    # Le niveau gagnant = résidu moyen minimal
    best = min(results, key=lambda k: results[k]['mean_abs_residual_s'])
    logger.info("Niveau assigné automatiquement : %s", best)
    results['_best'] = best

    return results


# ---------------------------------------------------------------------------
# 4. Fonction utilitaire principale
# ---------------------------------------------------------------------------

def sync_session(
    eeg_path: str,
    tags_path: str,
    flags_info_path: str,
    validate: bool = True
) -> Dict:
    """
    Point d'entrée principal : charge EEG + TAGS, calcule l'offset,
    valide le niveau Slackline si demandé.

    Retourne un dictionnaire complet de session :
    {
        'eeg_times'   : np.ndarray (timestamps relatifs, s),
        'eeg_data'    : np.ndarray (n_samples x n_channels),
        'eeg_channels': list[str],
        'tags_df'     : pd.DataFrame,
        'offset_ms'   : float,
        'offset_std'  : float,
        'level'       : str | None,
        'valid'       : bool
    }
    """
    eeg_times, eeg_data, channels = load_eeg_cgx(eeg_path)
    tags_df = load_tags(tags_path)
    offset_ms, offset_std = compute_session_offset(tags_df, eeg_times)

    level = None
    if validate:
        with open(flags_info_path) as f:
            flags_info = json.load(f)
        val_results = validate_level_assignment(tags_df, eeg_times, flags_info)
        level = val_results.get('_best')

    is_valid = offset_std < 20.0

    return {
        'eeg_times':    eeg_times,
        'eeg_data':     eeg_data,
        'eeg_channels': channels,
        'tags_df':      tags_df,
        'offset_ms':    offset_ms,
        'offset_std':   offset_std,
        'level':        level,
        'valid':        is_valid,
    }
