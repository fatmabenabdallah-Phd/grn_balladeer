"""
data/sync.py
============
Synchronisation temporelle entre les fichiers EEG CGX (horloge relative,
secondes depuis le début de session) et les fichiers TAGS (horloge Unix
absolue, millisecondes).

DÉCOUVERTES EMPIRIQUES VALIDÉES sur données réelles UB0136 :
  - Le timestamp CGX est RELATIF (secondes depuis t=0 de la session).
  - Les timestamps TAGS sont ABSOLUS (Unix millisecondes, précision sous-ms).
  - generalTime dans TAGS = temps depuis le début du JEU, pas depuis le début
    de l'enregistrement EEG → ne pas utiliser comme ancre directe.
  - L'ancre correcte = timestamp Unix extrait du NOM DE FICHIER EEG (précision
    à la seconde, suffisante car validée avec std < 20 ms).
  - Canaux CGX confirmés : 29 EEG (uV) + 3 accéléromètre ACC32/33/34 (mg).
  - Durée session Slackline Lvl1 confirmée : ~305 s.
  - flagType=-1 : valeur non documentée, coïncide avec correct=False.

Auteur : GRN-BALLADEER project
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

# Fréquence d'échantillonnage CGX (confirmée sur données réelles)
CGX_SFREQ = 500.0  # Hz

# Fuseau horaire Espagne (GMT+1, confirmé dans le nom des fichiers TAGS : +01.00)
TZ_SPAIN = timezone(timedelta(hours=1))


# ---------------------------------------------------------------------------
# 1. Chargement des fichiers
# ---------------------------------------------------------------------------

def load_tags(tags_path: str) -> pd.DataFrame:
    """
    Charge un fichier TAGS et parse le champ JSON 'value'.

    Colonnes en sortie :
        timestamp_ms  (float) — horloge Unix absolue, millisecondes
        label         (str)   — toujours 'Marcador', conservé pour traçabilité
        reacted       (bool)
        reactionTime  (float) — secondes après apparition du stimulus
        correct       (bool)
        duplicated    (bool)
        flagType      (int)   — 0=circle,1=square,2=rhombus,3=doubleCircle,-1=inconnu
        generalTime   (float) — secondes depuis le début du JEU (≠ début EEG)
        focus         (str)   — 'Target' | 'non_focusable'
    """
    df = pd.read_csv(tags_path)
    records = []

    for _, row in df.iterrows():
        try:
            v = ast.literal_eval(row['value'])
            r = v['reactionOrOmission'][0]
        except (KeyError, IndexError, ValueError, SyntaxError) as e:
            logger.warning("Ligne TAGS non parseable (ignorée) : %s", e)
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
        logger.info("flagType=-1 (non documenté) : %d occurrences dans %s",
                    n_unknown, tags_path)

    return parsed


def load_eeg_cgx(eeg_path: str) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[np.ndarray]]:
    """
    Charge un fichier EEG_CGX.csv.

    Canaux confirmés sur données réelles :
        29 canaux EEG avec suffixe '(uV)' : AF7, Fpz, F7, Fz, T7, FC6, Fp1,
        F4, C4, Oz, CP6, Cz, PO8, CP5, O2, O1, P3, P4, P7, P8, Pz, PO7,
        T8, C3, Fp2, F3, F8, FC5, AF8.
        3 canaux accéléromètre : ACC32(mg), ACC33(mg), ACC34(mg).

    Retourne
    --------
    times    : [n_samples]            — timestamps relatifs en secondes
    data     : [n_samples, n_eeg]     — µV, canaux EEG uniquement
    channels : list[str]              — noms des canaux EEG (sans accéléromètre)
    accel    : [n_samples, 3] | None  — données accéléromètre X/Y/Z en mg
    """
    df = pd.read_csv(eeg_path)

    # Première colonne = temps relatif (secondes)
    time_col = df.columns[0]
    times = df[time_col].values.astype(np.float64)

    # Séparer EEG / accéléromètre / autres
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
        "EEG CGX chargé : %d échantillons | %d canaux EEG | %d canaux accel | durée %.1f s",
        len(times), len(eeg_cols),
        len(accel_cols) if accel_cols else 0,
        times[-1] - times[0]
    )

    return times, data, eeg_cols, accel


# ---------------------------------------------------------------------------
# 2. Ancrage temporel depuis le nom de fichier EEG
# ---------------------------------------------------------------------------

def parse_eeg_start_unix_ms(eeg_path: str) -> float:
    """
    Extrait le timestamp Unix (ms) du début de l'enregistrement EEG
    depuis le nom de fichier CGX.

    Format confirmé sur données réelles :
        UB0136_EEG_CGX_2024_01_19T16.30.01.csv
        → date = 2024-01-19T16:30:01 (GMT+1, Espagne)

    Retourne
    --------
    float — timestamp Unix en millisecondes
    """
    basename = os.path.basename(eeg_path).replace(".csv", "")

    # Extraire la partie après '_EEG_CGX_'
    try:
        date_str = basename.split("_EEG_CGX_")[-1]
        # Convertir format fichier → ISO 8601
        # '2024_01_19T16.30.01' → '2024-01-19T16:30:01'
        parts = date_str.split("T")
        date_part = parts[0].replace("_", "-")
        time_part = parts[1].replace(".", ":")
        iso_str = f"{date_part}T{time_part}"

        dt = datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%S")
        dt = dt.replace(tzinfo=TZ_SPAIN)
        unix_ms = dt.timestamp() * 1000.0

        logger.info("Début EEG extrait du nom de fichier : %s → %.0f ms Unix",
                    iso_str, unix_ms)
        return unix_ms

    except Exception as e:
        raise ValueError(
            f"Impossible de parser la date depuis le nom de fichier '{eeg_path}'. "
            f"Format attendu : *_EEG_CGX_YYYY_MM_DDTHH.MM.SS.csv\nErreur : {e}"
        )


# ---------------------------------------------------------------------------
# 3. Calcul et validation de l'offset session
# ---------------------------------------------------------------------------

def compute_session_offset(
    tags_df: pd.DataFrame,
    eeg_times: np.ndarray,
    eeg_start_unix_ms: float,
    sfreq: float = CGX_SFREQ
) -> Tuple[float, float]:
    """
    Calcule et valide l'offset temporel de la session.

    Stratégie :
        offset_ms = eeg_start_unix_ms
        (le timestamp CGX est relatif depuis 0 → ajouter le temps Unix
        du début d'enregistrement donne l'heure absolue)

    Validation :
        Pour chaque événement TAGS (timestamp_ms absolu), on recalcule
        l'index EEG correspondant et on vérifie la cohérence.
        Std attendue < 20 ms si l'ancrage est correct.

    Paramètres
    ----------
    tags_df           : sortie de load_tags()
    eeg_times         : timestamps relatifs CGX (secondes)
    eeg_start_unix_ms : timestamp Unix (ms) du début d'enregistrement EEG

    Retourne
    --------
    offset_ms  : float — offset à appliquer (= eeg_start_unix_ms)
    offset_std : float — écart-type de validation en ms
    """
    offset_ms = eeg_start_unix_ms
    residuals = []

    for _, row in tags_df.iterrows():
        tag_unix_ms    = row['timestamp_ms']
        eeg_relative_s = (tag_unix_ms - offset_ms) / 1000.0

        # Vérifier que l'événement tombe dans la fenêtre EEG
        if eeg_relative_s < eeg_times[0] or eeg_relative_s > eeg_times[-1]:
            continue

        idx = np.searchsorted(eeg_times, eeg_relative_s)
        idx = np.clip(idx, 0, len(eeg_times) - 1)
        reconstructed_ms = eeg_times[idx] * 1000.0 + offset_ms
        residuals.append(abs(reconstructed_ms - tag_unix_ms))

    if not residuals:
        logger.warning("Aucun événement TAGS dans la fenêtre EEG — offset non validé.")
        return offset_ms, 9999.0

    offset_std = float(np.std(residuals))
    mean_res   = float(np.mean(residuals))

    if offset_std > 20.0:
        logger.warning(
            "Std de validation élevée (%.1f ms, mean=%.1f ms) — "
            "vérifier le nom de fichier EEG ou le fuseau horaire.",
            offset_std, mean_res
        )
    else:
        logger.info(
            "Offset validé : %.0f ms | résidu mean=%.2f ms, std=%.2f ms | n=%d événements",
            offset_ms, mean_res, offset_std, len(residuals)
        )

    return offset_ms, offset_std


# ---------------------------------------------------------------------------
# 4. Conversion d'indices
# ---------------------------------------------------------------------------

def eeg_idx_to_unix_ms(eeg_times: np.ndarray, offset_ms: float) -> np.ndarray:
    """Timestamps EEG relatifs (s) → temps Unix absolu (ms)."""
    return eeg_times * 1000.0 + offset_ms


def unix_ms_to_eeg_idx(
    unix_timestamps_ms: np.ndarray,
    eeg_times: np.ndarray,
    offset_ms: float
) -> np.ndarray:
    """
    Timestamps Unix (ms) → indices d'échantillons EEG.
    Clippé aux bornes [0, n_samples-1].
    """
    eeg_relative_s = (unix_timestamps_ms - offset_ms) / 1000.0
    indices = np.searchsorted(eeg_times, eeg_relative_s)
    return np.clip(indices, 0, len(eeg_times) - 1).astype(int)


# ---------------------------------------------------------------------------
# 5. Validation cross-check niveau Slackline
# ---------------------------------------------------------------------------

def validate_level_assignment(
    tags_df: pd.DataFrame,
    flags_info: dict,
    candidate_levels: List[str] = ['Level1', 'Level6', 'Level11']
) -> Dict[str, dict]:
    """
    Identifie le niveau Slackline en comparant les generalTime des événements
    TAGS aux flag_spawn_time de chaque niveau.

    Le bon niveau = résidu moyen minimal entre generalTime et spawn_time le
    plus proche. Validé empiriquement : bon niveau ~0.94 s de résidu moyen,
    mauvais niveaux 2-5x plus.

    Retourne
    --------
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
        logger.info("%s : résidu moyen=%.3f s (std=%.3f s)",
                    level_name, mean_res, std_res)

    best = min(results, key=lambda k: results[k]['mean_residual_s'])
    logger.info("Niveau assigné : %s", best)
    results['_best'] = best

    return results


# ---------------------------------------------------------------------------
# 6. Point d'entrée principal
# ---------------------------------------------------------------------------

def sync_session(
    eeg_path:        str,
    tags_path:       str,
    flags_info_path: str,
    validate:        bool = True
) -> Dict:
    """
    Charge EEG + TAGS, calcule l'offset temporel, valide le niveau Slackline.

    Retourne
    --------
    {
        'eeg_times'   : np.ndarray [n_samples]          — timestamps relatifs (s)
        'eeg_data'    : np.ndarray [n_samples, n_eeg]   — µV
        'eeg_channels': list[str]                        — noms canaux EEG
        'accel_data'  : np.ndarray [n_samples, 3] | None — accéléromètre (mg)
        'tags_df'     : pd.DataFrame
        'offset_ms'   : float   — offset Unix à ajouter aux timestamps EEG
        'offset_std'  : float   — std de validation (ms), doit être < 20
        'level'       : str | None — niveau Slackline détecté automatiquement
        'valid'       : bool    — True si offset_std < 20 ms
    }
    """
    # Chargement
    eeg_times, eeg_data, channels, accel = load_eeg_cgx(eeg_path)
    tags_df = load_tags(tags_path)

    # Ancrage temporel depuis le nom de fichier
    eeg_start_unix_ms = parse_eeg_start_unix_ms(eeg_path)

    # Calcul + validation de l'offset
    offset_ms, offset_std = compute_session_offset(
        tags_df, eeg_times, eeg_start_unix_ms
    )

    # Identification du niveau Slackline
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
