"""
data/epoching.py
================
Découpe les enregistrements EEG CGX en epochs verrouillés sur les événements
Slackline, après synchronisation via sync.py.

Conventions :
  - Un epoch = fenêtre [tmin, tmax] secondes autour de l'apparition d'un flag.
  - Seuls les événements où reacted=True ET correct=True sont inclus par défaut.
  - flagType=-1 est exclu (code non documenté, voir sync.py).
  - La normalisation est intra-sujet/intra-session (z-score par canal sur la session)
    car il n'y a PAS de resting-state dans BALLADEER.

Auteur : GRN-BALLADEER project
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field

from sync import unix_ms_to_eeg_idx, CGX_SFREQ

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Structure de données : un epoch annoté
# ---------------------------------------------------------------------------

@dataclass
class Epoch:
    """Un segment EEG avec ses métadonnées."""
    data:        np.ndarray          # shape [n_samples_epoch, n_channels]
    label:       int                 # 0 = contrôle, 1 = TDAH
    subject_id:  str
    session_id:  str
    flag_type:   int                 # 0/1/2/3
    flag_time_s: float               # generalTime (s depuis début jeu)
    reacted:     bool
    correct:     bool
    focus:       str                 # 'Target' | 'non_focusable'
    reaction_time_s: float


# ---------------------------------------------------------------------------
# 2. Prétraitement par canal
# ---------------------------------------------------------------------------

def bandpass_filter_np(
    data: np.ndarray,
    sfreq: float,
    l_freq: float,
    h_freq: float
) -> np.ndarray:
    """
    Filtre passe-bande FIR via numpy/scipy. Utilisé avant calcul de PLV.
    Pour un prétraitement MNE complet, préférez load_eeg_cgx → mne.Raw.
    """
    from scipy.signal import firwin, filtfilt
    n_taps = int(sfreq) + 1  # durée filtre = 1 seconde
    n_taps = n_taps if n_taps % 2 == 1 else n_taps + 1
    coeffs = firwin(n_taps, [l_freq, h_freq],
                    pass_zero=False, fs=sfreq)
    return filtfilt(coeffs, [1.0], data, axis=0)


def notch_filter_np(
    data: np.ndarray,
    sfreq: float,
    freq: float = 50.0
) -> np.ndarray:
    """Coupe-bande à freq Hz (bruit secteur, défaut 50 Hz pour EU/Afrique/Espagne)."""
    from scipy.signal import iirnotch, filtfilt
    b, a = iirnotch(freq, Q=30.0, fs=sfreq)
    return filtfilt(b, a, data, axis=0)


def zscore_normalize_session(data: np.ndarray) -> np.ndarray:
    """
    Normalisation z-score PAR CANAL sur l'ensemble de la session.
    Obligatoire car pas de resting-state dans BALLADEER pour normaliser
    inter-session. Appliqué AVANT le découpage en epochs.
    """
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    std[std == 0] = 1.0  # éviter division par zéro sur canaux plats
    return (data - mean) / std


def detect_motion_artifacts(
    eeg_data: np.ndarray,
    accel_data: Optional[np.ndarray],
    epoch_indices: np.ndarray,
    n_samples_epoch: int,
    accel_threshold_mg: float = 200.0
) -> np.ndarray:
    """
    Masque booléen : True = epoch contaminé par du mouvement.
    Utilise les colonnes accéléromètre CGX (disponibles dans le fichier).
    Sert au test de robustesse réel en Phase 4 (stratification haute/basse
    amplitude de mouvement).

    Paramètres
    ----------
    accel_data         : [n_samples, 3] — axes X,Y,Z en mg. None = désactivé.
    epoch_indices      : indices de début des epochs dans le signal EEG.
    accel_threshold_mg : seuil d'amplitude max acceptable.
    """
    if accel_data is None:
        return np.zeros(len(epoch_indices), dtype=bool)

    accel_norm = np.linalg.norm(accel_data, axis=1)  # magnitude globale
    masks = []
    for idx in epoch_indices:
        end = min(idx + n_samples_epoch, len(accel_norm))
        segment = accel_norm[idx:end]
        masks.append(segment.max() > accel_threshold_mg)
    return np.array(masks, dtype=bool)


# ---------------------------------------------------------------------------
# 3. Découpage principal
# ---------------------------------------------------------------------------

def cut_epochs(
    eeg_times:     np.ndarray,
    eeg_data:      np.ndarray,
    tags_df:       pd.DataFrame,
    subject_id:    str,
    session_id:    str,
    label:         int,
    offset_ms:     float,
    tmin:          float = -0.5,
    tmax:          float = 2.0,
    sfreq:         float = CGX_SFREQ,
    include_only_correct: bool = True,
    exclude_flag_types:   Optional[List[int]] = None,
    accel_data:    Optional[np.ndarray] = None
) -> Tuple[List[Epoch], dict]:
    """
    Découpe le signal EEG en epochs verrouillés sur les événements TAGS.

    Paramètres
    ----------
    tmin / tmax          : fenêtre en secondes autour du flag (défaut : -0.5s à +2.0s).
    include_only_correct : si True, exclut les essais incorrect ET les non-réponses.
    exclude_flag_types   : liste de flagTypes à exclure (ex. [-1] pour le code inconnu).
    accel_data           : données accéléromètre [n_samples, 3] pour détecter artefacts.

    Retourne
    --------
    epochs : liste d'objets Epoch
    stats  : dict de statistiques de découpage (pour les logs/rapport)
    """
    if exclude_flag_types is None:
        exclude_flag_types = [-1]

    n_before = int(abs(tmin) * sfreq)
    n_after  = int(tmax * sfreq)
    n_epoch  = n_before + n_after

    epochs = []
    stats = {
        'total_events':    len(tags_df),
        'excluded_flag':   0,
        'excluded_correct': 0,
        'excluded_boundary': 0,
        'excluded_motion': 0,
        'kept':            0,
    }

    # Filtrer les événements
    working = tags_df.copy()

    # Exclure flagTypes non souhaités
    mask_flag = working['flagType'].isin(exclude_flag_types)
    stats['excluded_flag'] = int(mask_flag.sum())
    working = working[~mask_flag]

    # Exclure les essais incorrects si demandé
    if include_only_correct:
        mask_bad = ~(working['reacted'] & working['correct'])
        stats['excluded_correct'] = int(mask_bad.sum())
        working = working[~mask_bad]

    # Convertir les timestamps TAGS en indices EEG
    event_unix_ms = working['timestamp_ms'].values
    # On utilise generalTime (ancre jeu) plutôt que timestamp réaction
    # car generalTime marque l'APPARITION du flag, pas la réponse
    # → on recalcule l'index depuis le temps EEG correspondant à generalTime
    general_times_s = working['generalTime'].values
    # Convertir generalTime (relatif session) → timestamp Unix ms
    general_times_unix_ms = general_times_s * 1000.0 + offset_ms
    event_indices = unix_ms_to_eeg_idx(general_times_unix_ms, eeg_times, offset_ms)

    # Détecter les artefacts de mouvement
    motion_mask = detect_motion_artifacts(
        eeg_data, accel_data, event_indices, n_epoch
    )

    for k, (idx, row) in enumerate(zip(event_indices, working.itertuples())):
        start = idx - n_before
        end   = idx + n_after

        # Hors bornes du signal
        if start < 0 or end > len(eeg_data):
            stats['excluded_boundary'] += 1
            continue

        # Artefact de mouvement
        if motion_mask[k]:
            stats['excluded_motion'] += 1
            continue

        segment = eeg_data[start:end, :]  # [n_epoch, n_channels]

        epoch = Epoch(
            data=segment,
            label=label,
            subject_id=subject_id,
            session_id=session_id,
            flag_type=int(row.flagType),
            flag_time_s=float(row.generalTime),
            reacted=bool(row.reacted),
            correct=bool(row.correct),
            focus=str(row.focus),
            reaction_time_s=float(row.reactionTime)
                if not np.isnan(row.reactionTime) else np.nan,
        )
        epochs.append(epoch)
        stats['kept'] += 1

    logger.info(
        "Sujet %s session %s : %d/%d epochs gardés "
        "(excl. flag=%d, correct=%d, bornes=%d, mouvement=%d)",
        subject_id, session_id,
        stats['kept'], stats['total_events'],
        stats['excluded_flag'], stats['excluded_correct'],
        stats['excluded_boundary'], stats['excluded_motion'],
    )

    return epochs, stats


# ---------------------------------------------------------------------------
# 4. Prétraitement complet d'une session
# ---------------------------------------------------------------------------

def preprocess_and_epoch_session(
    session: dict,
    label: int,
    subject_id: str,
    tmin: float = -0.5,
    tmax: float = 2.0,
    apply_notch: bool = True,
    apply_normalize: bool = True
) -> Tuple[List[Epoch], dict]:
    """
    Enchaîne notch + normalisation z-score + découpage en epochs sur une
    session déjà synchronisée (sortie de sync.sync_session).

    Le bandpass par bande est fait en AVAL dans le module connectivity/,
    qui a besoin du signal large-bande pour calculer le PLV par bande.
    Ici on applique seulement un filtre large [1–45 Hz] pour éliminer
    les artéfacts basse fréquence et l'aliasing.
    """
    eeg_times = session['eeg_times']
    eeg_data  = session['eeg_data'].copy()   # [n_samples, n_channels]
    tags_df   = session['tags_df']
    offset_ms = session['offset_ms']

    # Filtre coupe-bande secteur (50 Hz, Espagne)
    if apply_notch:
        eeg_data = notch_filter_np(eeg_data, sfreq=CGX_SFREQ, freq=50.0)

    # Filtre large-bande [1–45 Hz]
    eeg_data = bandpass_filter_np(eeg_data, sfreq=CGX_SFREQ,
                                   l_freq=1.0, h_freq=45.0)

    # Normalisation intra-session
    if apply_normalize:
        eeg_data = zscore_normalize_session(eeg_data)

    # Extraction accéléromètre si disponible dans le fichier EEG original
    # (champs 'Accel X', 'Accel Y', 'Accel Z' — cf. README)
    accel_data = session.get('accel_data', None)

    epochs, stats = cut_epochs(
        eeg_times=eeg_times,
        eeg_data=eeg_data,
        tags_df=tags_df,
        subject_id=subject_id,
        session_id=str(session.get('level', 'unknown')),
        label=label,
        offset_ms=offset_ms,
        tmin=tmin,
        tmax=tmax,
        sfreq=CGX_SFREQ,
        include_only_correct=True,
        exclude_flag_types=[-1],
        accel_data=accel_data,
    )

    return epochs, stats


# ---------------------------------------------------------------------------
# 5. Construction du dataset sujet-complet
# ---------------------------------------------------------------------------

def build_subject_epoch_array(
    epochs: List[Epoch]
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """
    Convertit une liste d'Epoch en tableaux numpy prêts pour le GRN.

    Retourne
    --------
    X     : [n_epochs, n_channels, n_samples] — format standard PyTorch conv
    y     : [n_epochs] — labels (0/1)
    meta  : liste de dicts avec subject_id, flag_type, reaction_time_s, focus
    """
    X = np.stack([e.data.T for e in epochs], axis=0)  # transposer → [ch, time]
    y = np.array([e.label for e in epochs], dtype=np.int64)
    meta = [
        {
            'subject_id':     e.subject_id,
            'flag_type':      e.flag_type,
            'reaction_time_s': e.reaction_time_s,
            'focus':          e.focus,
        }
        for e in epochs
    ]
    return X, y, meta
