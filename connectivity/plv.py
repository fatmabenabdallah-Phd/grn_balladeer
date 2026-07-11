"""
connectivity/plv.py
===================
Calcul de la connectivité de phase entre électrodes EEG.

Produit les matrices W_ij = a_ij * exp(i * theta_ij) qui constituent
les arêtes du Laplacien magnétique dans le GRN.

  a_ij     = PLV (Phase Locking Value)  — amplitude de la connexion
  theta_ij = différence de phase moyenne — déphasage inter-électrode

Bandes de fréquence utilisées (standard clinique TDAH) :
  delta : 1–4 Hz
  theta : 4–8 Hz   ← biomarqueur TDAH principal (ratio theta/beta élevé)
  alpha : 8–13 Hz
  beta  : 13–30 Hz ← biomarqueur TDAH principal
  gamma : 30–45 Hz

Référence : Stam et al. (2007) pour le Phase Lag Index (PLI, ablation).

Auteur : GRN-BALLADEER project
"""

import numpy as np
from scipy.signal import hilbert, firwin, filtfilt
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Bandes de fréquence
# ---------------------------------------------------------------------------

BANDS: Dict[str, Tuple[float, float]] = {
    'delta': (1.0,  4.0),
    'theta': (4.0,  8.0),
    'alpha': (8.0, 13.0),
    'beta':  (13.0, 30.0),
    'gamma': (30.0, 45.0),
}


# ---------------------------------------------------------------------------
# 2. Filtrage par bande
# ---------------------------------------------------------------------------

def bandpass_band(
    signal: np.ndarray,
    l_freq: float,
    h_freq: float,
    sfreq:  float
) -> np.ndarray:
    """
    Filtre FIR passe-bande appliqué à un signal [n_samples] ou [n_samples, n_channels].
    """
    n_taps = int(sfreq) + 1
    n_taps = n_taps if n_taps % 2 == 1 else n_taps + 1
    coeffs = firwin(n_taps, [l_freq, h_freq], pass_zero=False, fs=sfreq)
    return filtfilt(coeffs, [1.0], signal, axis=0)


# ---------------------------------------------------------------------------
# 3. Phase instantanée (transformée de Hilbert)
# ---------------------------------------------------------------------------

def instantaneous_phase(band_signal: np.ndarray) -> np.ndarray:
    """
    Calcule la phase instantanée via la transformée de Hilbert.

    Paramètre
    ---------
    band_signal : [n_samples, n_channels]

    Retourne
    --------
    phase : [n_samples, n_channels] — valeurs en radians dans [-pi, pi]
    """
    analytic = hilbert(band_signal, axis=0)
    return np.angle(analytic)


# ---------------------------------------------------------------------------
# 4. Matrice PLV (arête amplitude a_ij)
# ---------------------------------------------------------------------------

def compute_plv_matrix(phases: np.ndarray) -> np.ndarray:
    """
    PLV_ij = |mean_t( exp(i * (phi_i(t) - phi_j(t))) )|

    Paramètre
    ---------
    phases : [n_samples, n_channels]

    Retourne
    --------
    plv : [n_channels, n_channels] — symétrique, valeurs dans [0, 1]
    """
    n_channels = phases.shape[1]
    # Représentation complexe de la phase
    z = np.exp(1j * phases)               # [n_samples, n_channels]
    # Produit extérieur → différences de phase
    # z_i * conj(z_j) = exp(i*(phi_i - phi_j))
    outer = z[:, :, np.newaxis] * np.conj(z[:, np.newaxis, :])   # [T, C, C]
    plv = np.abs(outer.mean(axis=0))      # moyenne temporelle + module
    # La diagonale vaut 1 par construction — la mettre à 0 (pas d'auto-boucle)
    np.fill_diagonal(plv, 0.0)
    return plv.astype(np.float32)


# ---------------------------------------------------------------------------
# 5. Matrice de différence de phase moyenne (arête phase theta_ij)
# ---------------------------------------------------------------------------

def compute_mean_phase_diff(phases: np.ndarray) -> np.ndarray:
    """
    theta_ij = angle( mean_t( exp(i*(phi_i - phi_j)) ) )

    C'est la phase du vecteur moyen dans le plan complexe.
    Anti-symétrique : theta_ji = -theta_ij.

    Paramètre
    ---------
    phases : [n_samples, n_channels]

    Retourne
    --------
    phase_diff : [n_channels, n_channels] — radians dans [-pi, pi]
    """
    z = np.exp(1j * phases)
    outer = z[:, :, np.newaxis] * np.conj(z[:, np.newaxis, :])
    mean_complex = outer.mean(axis=0)
    phase_diff = np.angle(mean_complex)
    np.fill_diagonal(phase_diff, 0.0)
    return phase_diff.astype(np.float32)


# ---------------------------------------------------------------------------
# 6. Matrice PLI (ablation — pas le modèle principal)
# ---------------------------------------------------------------------------

def compute_pli_matrix(phases: np.ndarray) -> np.ndarray:
    """
    PLI_ij = |mean_t( sign(sin(phi_i - phi_j)) )|

    Insensible aux connexions de volume conducteur (bias de phase nulle).
    Utilisé UNIQUEMENT comme baseline d'ablation (PLV vs PLI en Phase 4).

    Référence : Stam et al. (2007).
    """
    n_channels = phases.shape[1]
    pli = np.zeros((n_channels, n_channels), dtype=np.float32)
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            diff = phases[:, i] - phases[:, j]
            val = abs(np.mean(np.sign(np.sin(diff))))
            pli[i, j] = val
            pli[j, i] = val
    return pli


# ---------------------------------------------------------------------------
# 7. Construction des arêtes complexes W_ij (Pilier II du GRN)
# ---------------------------------------------------------------------------

def build_complex_edge_weights(
    plv: np.ndarray,
    phase_diff: np.ndarray,
    threshold: float = 0.1
) -> np.ndarray:
    """
    W_ij = a_ij * exp(i * theta_ij)   avec a_ij = PLV_ij

    Vérifie l'hermiticité : W_ji = conj(W_ij) (car PLV symétrique
    et phase_diff antisymétrique).

    Paramètres
    ----------
    plv        : [n_channels, n_channels] — amplitude (symétrique)
    phase_diff : [n_channels, n_channels] — phase (antisymétrique)
    threshold  : seuil minimal de PLV — les arêtes sous ce seuil
                 sont mises à zéro (graphe épars, réduit le bruit)

    Retourne
    --------
    W : [n_channels, n_channels] complexe (complex64)
    """
    # Seuillage
    a = plv.copy()
    a[a < threshold] = 0.0

    W = a * np.exp(1j * phase_diff)

    # Vérification hermiticité (debug)
    hermitian_error = np.max(np.abs(W - np.conj(W.T)))
    if hermitian_error > 1e-5:
        logger.warning(
            "W non hermitienne : erreur max = %.2e (attendu < 1e-5)", hermitian_error
        )

    return W.astype(np.complex64)


# ---------------------------------------------------------------------------
# 8. Laplacien magnétique L_C = D - W
# ---------------------------------------------------------------------------

def build_magnetic_laplacian(W: np.ndarray) -> np.ndarray:
    """
    L_C = D - W

    D_ii = sum_j a_ij  (degré basé sur les amplitudes PLV, réel)
    L_C est hermitienne → énergie E(h) = h^H L_C h ≥ 0 garantie.

    Paramètre
    ---------
    W : [n_channels, n_channels] complexe

    Retourne
    --------
    L_C : [n_channels, n_channels] complexe (complex64)
    """
    # Degrés = somme des amplitudes (partie réelle positive)
    a = np.abs(W)
    degrees = a.sum(axis=1)
    D = np.diag(degrees).astype(np.complex64)

    L_C = D - W

    # Vérification positivité semi-définie (valeurs propres ≥ 0)
    # Coûteux sur grand graphe — activé seulement en mode debug
    if logger.isEnabledFor(logging.DEBUG):
        eigenvalues = np.linalg.eigvalsh(L_C)
        min_eig = eigenvalues.min().real
        if min_eig < -1e-6:
            logger.debug("L_C non PSD : valeur propre min = %.4e", min_eig)

    return L_C


# ---------------------------------------------------------------------------
# 9. Pipeline complet : epoch → graphe multi-bandes
# ---------------------------------------------------------------------------

def epoch_to_graph(
    epoch_data: np.ndarray,
    sfreq: float,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    plv_threshold: float = 0.1,
    use_pli: bool = False
) -> Dict[str, Dict]:
    """
    Transforme un epoch EEG en dictionnaire de graphes, un par bande.

    Paramètre
    ---------
    epoch_data : [n_channels, n_samples] — sortie de build_subject_epoch_array
    sfreq      : fréquence d'échantillonnage (500 Hz pour CGX)
    bands      : dict de bandes à calculer (défaut = BANDS global)
    use_pli    : si True, utilise PLI au lieu de PLV (ablation Phase 4)

    Retourne
    --------
    graphs : {
        'theta': {
            'W'      : np.ndarray complexe [C, C],
            'L_C'    : np.ndarray complexe [C, C],
            'plv'    : np.ndarray float [C, C],
            'phase_diff': np.ndarray float [C, C],
        },
        'alpha': { ... },
        ...
    }
    """
    if bands is None:
        bands = BANDS

    # Transposer pour scipy : [n_samples, n_channels]
    signal = epoch_data.T

    graphs = {}
    for band_name, (l_freq, h_freq) in bands.items():
        # Filtrage
        filtered = bandpass_band(signal, l_freq, h_freq, sfreq)

        # Phase instantanée
        phases = instantaneous_phase(filtered)

        # Connectivité
        if use_pli:
            amplitude = compute_pli_matrix(phases)
        else:
            amplitude = compute_plv_matrix(phases)

        phase_diff = compute_mean_phase_diff(phases)

        # Arêtes complexes + Laplacien
        W   = build_complex_edge_weights(amplitude, phase_diff, plv_threshold)
        L_C = build_magnetic_laplacian(W)

        graphs[band_name] = {
            'W':          W,
            'L_C':        L_C,
            'plv':        amplitude,
            'phase_diff': phase_diff,
        }

    return graphs


def batch_epochs_to_graphs(
    X: np.ndarray,
    sfreq: float,
    bands: Optional[Dict] = None,
    plv_threshold: float = 0.1,
    use_pli: bool = False
) -> List[Dict]:
    """
    Applique epoch_to_graph sur un batch entier.

    Paramètre
    ---------
    X : [n_epochs, n_channels, n_samples]

    Retourne
    --------
    Liste de n_epochs graphes multi-bandes.
    """
    results = []
    for i in range(X.shape[0]):
        g = epoch_to_graph(
            X[i], sfreq, bands=bands,
            plv_threshold=plv_threshold,
            use_pli=use_pli
        )
        results.append(g)

    logger.info("batch_epochs_to_graphs : %d epochs traités.", len(results))
    return results
