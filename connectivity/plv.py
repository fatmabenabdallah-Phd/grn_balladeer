"""
connectivity/plv.py
===================
NOTE: this module is a stale duplicate of phase_connectivity.py and is not
imported anywhere else in the package. Kept for reference only pending
removal — see README "Known issues".

Computes phase connectivity between EEG electrodes.

Produces the W_ij = a_ij * exp(i * theta_ij) matrices that form the
edges of the magnetic Laplacian in the GRN.

  a_ij     = PLV (Phase Locking Value)  — connection amplitude
  theta_ij = mean phase difference      — inter-electrode phase lag

Frequency bands used (standard ADHD clinical bands):
  delta : 1-4 Hz
  theta : 4-8 Hz   <- primary ADHD biomarker (elevated theta/beta ratio)
  alpha : 8-13 Hz
  beta  : 13-30 Hz <- primary ADHD biomarker
  gamma : 30-45 Hz

Reference: Stam et al. (2007) for the Phase Lag Index (PLI, ablation).

Author: GRN-BALLADEER project
"""

import numpy as np
from scipy.signal import hilbert, firwin, filtfilt
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Frequency bands
# ---------------------------------------------------------------------------

BANDS: Dict[str, Tuple[float, float]] = {
    'delta': (1.0,  4.0),
    'theta': (4.0,  8.0),
    'alpha': (8.0, 13.0),
    'beta':  (13.0, 30.0),
    'gamma': (30.0, 45.0),
}


# ---------------------------------------------------------------------------
# 2. Per-band filtering
# ---------------------------------------------------------------------------

def bandpass_band(
    signal: np.ndarray,
    l_freq: float,
    h_freq: float,
    sfreq:  float
) -> np.ndarray:
    """
    FIR bandpass filter applied to a [n_samples] or [n_samples, n_channels] signal.
    """
    n_taps = int(sfreq) + 1
    n_taps = n_taps if n_taps % 2 == 1 else n_taps + 1
    coeffs = firwin(n_taps, [l_freq, h_freq], pass_zero=False, fs=sfreq)
    return filtfilt(coeffs, [1.0], signal, axis=0)


# ---------------------------------------------------------------------------
# 3. Instantaneous phase (Hilbert transform)
# ---------------------------------------------------------------------------

def instantaneous_phase(band_signal: np.ndarray) -> np.ndarray:
    """
    Computes the instantaneous phase via the Hilbert transform.

    Parameter
    ---------
    band_signal : [n_samples, n_channels]

    Returns
    -------
    phase : [n_samples, n_channels] — values in radians in [-pi, pi]
    """
    analytic = hilbert(band_signal, axis=0)
    return np.angle(analytic)


# ---------------------------------------------------------------------------
# 4. PLV matrix (edge amplitude a_ij)
# ---------------------------------------------------------------------------

def compute_plv_matrix(phases: np.ndarray) -> np.ndarray:
    """
    PLV_ij = |mean_t( exp(i * (phi_i(t) - phi_j(t))) )|

    Parameter
    ---------
    phases : [n_samples, n_channels]

    Returns
    -------
    plv : [n_channels, n_channels] — symmetric, values in [0, 1]
    """
    n_channels = phases.shape[1]
    # Complex representation of the phase
    z = np.exp(1j * phases)               # [n_samples, n_channels]
    # Outer product -> phase differences
    # z_i * conj(z_j) = exp(i*(phi_i - phi_j))
    outer = z[:, :, np.newaxis] * np.conj(z[:, np.newaxis, :])   # [T, C, C]
    plv = np.abs(outer.mean(axis=0))      # temporal mean + magnitude
    # The diagonal is 1 by construction — set it to 0 (no self-loop)
    np.fill_diagonal(plv, 0.0)
    return plv.astype(np.float32)


# ---------------------------------------------------------------------------
# 5. Mean phase difference matrix (edge phase theta_ij)
# ---------------------------------------------------------------------------

def compute_mean_phase_diff(phases: np.ndarray) -> np.ndarray:
    """
    theta_ij = angle( mean_t( exp(i*(phi_i - phi_j)) ) )

    This is the phase of the mean vector in the complex plane.
    Anti-symmetric: theta_ji = -theta_ij.

    Parameter
    ---------
    phases : [n_samples, n_channels]

    Returns
    -------
    phase_diff : [n_channels, n_channels] — radians in [-pi, pi]
    """
    z = np.exp(1j * phases)
    outer = z[:, :, np.newaxis] * np.conj(z[:, np.newaxis, :])
    mean_complex = outer.mean(axis=0)
    phase_diff = np.angle(mean_complex)
    np.fill_diagonal(phase_diff, 0.0)
    return phase_diff.astype(np.float32)


# ---------------------------------------------------------------------------
# 6. PLI matrix (ablation — not the main model)
# ---------------------------------------------------------------------------

def compute_pli_matrix(phases: np.ndarray) -> np.ndarray:
    """
    PLI_ij = |mean_t( sign(sin(phi_i - phi_j)) )|

    Insensitive to volume-conduction connections (zero-phase bias).
    Used ONLY as an ablation baseline (PLV vs PLI in Phase 4).

    Reference: Stam et al. (2007).
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
# 7. Building the complex edges W_ij (GRN Pillar II)
# ---------------------------------------------------------------------------

def build_complex_edge_weights(
    plv: np.ndarray,
    phase_diff: np.ndarray,
    threshold: float = 0.1
) -> np.ndarray:
    """
    W_ij = a_ij * exp(i * theta_ij)   with a_ij = PLV_ij

    Checks Hermitian symmetry: W_ji = conj(W_ij) (since PLV is symmetric
    and phase_diff is anti-symmetric).

    Parameters
    ----------
    plv        : [n_channels, n_channels] — amplitude (symmetric)
    phase_diff : [n_channels, n_channels] — phase (anti-symmetric)
    threshold  : minimum PLV threshold — edges below this threshold
                 are zeroed out (sparse graph, reduces noise)

    Returns
    -------
    W : [n_channels, n_channels] complex (complex64)
    """
    # Thresholding
    a = plv.copy()
    a[a < threshold] = 0.0

    W = a * np.exp(1j * phase_diff)

    # Hermiticity check (debug)
    hermitian_error = np.max(np.abs(W - np.conj(W.T)))
    if hermitian_error > 1e-5:
        logger.warning(
            "W is not Hermitian: max error = %.2e (expected < 1e-5)", hermitian_error
        )

    return W.astype(np.complex64)


# ---------------------------------------------------------------------------
# 8. Magnetic Laplacian L_C = D - W
# ---------------------------------------------------------------------------

def build_magnetic_laplacian(W: np.ndarray) -> np.ndarray:
    """
    L_C = D - W

    D_ii = sum_j a_ij  (degree based on PLV amplitudes, real)
    L_C is Hermitian -> energy E(h) = h^H L_C h >= 0 is guaranteed.

    Parameter
    ---------
    W : [n_channels, n_channels] complex

    Returns
    -------
    L_C : [n_channels, n_channels] complex (complex64)
    """
    # Degrees = sum of amplitudes (positive real part)
    a = np.abs(W)
    degrees = a.sum(axis=1)
    D = np.diag(degrees).astype(np.complex64)

    L_C = D - W

    # Positive semi-definiteness check (eigenvalues >= 0)
    # Expensive on large graphs — only enabled in debug mode
    if logger.isEnabledFor(logging.DEBUG):
        eigenvalues = np.linalg.eigvalsh(L_C)
        min_eig = eigenvalues.min().real
        if min_eig < -1e-6:
            logger.debug("L_C not PSD: min eigenvalue = %.4e", min_eig)

    return L_C


# ---------------------------------------------------------------------------
# 9. Full pipeline: epoch -> multi-band graph
# ---------------------------------------------------------------------------

def epoch_to_graph(
    epoch_data: np.ndarray,
    sfreq: float,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    plv_threshold: float = 0.1,
    use_pli: bool = False
) -> Dict[str, Dict]:
    """
    Turns an EEG epoch into a dict of graphs, one per band.

    Parameter
    ---------
    epoch_data : [n_channels, n_samples] — output of build_subject_epoch_array
    sfreq      : sampling rate (500 Hz for CGX)
    bands      : dict of bands to compute (default = global BANDS)
    use_pli    : if True, uses PLI instead of PLV (Phase 4 ablation)

    Returns
    -------
    graphs : {
        'theta': {
            'W'      : complex np.ndarray [C, C],
            'L_C'    : complex np.ndarray [C, C],
            'plv'    : float np.ndarray [C, C],
            'phase_diff': float np.ndarray [C, C],
        },
        'alpha': { ... },
        ...
    }
    """
    if bands is None:
        bands = BANDS

    # Transpose for scipy: [n_samples, n_channels]
    signal = epoch_data.T

    graphs = {}
    for band_name, (l_freq, h_freq) in bands.items():
        # Filtering
        filtered = bandpass_band(signal, l_freq, h_freq, sfreq)

        # Instantaneous phase
        phases = instantaneous_phase(filtered)

        # Connectivity
        if use_pli:
            amplitude = compute_pli_matrix(phases)
        else:
            amplitude = compute_plv_matrix(phases)

        phase_diff = compute_mean_phase_diff(phases)

        # Complex edges + Laplacian
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
    Applies epoch_to_graph over a full batch.

    Parameter
    ---------
    X : [n_epochs, n_channels, n_samples]

    Returns
    -------
    List of n_epochs multi-band graphs.
    """
    results = []
    for i in range(X.shape[0]):
        g = epoch_to_graph(
            X[i], sfreq, bands=bands,
            plv_threshold=plv_threshold,
            use_pli=use_pli
        )
        results.append(g)

    logger.info("batch_epochs_to_graphs: %d epochs processed.", len(results))
    return results
