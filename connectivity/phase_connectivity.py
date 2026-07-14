"""
grn_balladeer.connectivity.phase_connectivity
=================================================
Module 3 — phase connectivity (PLV/PLI) and the complex magnetic
Laplacian, the GRN's graph input.

All functions operate on a SINGLE epoch, shape (n_channels, n_samples),
to keep the math explicit and testable in isolation. Batch over epochs
at the call site (Module 4+) once this is stable.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import butter, filtfilt, hilbert


def extract_band_signal(epoch_data: np.ndarray, band: Tuple[float, float], sfreq: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter, applied per channel.
    epoch_data: (n_channels, n_samples). Returns same shape."""
    low, high = band
    nyquist = sfreq / 2.0
    if high >= nyquist:
        raise ValueError(f"extract_band_signal: band upper bound {high} Hz >= Nyquist {nyquist} Hz")
    b, a = butter(order, [low / nyquist, high / nyquist], btype="band")
    return filtfilt(b, a, epoch_data, axis=-1)


def compute_instantaneous_phase(band_signal: np.ndarray) -> np.ndarray:
    """Instantaneous phase per channel via the analytic signal (Hilbert
    transform). band_signal: (n_channels, n_samples). Returns phase in
    radians, same shape."""
    analytic = hilbert(band_signal, axis=-1)
    return np.angle(analytic)


def _mean_phase_diff_complex(phases: np.ndarray) -> np.ndarray:
    """Internal helper: for every channel pair (i, j), the complex mean
    C_ij = mean_t(exp(i*(phi_i(t) - phi_j(t)))). |C_ij| is the PLV;
    angle(C_ij) is the circular mean phase difference. Computing both
    from the same complex mean (rather than separately) avoids angle-
    wrapping inconsistencies between magnitude and phase.
    phases: (n_channels, n_samples). Returns (n_channels, n_channels) complex."""
    n_channels = phases.shape[0]
    exp_phase = np.exp(1j * phases)  # (n_channels, n_samples)
    # C_ij = mean_t( exp_phase[i] * conj(exp_phase[j]) )
    C = (exp_phase @ exp_phase.conj().T) / phases.shape[1]
    return C


def compute_plv_matrix(phases: np.ndarray) -> np.ndarray:
    """a_ij = PLV_ij = |mean_t(exp(i(phi_i - phi_j)))|.
    phases: (n_channels, n_samples). Returns real (n_channels, n_channels),
    symmetric, diagonal = 1."""
    C = _mean_phase_diff_complex(phases)
    return np.abs(C)


def compute_mean_phase_diff(phases: np.ndarray) -> np.ndarray:
    """Circular mean phase difference per pair (radians), consistent
    with compute_plv_matrix's magnitude (same underlying complex mean —
    see _mean_phase_diff_complex). Antisymmetric: result[j,i] = -result[i,j]."""
    C = _mean_phase_diff_complex(phases)
    return np.angle(C)


def compute_pli_matrix(phases: np.ndarray) -> np.ndarray:
    """Phase Lag Index — alternative connectivity metric, reserved for
    the Module 11 connectivity ablation (PLV vs PLI). Less sensitive to
    volume conduction than PLV since it discards zero-lag synchrony.
    phases: (n_channels, n_samples). Returns real (n_channels, n_channels)."""
    n_channels, n_samples = phases.shape
    pli = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(n_channels):
            if i == j:
                pli[i, j] = 0.0  # PLI undefined/zero at zero lag by definition
                continue
            diff = phases[i] - phases[j]
            pli[i, j] = np.abs(np.mean(np.sign(np.sin(diff))))
    return pli


def build_complex_edge_weights(amplitude: np.ndarray, phase_diff_mean: np.ndarray) -> np.ndarray:
    """W_ij = a_ij * exp(i * phase_diff_mean_ij). Verifies Hermiticity
    (W_ji == conj(W_ij)) before returning — raises if violated beyond
    floating-point tolerance, rather than silently returning a broken
    graph to the GRN.

    NOTE: since amplitude and phase_diff_mean both come from the same
    complex mean when using compute_plv_matrix/compute_mean_phase_diff
    together, this reconstructs that same complex matrix — but the
    signature is kept as (amplitude, phase_diff_mean) per the original
    design so amplitude can come from a different source (e.g. a
    correlation-based weighting) if ever needed."""
    W = amplitude * np.exp(1j * phase_diff_mean)
    if not np.allclose(W, W.conj().T, atol=1e-8):
        max_err = np.max(np.abs(W - W.conj().T))
        raise ValueError(
            f"build_complex_edge_weights: Hermiticity violated, max |W - W^H| = {max_err:.2e} "
            "— check that amplitude is symmetric and phase_diff_mean is antisymmetric."
        )
    return W


def build_magnetic_laplacian(W: np.ndarray, amplitude: np.ndarray) -> np.ndarray:
    """L_C = D - W, where D_ii = sum_j amplitude_ij (real amplitudes,
    NOT the complex W, per the original design — the degree is a
    real-valued measure of total connection strength at node i).
    W: complex (n,n), amplitude: real (n,n). Returns complex (n,n),
    Hermitian by construction if W is Hermitian."""
    D = np.diag(amplitude.sum(axis=1))
    return D - W
