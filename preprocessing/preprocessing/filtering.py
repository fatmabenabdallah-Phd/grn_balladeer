"""
grn_balladeer.preprocessing.filtering
=======================================
Module 2b (part 2, step 1) — bandpass and notch filtering.

Thin wrappers around MNE's built-in filtering, applied in-place on a copy
of the Raw object (never mutates the caller's original Raw).
"""

from __future__ import annotations

from typing import List

import mne


def bandpass_filter(raw: mne.io.Raw, l_freq: float = 1.0, h_freq: float = 45.0) -> mne.io.Raw:
    """Zero-phase FIR bandpass filter, default 1-45 Hz (covers all 5 EEG
    bands used later in Module 3/10: delta through gamma).
    Returns a filtered COPY — the original raw passed in is untouched."""
    raw_filtered = raw.copy()
    raw_filtered.filter(l_freq=l_freq, h_freq=h_freq, method="fir", phase="zero", verbose=False)
    return raw_filtered


def notch_filter(raw: mne.io.Raw, freqs: List[float] = None) -> mne.io.Raw:
    """Removes powerline noise. Default targets 50 Hz + its first
    harmonic (100 Hz) — Tunisia uses 50 Hz mains, NOT 60 Hz. Adjust if
    recordings were made on 60 Hz-mains hardware/location.
    Returns a filtered COPY."""
    if freqs is None:
        freqs = [50.0, 100.0]
    raw_filtered = raw.copy()
    raw_filtered.notch_filter(freqs=freqs, verbose=False)
    return raw_filtered


def apply_standard_filters(
    raw: mne.io.Raw, l_freq: float = 1.0, h_freq: float = 45.0, notch_freqs: List[float] = None
) -> mne.io.Raw:
    """Convenience wrapper: notch first (remove powerline noise before it
    can alias into the passband edges), then bandpass. Returns a copy."""
    raw_out = notch_filter(raw, freqs=notch_freqs)
    raw_out = bandpass_filter(raw_out, l_freq=l_freq, h_freq=h_freq)
    return raw_out
