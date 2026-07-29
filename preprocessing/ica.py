"""
grn_balladeer.preprocessing.ica
=================================
Module 2b (part 2, step 2) — ICA-based artifact removal.

Two scenarios, handled automatically:
  1. CGX with ExG channels loaded as 'eog' type (see
     load_eeg_cgx(include_exg_as_eog=True)) -> standard MNE
     ica.find_bads_eog() against the real EOG reference.
  2. No dedicated EOG channel (Emotiv EPOCX has none) -> fallback:
     correlate each component's time course against the average of the
     frontal-most channels available (a common practical proxy for
     blink artifacts when no EOG reference exists) and flag high-
     correlation components.

The fallback is a heuristic, not a validated method — documented as such
so it is not mistaken for the standard EOG-based approach in the paper.
"""

from __future__ import annotations

from typing import List, Optional

import mne
import numpy as np
from mne.preprocessing import ICA

# Frontal-most channels available per device, used only by the fallback
# heuristic when no EOG/ExG reference channel is present.
_FRONTAL_FALLBACK_CHANNELS = {
    "cgx": ["Fp1", "Fp2", "AF7", "AF8", "Fpz"],
    "emotiv": ["AF3", "AF4"],
}


def _fallback_frontal_proxy(raw: mne.io.Raw) -> Optional[np.ndarray]:
    """Builds a blink-proxy signal by averaging whichever frontal
    channels from _FRONTAL_FALLBACK_CHANNELS are actually present in
    `raw`. Returns None if none of them are found."""
    available = [ch for chs in _FRONTAL_FALLBACK_CHANNELS.values() for ch in chs if ch in raw.ch_names]
    if not available:
        return None
    data = raw.get_data(picks=available)
    return data.mean(axis=0)


def _is_dead_reference(raw: mne.io.Raw, ch_names: List[str], std_threshold: float = 1e-10) -> bool:
    """Checks whether the given channel(s) are effectively flat
    (std below std_threshold) -- i.e. a dead/non-functional reference,
    not a genuine signal. CGX's ExG channels (typed 'eog' via
    load_eeg_cgx(include_exg_as_eog=True)) are flat on every subject
    checked, meaning find_bads_eog() against them never excludes
    anything -- ICA-based artifact removal was silently inactive
    throughout this project as a result. Returns True if ALL given
    channels are flat (a partial reference, e.g. one working EOG
    channel out of two, is still usable and should not trigger the
    fallback).
    """
    data = raw.get_data(picks=ch_names)
    return bool(np.all(data.std(axis=1) < std_threshold))


def run_ica_artifact_removal(
    raw: mne.io.Raw,
    n_components: int = 15,
    random_state: int = 42,
    corr_threshold: float = 0.3,
) -> "tuple[mne.io.Raw, dict]":
    """Fits ICA (FastICA) and removes blink/artifact components, then
    returns (cleaned_raw_copy, report_dict) — original `raw` untouched.

    - If `raw` contains channel(s) of type 'eog' AND they carry a real
      (non-flat) signal, uses MNE's standard ica.find_bads_eog()
      against them (preferred path).
    - If the 'eog'-typed channels are flat (a dead reference -- see
      _is_dead_reference, discovered via CGX's ExG channels being
      silently non-functional on every subject checked), automatically
      falls back to the frontal-channel-average proxy heuristic
      instead of trusting a reference known not to carry a real
      signal. This is what actually activates artifact removal on
      CGX for the first time in this project; previously the 'eog' type
      alone was enough to take the (broken) preferred path.
    - Otherwise (no 'eog'-typed channel at all, e.g. Emotiv), falls back
      to the same frontal-channel-average proxy directly, and flags any
      component with |correlation| > corr_threshold against it
      (heuristic — document this choice explicitly if used in the paper).
    """
    raw_for_ica = raw.copy()
    # ICA needs an average or similar reference and benefits from a
    # mild high-pass; assumes bandpass_filter() was already applied
    # upstream (Module 2b step 1) — this function does not re-filter.

    ica = ICA(n_components=n_components, random_state=random_state, method="fastica")
    ica.fit(raw_for_ica, verbose=False)

    eog_ch_names = [ch for ch, kind in zip(raw.ch_names, raw.get_channel_types()) if kind == "eog"]
    eog_is_dead = bool(eog_ch_names) and _is_dead_reference(raw_for_ica, eog_ch_names)

    if eog_ch_names and not eog_is_dead:
        bad_idx, scores = ica.find_bads_eog(raw_for_ica, ch_name=eog_ch_names, verbose=False)
        method_used = f"find_bads_eog against {eog_ch_names}"
    else:
        proxy = _fallback_frontal_proxy(raw_for_ica)
        if proxy is None:
            # No usable EOG channel and no frontal proxy available — return
            # unchanged rather than silently skipping artifact removal
            # without any signal to base it on.
            raise ValueError(
                "No usable EOG channel and no frontal fallback channel available for "
                "ICA artifact detection — check device channel list."
            )
        sources = ica.get_sources(raw_for_ica).get_data()
        corrs = np.array([np.corrcoef(s, proxy)[0, 1] for s in sources])
        bad_idx = np.where(np.abs(corrs) > corr_threshold)[0].tolist()
        method_used = (
            f"frontal-proxy correlation (threshold={corr_threshold}), "
            f"triggered by dead EOG reference {eog_ch_names}"
            if eog_is_dead
            else f"frontal-proxy correlation (threshold={corr_threshold})"
        )

    ica.exclude = bad_idx
    raw_clean = raw.copy()
    ica.apply(raw_clean, verbose=False)

    ica_report = {
        "n_components_fit": n_components,
        "n_excluded": len(bad_idx),
        "excluded_indices": bad_idx,
        "method": method_used,
        "eog_reference_was_dead": eog_is_dead,
    }
    return raw_clean, ica_report
