"""
grn_balladeer.data.load_tdbrain_eeg
=======================================
Loads TDBRAIN's BIDS-structured BDF recordings and extracts features
for classification, mirroring the feature-engineering conventions
already used for BALLADEER/Nasrabadi in this project.

File structure (fully deterministic BIDS, unlike the CGX/Emotiv
datasets elsewhere in this project, which needed a tolerant glob):
  <dataset_root>/sub-<ID>/ses-1/eeg/sub-<ID>_ses-1_task-<TASK>_eeg.bdf

26 real EEG channels (10-20/10-10, modern nomenclature). 6 auxiliary
channels (EOG/ECG/EMG) are RETAINED through loading and correctly
typed (not immediately dropped), since preprocessing.ica.
run_ica_artifact_removal needs the real EOG channels (VPVA/VNVB,
HPHL/HNHR) to detect and remove blink/eye-movement components before
they are dropped for downstream feature extraction. This differs from
an earlier version of this module, which excluded all 6 auxiliary
channels immediately at load time and applied no artifact removal at
all -- a gap identified and closed in this revision, reusing
preprocessing.ica unchanged rather than reimplementing ICA logic here.

Unlike BALLADEER's CGX system (whose nominal EOG reference was found
silently non-functional throughout that project), TDBRAIN's VPVA/VNVB/
HPHL/HNHR channels are expected to carry a genuine EOG signal (no
evidence of a dead reference found in this dataset so far) -- so the
standard find_bads_eog() path in preprocessing.ica should be the one
actually used here, not its frontal-proxy fallback. This is verified
per-subject at runtime by run_ica_artifact_removal's own dead-reference
check, not assumed.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import mne

from grn_balladeer.preprocessing.ica import run_ica_artifact_removal

TDBRAIN_EEG_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC3", "FCz", "FC4",
    "T7", "C3", "Cz", "C4", "T8", "CP3", "CPz", "CP4",
    "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2",
]
TDBRAIN_EOG_CHANNELS = ["VPVA", "VNVB", "HPHL", "HNHR"]
TDBRAIN_ECG_CHANNELS = ["Erbs"]
TDBRAIN_EMG_CHANNELS = ["Mass"]
TDBRAIN_SFREQ = 500.0
TDBRAIN_WINDOW_SAMPLES = 2000  # 4.0s at 500Hz, matching this project's Nasrabadi epoching

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def find_tdbrain_subject_file(dataset_root: str, user_id: str, task: str = "restEC") -> Optional[str]:
    """Deterministic BIDS path construction. user_id may be given either
    as "sub-88053677" or bare "88053677" -- normalized here so callers
    don't need to track which form label_df happens to use.
    """
    import os
    bare_id = user_id.replace("sub-", "")
    path = os.path.join(
        dataset_root, f"sub-{bare_id}", "ses-1", "eeg",
        f"sub-{bare_id}_ses-1_task-{task}_eeg.bdf",
    )
    return path if os.path.isfile(path) else None


def load_tdbrain_raw_epochs(
    bdf_path: str,
    window_samples: int = TDBRAIN_WINDOW_SAMPLES,
    apply_ica: bool = True,
    ica_n_components: float = 0.999999,
    ica_random_state: int = 42,
) -> "tuple[np.ndarray, dict]":
    """Loads one subject's BDF recording via MNE, optionally applies
    ICA-based ocular-artifact removal (reusing preprocessing.ica.
    run_ica_artifact_removal unchanged), THEN picks only the 26 real
    EEG channels (dropping EOG/ECG/EMG/Status only after ICA has had a
    chance to use the EOG channels), and splits into non-overlapping
    windows.

    apply_ica: if True (default), loads with EOG/ECG/EMG channels
    correctly typed, runs run_ica_artifact_removal, then drops them.
    If False, reproduces the earlier (pre-ICA) behavior exactly --
    channel selection only, no artifact removal -- kept for direct
    before/after comparison against previously-obtained results, not
    as a recommended default going forward.

    ica_n_components: a FRACTION of explained variance (default
    0.999999, MNE's own convention for "keep effectively all real
    signal, drop only numerical noise"), NOT a fixed integer count.
    A first attempt with a fixed n_components=15 produced a
    RuntimeWarning on most subjects ("ratio between the largest and
    smallest variances is too large (> 1e6)"), with the safe integer
    ceiling MNE itself suggested varying wildly by subject (as low as
    3, as high as 14) -- meaning a fixed count is unstable for some
    subjects and needlessly conservative for others. The
    variance-fraction convention lets MNE select the actual rank of
    each subject's own data automatically, rather than assuming every
    subject's 26-channel recording has the same effective
    dimensionality.

    Returns (epochs, ica_report) where epochs has shape
    (n_epochs, 26, window_samples), and ica_report is None if
    apply_ica=False, else the dict returned by run_ica_artifact_removal
    (method used, number of components excluded, whether the EOG
    reference was found dead) -- callers should log/aggregate this
    across subjects rather than discard it, since a dead-reference
    finding here would be as reportable as it was for BALLADEER.
    """
    raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False)

    ica_report = None
    if apply_ica:
        ch_type_map = {}
        for ch in TDBRAIN_EOG_CHANNELS:
            if ch in raw.ch_names:
                ch_type_map[ch] = "eog"
        for ch in TDBRAIN_ECG_CHANNELS:
            if ch in raw.ch_names:
                ch_type_map[ch] = "ecg"
        for ch in TDBRAIN_EMG_CHANNELS:
            if ch in raw.ch_names:
                ch_type_map[ch] = "emg"
        if ch_type_map:
            raw.set_channel_types(ch_type_map, verbose=False)

        # ICA needs a mild high-pass and benefits from an average-ish
        # reference; apply a standard 1-45Hz bandpass here (matching
        # this project's other datasets' preprocessing) since nothing
        # upstream of this function does so for TDBRAIN otherwise.
        raw.filter(l_freq=1.0, h_freq=45.0, verbose=False)

        raw, ica_report = run_ica_artifact_removal(
            raw, n_components=ica_n_components, random_state=ica_random_state
        )

    raw.pick(TDBRAIN_EEG_CHANNELS)  # explicit whitelist, not exclude()
    data = raw.get_data()  # (26, n_samples), already in the order of TDBRAIN_EEG_CHANNELS
    n_samples = data.shape[1]
    n_epochs = n_samples // window_samples
    if n_epochs == 0:
        raise ValueError(
            f"load_tdbrain_raw_epochs: recording has only {n_samples} samples, "
            f"shorter than one window ({window_samples})."
        )
    trimmed = data[:, : n_epochs * window_samples]
    epochs = trimmed.reshape(len(TDBRAIN_EEG_CHANNELS), n_epochs, window_samples)
    epochs = np.transpose(epochs, (1, 0, 2))  # (n_epochs, 26, window_samples)
    return epochs, ica_report


def extract_band_power_features(epochs: np.ndarray, sfreq: float = TDBRAIN_SFREQ) -> np.ndarray:
    """Welch-method band-power features (5 bands x 26 channels + 1
    global theta/beta ratio), averaged across epochs -- same
    convention as this project's eval.baselines.extract_band_power_features.

    epochs: (n_epochs, 26, window_samples). Returns (131,) float64.
    """
    from scipy.signal import welch

    n_epochs, n_channels, n_samples = epochs.shape
    band_powers = np.zeros((n_epochs, n_channels, len(BANDS)))

    for e in range(n_epochs):
        for ch in range(n_channels):
            freqs, psd = welch(epochs[e, ch], fs=sfreq, nperseg=min(n_samples, int(sfreq * 2)))
            for b_idx, (fmin, fmax) in enumerate(BANDS.values()):
                mask = (freqs >= fmin) & (freqs < fmax)
                band_powers[e, ch, b_idx] = np.trapezoid(psd[mask], freqs[mask]) if mask.any() else 0.0

    mean_band_powers = band_powers.mean(axis=0)  # (26, 5)
    theta_idx, beta_idx = list(BANDS.keys()).index("theta"), list(BANDS.keys()).index("beta")
    theta_beta_ratio = np.mean(
        mean_band_powers[:, theta_idx] / np.where(mean_band_powers[:, beta_idx] > 0, mean_band_powers[:, beta_idx], 1e-12)
    )
    return np.concatenate([mean_band_powers.flatten(), [theta_beta_ratio]])


def extract_connectivity_features(
    epochs: np.ndarray,
    sfreq: float = TDBRAIN_SFREQ,
    metric: str = "plv",
    band: tuple = (8.0, 13.0),
) -> np.ndarray:
    """Phase-connectivity features (PLV or PLI) between all channel
    pairs, averaged across epochs, in a single frequency band
    (default alpha, 8-13Hz -- matching this project's default
    single-band connectivity convention for BALLADEER/Nasrabadi).

    epochs: (n_epochs, 26, window_samples).
    Returns the upper-triangle of the (26,26) connectivity matrix,
    flattened: (26*25/2,) = (325,) features.
    """
    from scipy.signal import butter, filtfilt, hilbert

    n_epochs, n_channels, n_samples = epochs.shape
    fmin, fmax = band
    nyq = sfreq / 2
    b, a = butter(4, [fmin / nyq, fmax / nyq], btype="band")

    n_pairs = n_channels * (n_channels - 1) // 2
    conn_per_epoch = np.zeros((n_epochs, n_pairs))

    for e in range(n_epochs):
        filtered = filtfilt(b, a, epochs[e], axis=-1)
        phases = np.angle(hilbert(filtered, axis=-1))  # (26, window_samples)

        pair_idx = 0
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                phase_diff = phases[i] - phases[j]
                if metric == "plv":
                    conn_per_epoch[e, pair_idx] = np.abs(np.mean(np.exp(1j * phase_diff)))
                elif metric == "pli":
                    conn_per_epoch[e, pair_idx] = np.abs(np.mean(np.sign(np.sin(phase_diff))))
                else:
                    raise ValueError(f"extract_connectivity_features: unknown metric '{metric}'")
                pair_idx += 1

    return conn_per_epoch.mean(axis=0)


def load_all_subjects_tdbrain(
    label_df,
    dataset_root: str,
    task: str = "restEC",
    feature_type: str = "band_power",
    metric: str = "plv",
    apply_ica: bool = True,
):
    """Full loop over label_df's subjects: loads EEG (optionally via ICA
    artifact removal, see load_tdbrain_raw_epochs), extracts the
    requested feature type, aggregates to one row per subject.

    Returns (X, y, subject_ids, failed_subjects, ica_reports) --
    failed_subjects is a list of (user_id, reason) for subjects whose
    file was missing or failed to load, so callers can report coverage
    honestly rather than silently dropping subjects. ica_reports is a
    dict {user_id: report} (None values if apply_ica=False) -- callers
    should check how many subjects had eog_reference_was_dead=True
    across the cohort and report this explicitly if non-zero, the same
    way BALLADEER's dead-EOG-reference finding was reported rather than
    silently absorbed into a "cleaning" step.
    """
    features_by_subject = {}
    failed_subjects = []
    ica_reports = {}

    for _, row in label_df.iterrows():
        user_id = row["user_id"]
        bdf_path = find_tdbrain_subject_file(dataset_root, user_id, task=task)
        if bdf_path is None:
            failed_subjects.append((user_id, "fichier introuvable"))
            continue
        try:
            epochs, ica_report = load_tdbrain_raw_epochs(
                bdf_path, TDBRAIN_WINDOW_SAMPLES, apply_ica=apply_ica
            )
            ica_reports[user_id] = ica_report
            if feature_type == "band_power":
                features = extract_band_power_features(epochs)
            elif feature_type == "connectivity":
                features = extract_connectivity_features(epochs, metric=metric)
            else:
                raise ValueError(f"load_all_subjects_tdbrain: unknown feature_type '{feature_type}'")
            features_by_subject[user_id] = features
        except Exception as e:
            failed_subjects.append((user_id, str(e)))

    subject_ids = sorted(features_by_subject.keys())
    X = np.stack([features_by_subject[s] for s in subject_ids]) if subject_ids else np.array([])
    label_lookup = label_df.set_index("user_id")["label"]
    y = np.array([label_lookup.loc[s] for s in subject_ids])

    return X, y, subject_ids, failed_subjects, ica_reports
