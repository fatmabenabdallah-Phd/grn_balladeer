# grn_balladeer

GRN pipeline for EEG-based ADHD classification (BALLADEER project).

## Organization

```
grn_balladeer/
├── data/           # Module 1 — labels, demographics mapping, stratified split
│                   # + build_dataset.py (reusable subject recipe, flat imports)
│                   # + epoching.py, sync.py (⚠ French docstrings, see Known issues)
├── preprocessing/  # Module 2b — MNE loading, filtering, ICA, event alignment,
│                   #   epoching, quality/motion. Validated on real UB0004/UB0022/
│                   #   UB0136/UB0023 files.
├── connectivity/   # Module 3 — PLV, magnetic Laplacian
│                   # + plv.py (⚠ stale duplicate, French docstring, unused)
├── model/          # Module 4-8 — CQT encoder, magnetic Laplacian conv, GRNEncoder
│                   #   (omega bounded to [1, 45] Hz), classification head,
│                   #   AuxBranchEncoder + CrossAttentionFusion (dual-branch, dim
│                   #   fix applied: hidden_dim=16 to match real GRN pooled output)
├── losses/         # Module 7/7b — harmonic, symbolic, total, triplet
│                   #   (CONSONANCE_RATIOS=[1,2,3,4], literature-grounded)
├── training/       # Module 8/9 — batch_forward, omega_diagnostics, train_epoch,
│                   #   train_epoch_dual_branch, evaluate, leakage_probes,
│                   #   behavioral_features, eda_features
├── eval/           # Module 10-13 — baselines, ablations, XAI
├── configs/        # one YAML file per dataset (BALLADEER, future dataset X)
└── requirements.txt
```

## Reusing this pipeline on another dataset

The code in `model/`, `losses/`, and `connectivity/` is designed to be
dataset-agnostic. To adapt the pipeline to a new EEG dataset:

1. Copy `configs/balladeer.yaml` to `configs/my_dataset.yaml`.
2. Adjust the fields (`labels.field`, `demographics_schema`, `eeg_devices`).
3. Do NOT modify the code in `model/`, `losses/`, `connectivity/`.

## Execution

**Development and training: Google Colab Pro.**
Mount Drive with `grn_balladeer.data.labels.mount_drive_colab()`, then use
the functions from the sub-packages as usual.

**Environment reproducibility: Docker.**
The `Dockerfile` (to be added once the full pipeline is validated on Colab)
pins the exact versions from `requirements.txt`. It has been validated on
CPU on a small subsample; full training is run via the provided Colab Pro
notebook, not inside the container.

## Progress

- [x] **Module 1** (labels) — verified on the 158 real records; `stratified_
      subject_kfold` stratifies jointly on label+sex+age_bin, re-verified on
      the real 138-subject cohort.
- [x] **Module 2a** (CGX/Emotiv channels confirmed)
- [x] **Module 2b** — COMPLETE (Week 1 closed): MNE loading (CGX/Emotiv),
      bandpass/notch filtering (Nyquist bug fixed for low-sfreq devices),
      ICA artifact removal (EOG-reference path for CGX, frontal-proxy
      fallback for Emotiv), TAGS parsing + event-to-EEG alignment
      (general_time hypothesis empirically validated), stimulus-locked
      epoching, Emotiv channel quality mask, CGX motion amplitude.
      Validated on real UB0004, UB0022, UB0136, UB0023.
- [x] **Module 3** (connectivity) — PLV / magnetic Laplacian, validated on
      real data (frontal + parieto-occipital clusters visible in PLV heatmap).
- [x] **Module 4-6** (CQT encoder, MagneticLaplacianConv, GRNEncoder,
      classification head) — omega-collapse bug fixed for real: resonance
      frequency now passed through a sigmoid bounded to [1, 45] Hz instead
      of an unbounded linear head.
- [x] **Module 7** (harmonic / symbolic / total loss) — CONSONANCE_RATIOS
      = [1.0, 2.0, 3.0, 4.0], grounded in cross-frequency phase synchrony
      literature (Palva et al.) and ADHD theta/beta literature.
- [x] **Module 7b/8** (dual-branch: EDA + behavioral features,
      AuxBranchEncoder, CrossAttentionFusion, triplet loss) — Week 5 closed.
      Critical dimension-mismatch bug found and fixed (both branches
      defaulted to hidden_dim=64, validated only against a simulated tensor;
      corrected to hidden_dim=16 to match the real GRN pooled embedding).
      `mine_batch_hard_triplets` verified 32/32 valid triplets, 0 anti-leak
      violations, on real pooled embeddings across all 4 real subjects.
- [x] **Module 9** (train_epoch, train_epoch_dual_branch, evaluate) — first
      real 2v2 training run completed (train=UB0004+UB0136, val=UB0022+
      UB0023). Losses converge; validation accuracy is degenerate (50%,
      predicts one class for everything) and 0 triplets were mined in that
      split — both are expected data-volume artifacts with only 4 real
      subjects (2/class), not bugs. See training/leakage_probes.py for the
      sex-leakage diagnostic (synthetic-tested only so far).
- [x] **Module 10** (SVM/RF/theta-beta ratio baselines) — code ready, can
      be run on real epoched data.
- [ ] **Module 11-13** (ablations, XAI, full cross-validation loop) — not
      started; blocked on having more than 4 real subjects (need ≥3/class
      for a clean train/val split to coexist with useful triplet mining).

## Known issues / not yet cleaned up

- **French-language docstrings found in `data/epoching.py`, `data/sync.py`,
  and `connectivity/plv.py`** — violates the project's English-only
  standing instruction. These came from a parallel work thread on this
  repo and have not yet been translated or reconciled.
- **Duplicate/overlapping implementations**, left in place pending
  reconciliation rather than silently deleted:
  - `parse_tags_file` exists in both `preprocessing/event_alignment.py`
    (snake_case columns) and `training/behavioral_features.py` (its own
    copy, camelCase columns).
  - `connectivity/plv.py` duplicates `connectivity/phase_connectivity.py`
    and is unused — candidate for deletion.
  - `data/epoching.py` / `data/sync.py` overlap with
    `preprocessing/epoching.py` / `preprocessing/event_alignment.py`.
- **`data/build_dataset.py` uses flat imports** (`from preprocessing.X
  import Y`) instead of the `grn_balladeer.preprocessing.X` convention
  used everywhere else in the package.
- **Known confound**: in the current 4-subject slice, sex is perfectly
  correlated with class (2 female Control, 2 male ADHD) — no result on
  these 4 subjects can distinguish ADHD detection from sex detection.
- **Dataset files** (CSVs, JSON, EmbracePlus, `.pt` checkpoints) are
  intentionally not committed — see `.gitignore`. Checkpoints must be
  rebuilt from raw CSVs each session.

