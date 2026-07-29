# grn_balladeer

GRN (Graph Resonance Network) pipeline for EEG-based ADHD
classification, developed and validated on the BALLADEER multimodal
ADHD dataset, with cross-dataset validation on the Nasrabadi IEEE
DataPort ADHD cohort. Full methodology, results, and discussion are in
the accompanying manuscript (target: *Biomedical Signal Processing and
Control*, Elsevier).

## Summary of validated results

- **BALLADEER (N=114-121, subject-disjoint 5-fold CV):** every deep
  architecture tested (GRN, lightweight TCN+structural-graph, TCN
  alone, EEGNet, self-supervised autoencoder) plateaus at or near
  chance (AUC 0.469-0.517). Random Forest on hand-crafted band-power
  features, trained at the epoch level, is the most accurate and
  stable configuration found (AUC 0.662-0.696).
- **Nasrabadi cross-dataset validation (N=121, identical
  subject-disjoint protocol, unmodified code):** the same Random
  Forest, EEGNet, and GRN pipelines reach AUC 0.746, 0.869±0.095, and
  0.763±0.130 respectively -- matching or exceeding published
  subject-disjoint benchmarks on that cohort. This resolves an
  ambiguity BALLADEER alone could not: GRN's fixed-connectivity,
  single-scalar-per-node design is not itself incapable of learning
  EEG-ADHD signal (statistically indistinguishable from TCN alone,
  p=1.000); the chance-level BALLADEER result reflects the dataset,
  not the architecture.
- **Preprocessing audit:** BALLADEER's CGX EOG reference channels were
  found silently non-functional throughout, and 85.1% of subjects had
  at least one detected bad channel, concentrated frontally (54.6% of
  detections) -- overlapping the electrode cluster GRN's own
  connectivity graph and symbolic loss depend on, a candidate
  mechanism for GRN's specific vulnerability to this data-quality
  issue.
- **Fusion strategies:** across four strategies for combining a
  validated classical baseline (Random Forest) with an exploratory
  deep-learning branch, late-fusion stacking degrades gracefully when
  the deep branch is uninformative, while naive mid-fusion
  concatenation can actively suppress a genuinely useful branch
  through dimensional imbalance.

See the manuscript for the full benchmark, ablations, statistical
tests, and discussion.

## Organization

```
grn_balladeer/
├── data/           # Dataset loading, labels/demographics, subject-file
│                   #   discovery, dataset construction (BALLADEER + Nasrabadi),
│                   #   eye-tracking and EPOCX (session-1 EEG) feature extraction
├── preprocessing/  # MNE loading, filtering, ICA (with dead-reference
│                   #   detection and fallback), bad-channel detection/
│                   #   interpolation, epoch rejection, event alignment
├── connectivity/   # PLV/PLI functional connectivity, magnetic Laplacian,
│                   #   structural (k-NN) graph construction
├── model/          # CQT encoder, magnetic Laplacian graph convolution,
│                   #   GRNEncoder, classification head, auxiliary-branch
│                   #   encoder/decoder, cross-attention fusion, EEGNet,
│                   #   lightweight TCN encoder, EEG autoencoder
├── losses/         # Harmonic/symbolic/triplet/total losses (neurosymbolic
│                   #   consonance constraints, literature-grounded), and
│                   #   a clinician-rule bridge (structured, human-confirmed
│                   #   natural-language rules -> loss terms; prototype,
│                   #   not yet empirically validated -- see its own
│                   #   module docstring)
├── training/       # Training loops (EEG-only, dual-branch, batched/
│                   #   vectorized variants), cross-validation driver,
│                   #   evaluation metrics, leakage probes (subject-identity,
│                   #   sex), behavioral/EDA feature extraction
├── eval/           # Classical baselines (SVM/RF/XGBoost/LightGBM),
│                   #   band-power feature extraction, evaluation metrics
├── configs/        # One YAML file per dataset (BALLADEER, future datasets)
└── requirements.txt
```

## Reusing this pipeline on another dataset

The code in `model/`, `losses/`, and `connectivity/` is designed to be
dataset-agnostic. To adapt the pipeline to a new EEG dataset:

1. Copy `configs/balladeer.yaml` to `configs/my_dataset.yaml`.
2. Adjust the fields (`labels.field`, `demographics_schema`, `eeg_devices`).
3. Do NOT modify the code in `model/`, `losses/`, `connectivity/`.
4. If the new dataset's recordings are continuous rather than
   event-locked, see `data/build_dataset_nasrabadi.py` for the
   fixed-window adaptation used for Nasrabadi.

## Execution

**Development and training: Google Colab.**
Mount Drive with `grn_balladeer.data.labels.mount_drive_colab()`, then use
the functions from the sub-packages as usual.

**Environment reproducibility:** exact package versions are pinned in
`requirements.txt`.

## Known issues / not yet fully reconciled

- **Duplicate/overlapping implementations**, left in place pending
  reconciliation rather than silently deleted:
  - `connectivity/plv.py` duplicates `connectivity/phase_connectivity.py`
    and is unused -- self-documented in its own docstring as a removal
    candidate.
  - `data/epoching.py` / `data/sync.py` (subject-level epoch cutting +
    TAGS/EEG sync) overlap with `preprocessing/epoching.py` /
    `preprocessing/event_alignment.py` -- not yet reconciled into one
    code path.
  - `preprocessing/build_dataset_lightweight.py` and
    `data/build_dataset_lightweight.py` overlap significantly; the
    `data/` version is the more complete, actively used one
    (adds common-average-reference and surface-Laplacian options).
- **`losses/clinician_rules.py`** is a functional prototype (structured
  schema, validation, human-confirmation gate, loss-term generation)
  but has not been empirically validated: no translation-fidelity
  study, and no ablation showing clinician-authored rules change model
  behavior. Treat as a starting point for a dedicated study, not a
  validated contribution.
- **Dataset files** (CSVs, JSON, EmbracePlus, `.pt` checkpoints) are
  intentionally not committed -- see `.gitignore`. Checkpoints must be
  rebuilt from raw CSVs, or reloaded from Drive if already built.
