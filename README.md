# grn_balladeer

GRN pipeline for EEG-based ADHD classification (BALLADEER project).

## Organization

```
grn_balladeer/
├── data/          # Module 1 — labels, demographics mapping, stratified split
├── connectivity/  # Module 3 — PLV, magnetic Laplacian (coming soon)
├── model/         # Module 4-6 — CQT, GRNEncoder, classification head (coming soon)
├── losses/        # Module 7, 7b — harmonic, symbolic, triplet (coming soon)
├── training/       # Module 8, 9 — dual-branch, CV loop (coming soon)
├── eval/          # Module 10-13 — baselines, ablations, XAI
├── configs/       # one YAML file per dataset (BALLADEER, future dataset X)
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

- [x] Module 1 (labels) — verified on the 158 real records
- [x] Module 2a (CGX/Emotiv channels confirmed)
- [x] Module 10 (SVM/RF/theta-beta ratio baselines) — code ready, awaiting
      preprocessed epochs (Module 2b) to actually run
- [ ] Module 2b, 3, 4-9, 11-13

