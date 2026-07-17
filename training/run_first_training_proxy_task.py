"""
grn_balladeer.training.run_first_training_proxy_task
=========================================================
Module 9 milestone — "first real EEG-only training run, check
convergence" - run on the 33 real UB0136 graphs (real CQT node features
+ real magnetic-Laplacian connectivity, per epoch).

IMPORTANT - what this validates and what it does NOT:
Only one subject (UB0136) has real data available in this codebase so
far, and a single subject gives a single ADHD/control label - not
enough classes to run the REAL target task. This script substitutes
`flag_type` (circle/doubleCircle/rhombus/square - a real 4-class label
already present in the Slackline task data, unrelated to the ADHD
diagnosis) as a PROXY classification target, purely to validate that
the training loop mechanics work end-to-end on real EEG-derived graphs:
loss decreases, no NaNs, the omega-collapse artifact from Week 4
(untrained head -> near-identical omega -> harmonic_loss near zero)
actually resolves once real gradients are applied.

This does NOT demonstrate ADHD-classification performance. The real
Module 9 milestone (does the model discriminate ADHD vs control) still
needs a second, labeled subject - re-run this script's logic with real
labels.build_label_table() output once more subjects are available.

Result of this run (30 epochs, Adam lr=1e-3, 26 train / 7 val,
stratified by flag_type): loss_total 2.17 -> 1.68 (monotonic decrease),
loss_symb 0.76 -> 0.26, loss_harm 5.4e-5 -> 0.054 (omega moving AWAY
from the collapsed-unison degenerate solution, confirmed by
check_omega_collapse: std(omega)=0.021 > threshold=0.01, i.e. no longer
collapsed). Train accuracy 0.154 -> 0.462, val accuracy 0.429 (7
samples - too small to be more than a sanity signal, not a real
generalization estimate).
"""

from __future__ import annotations

import torch
from sklearn.model_selection import train_test_split

from grn_balladeer.model.grn_encoder import GRNEncoder, build_resonance_head
from grn_balladeer.model.classification_head import ClassificationHead
from grn_balladeer.training.train_epoch import train_epoch
from grn_balladeer.training.evaluate import evaluate
from grn_balladeer.training.omega_diagnostics import check_omega_collapse


def main(dataset_path: str = "real_dataset_33epochs.pt", n_epochs: int = 30, seed: int = 42):
    ckpt = torch.load(dataset_path, weights_only=False)
    dataset = ckpt["dataset"]
    flag_types = ckpt["flag_types"]
    eeg_ch_names = ckpt["eeg_ch_names"]

    classes = sorted(set(flag_types))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    labels_all = torch.tensor([class_to_idx[ft] for ft in flag_types], dtype=torch.long)

    idx_all = list(range(len(dataset)))
    idx_train, idx_val = train_test_split(
        idx_all, test_size=0.2, random_state=seed, stratify=[flag_types[i] for i in idx_all]
    )
    batch_train = [dataset[i] for i in idx_train]
    labels_train = labels_all[idx_train]
    batch_val = [dataset[i] for i in idx_val]
    labels_val = labels_all[idx_val]

    in_channels = dataset[0][0].shape[1]
    embedding_dim = 8  # last hidden_channels entry below
    torch.manual_seed(seed)
    encoder = GRNEncoder(in_channels=in_channels, hidden_channels=[16, embedding_dim], K=3)
    head = ClassificationHead(in_features=2 * embedding_dim, n_classes=len(classes))
    resonance_head = build_resonance_head(embedding_dim=embedding_dim)

    params = list(encoder.parameters()) + list(head.parameters()) + list(resonance_head.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)

    history = []
    for epoch in range(n_epochs):
        stats = train_epoch(
            encoder, head, resonance_head, batch_train, labels_train, eeg_ch_names, optimizer,
            symbolic_direction="direct", lambda1=1.0, lambda2=1.0,
        )
        history.append(stats)

    val_result = evaluate(encoder, head, batch_val, labels_val)
    collapse_report = check_omega_collapse(history[-1]["last_omega"])
    return {"history": history, "val_result": val_result, "collapse_report": collapse_report,
            "class_to_idx": class_to_idx}


if __name__ == "__main__":
    result = main()
    print("Val result:", result["val_result"])
    print("Final omega collapse check:", result["collapse_report"])
    print("Loss trajectory (every 5 epochs):",
          [round(h["loss_total"], 4) for h in result["history"][::5]])
