"""
grn_balladeer.model.eegnet
============================
EEGNet (Lawhern et al. 2018, "EEGNet: A Compact Convolutional Network
for EEG-based Brain-Computer Interfaces") -- the standard lightweight
CNN reference architecture for small-sample EEG classification.

Motivation for adding this now: EEG-TACT (Cardenas-Pena, Technologies
2026), a CONFIRMED subject-disjoint (patient-independent, stratified
group CV) study on the Nasrabadi IEEE DataPort ADHD dataset (also
N=121, comparable size to our BALLADEER cohort), uses an EEGNet-style
convolutional embedding as its backbone and reports 87.5% subject-level
accuracy -- a genuine, verified data point (not a sample-level-leakage
artifact) that our own GRN/TCN architectures have not come close to on
BALLADEER. EEGNet's core design difference from our lightweight TCN
(Section on the lightweight architecture, this session): EEGNet learns
SPATIAL (cross-channel) filters JOINTLY with temporal filters via a
depthwise convolution across the channel dimension, rather than
processing each channel independently before a separate, fixed
graph-aggregation step. This is a genuine architectural difference,
not a cosmetic one, and is the one concrete lead this session's
literature search turned up that is backed by a verified subject-level
result on a comparably-sized cohort.

Standard hyperparameters below (F1=8, D=2, F2=16, kernel_length=64,
dropout=0.5) are Lawhern et al.'s own defaults, not re-derived here --
deviating from them without a specific reason would make comparison to
the published architecture less meaningful.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """Standard EEGNet architecture, PyTorch reimplementation.

    Input: (B, 1, n_channels, n_timepoints) -- a single "image channel"
    holding the raw EEG signal, spatial dim = electrodes, temporal dim
    = time (matches the (n_channels, n_timepoints) raw epoch tensors
    already produced by data/build_dataset_lightweight.py -- reshape
    with .unsqueeze(1) before feeding in, no new data pipeline needed).

    Output: (B, n_classes) logits.
    """

    def __init__(
        self,
        n_channels: int,
        n_timepoints: int,
        n_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()

        # Block 1: temporal conv, then depthwise spatial conv across channels
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False),  # depthwise spatial
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        # Block 2: separable conv (depthwise temporal + pointwise)
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=1, bias=False),  # pointwise
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        # Infer the flattened feature size with a dummy forward pass,
        # rather than hand-computing it -- avoids an off-by-one error
        # from padding/pooling arithmetic silently producing a wrong
        # Linear layer input size.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            dummy_out = self.block2(self.block1(dummy))
            flat_size = dummy_out.numel()

        self.classify = nn.Linear(flat_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, n_channels, n_timepoints) or (B, n_channels,
        n_timepoints) -- the unsqueeze is done automatically if the
        channel dim is missing, for convenience."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(start_dim=1)
        return self.classify(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
