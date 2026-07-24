"""
grn_balladeer.model.eeg_autoencoder
======================================
Self-supervised autoencoder for raw EEG epoch reconstruction. Motivated
by this session's finding that four supervised deep architectures
(GRN, lightweight TCN, TCN-only, EEGNet) all perform at chance on
BALLADEER, while classical ML on hand-crafted band-power features
reaches AUC~0.66-0.67: the hypothesis here is that SUPERVISED training
on ~91-97 labeled subjects may simply be too little signal for a deep
net to learn a good representation from scratch, whereas an
unsupervised reconstruction objective (no labels needed) can use the
same raw signal more efficiently, since every epoch is useful training
signal regardless of its label.

LEAKAGE DISCIPLINE: the encoder must be pretrained ONLY on the current
fold's TRAINING subjects' epochs, never on validation-subject epochs,
even though pretraining itself doesn't use labels -- otherwise the
encoder would have "seen" validation-subject signal characteristics
before evaluation, a real (if subtle) form of leakage. This module
does not enforce that itself (it just reconstructs whatever epochs
it's given); the caller (training loop) is responsible for only ever
passing training-fold subjects' epochs to fit_autoencoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EEGAutoencoder(nn.Module):
    """Encoder: per-channel dilated causal conv stack (same TCN block
    design as LightweightTCNEncoder, for a fair architectural
    comparison) -> per-node embedding. Decoder: mirrors the encoder to
    reconstruct the raw per-channel time series from the embedding.

    Encoder output (the reusable part) matches LightweightTCNEncoder's
    shape convention: (B, N, hidden_channels), one embedding per node
    (electrode), so the SAME downstream fusion/classification code
    (LightweightClassifier from train_epoch_lightweight.py) can be
    reused without modification -- only the encoder's weights differ
    (pretrained via reconstruction here, vs. trained from scratch
    jointly with the classifier before).
    """

    def __init__(self, hidden_channels: int = 8, n_layers: int = 4, kernel_size: int = 3):
        super().__init__()
        from grn_balladeer.model.tcn_encoder import TCNBlock

        self.input_proj = nn.Conv1d(1, hidden_channels, kernel_size=1)
        self.encoder_blocks = nn.ModuleList([
            TCNBlock(hidden_channels, kernel_size, dilation=2 ** i) for i in range(n_layers)
        ])
        self.hidden_channels = hidden_channels

        # Decoder: mirrors the encoder (same dilation schedule reversed),
        # ending in a 1-channel output to reconstruct the raw signal.
        self.decoder_blocks = nn.ModuleList([
            TCNBlock(hidden_channels, kernel_size, dilation=2 ** i) for i in reversed(range(n_layers))
        ])
        self.output_proj = nn.Conv1d(hidden_channels, 1, kernel_size=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, T) raw per-node time series. Returns (B, N,
        hidden_channels) per-node embeddings -- same shape/convention
        as LightweightTCNEncoder's output, WITHOUT the structural-graph
        aggregation step (this encoder is trained node-independently,
        since reconstruction is a per-channel task; graph aggregation,
        if wanted, can still be applied downstream to these embeddings
        exactly as before)."""
        B, N, T = x.shape
        x = x.reshape(B * N, 1, T)
        x = self.input_proj(x)
        for block in self.encoder_blocks:
            x = block(x)
        embeddings = x.mean(dim=-1)  # (B*N, hidden_channels) -- temporal pooling, matches TCN encoder
        return embeddings.reshape(B, N, self.hidden_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full autoencoder forward pass for pretraining: reconstructs
        x from its own embedding. x: (B, N, T). Returns (B, N, T)
        reconstruction, same shape as input, for an MSE reconstruction
        loss against the original signal."""
        B, N, T = x.shape
        x_flat = x.reshape(B * N, 1, T)
        h = self.input_proj(x_flat)
        for block in self.encoder_blocks:
            h = block(h)
        # h: (B*N, hidden_channels, T) -- NOTE: unlike encode() above,
        # we do NOT temporally pool here, since the decoder needs the
        # full temporal resolution to reconstruct T timepoints, not a
        # single pooled vector. encode() (used at classification time)
        # deliberately pools; forward() (used only during pretraining)
        # deliberately does not.
        for block in self.decoder_blocks:
            h = block(h)
        recon = self.output_proj(h)  # (B*N, 1, T)
        return recon.reshape(B, N, T)


def pretrain_autoencoder(
    autoencoder: EEGAutoencoder,
    X_train_epochs: torch.Tensor,
    n_epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: "torch.device | None" = None,
) -> list:
    """Trains the autoencoder via MSE reconstruction on X_train_epochs
    ONLY (must be training-fold subjects' epochs -- see module
    docstring on leakage discipline). Returns the per-epoch loss
    history (for a sanity-check reconstruction-loss curve, analogous to
    the loss diagnostics used earlier this session for the supervised
    models).
    """
    if device is not None:
        autoencoder = autoencoder.to(device)
        X_train_epochs = X_train_epochs.to(device)

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    n_samples = X_train_epochs.shape[0]
    history = []

    for epoch in range(n_epochs):
        autoencoder.train()
        perm = torch.randperm(n_samples, device=X_train_epochs.device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            batch = X_train_epochs[idx]
            optimizer.zero_grad()
            recon = autoencoder(batch)
            loss = nn.functional.mse_loss(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        history.append(epoch_loss / n_batches)

    return history
