"""
grn_balladeer.connectivity.structural_graph
==============================================
Structural (physical-distance) k-NN adjacency, precomputed ONCE from
real electrode positions (MNE's standard_1020 montage), not recomputed
per-epoch like PLV/PLI. This is the near-zero-inference-cost graph
construction strategy documented in ICCCI2026 (Sec. 3.2: "<1ms in our
measurements... enabling real-time operation on resource-constrained
devices") -- the deliberate trade for a lightweight, edge-deployable
architecture: sacrifice the ability to capture dynamic functional
connectivity (PLV/PLI's real advantage) for near-zero adjacency cost,
since the earlier ablations this session (PLV vs PLI, single- vs
multi-band) found no measurable accuracy cost to simplifying
connectivity on this cohort anyway.
"""

from __future__ import annotations

from typing import List

import numpy as np
import mne


def get_standard_positions(ch_names: List[str]) -> np.ndarray:
    """Fetches real 3D electrode positions for ch_names from MNE's
    standard_1020 montage (10-20/10-10 system). Returns (n_channels, 3).

    Raises KeyError with a clear message if any channel name isn't in
    the standard montage -- fail loudly rather than silently drop or
    guess a position, since a missing/wrong position would silently
    corrupt the whole graph.
    """
    montage = mne.channels.make_standard_montage("standard_1020")
    positions = montage.get_positions()["ch_pos"]
    missing = [ch for ch in ch_names if ch not in positions]
    if missing:
        raise KeyError(
            f"get_standard_positions: {missing} not found in MNE's standard_1020 "
            "montage -- cannot build a structural graph without a real position "
            "for every channel. Check for a naming mismatch (e.g. case, aliases)."
        )
    return np.array([positions[ch] for ch in ch_names])


def build_structural_knn_graph(ch_names: List[str], k: int = 8) -> np.ndarray:
    """Builds a symmetric k-NN adjacency matrix from real electrode
    positions -- computed ONCE (offline, not per-epoch), reused
    identically across every subject and every epoch, since physical
    electrode geometry does not change. This is the near-zero-cost
    connectivity construction strategy from ICCCI2026's structural-graph
    branch of their taxonomy.

    Returns a (n_channels, n_channels) binary adjacency matrix
    (symmetric: an edge exists if either node is among the other's k
    nearest neighbors -- the usual convention to keep k-NN graphs
    undirected). Diagonal is zero (no self-loops here; self-loops, if
    wanted, are added at the graph-conv layer level, not baked into the
    adjacency itself).
    """
    positions = get_standard_positions(ch_names)
    n = len(ch_names)

    dist = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)  # a node is never its own neighbor

    adjacency = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        nearest_k = np.argsort(dist[i])[:k]
        adjacency[i, nearest_k] = 1.0

    # Symmetrize: edge exists if EITHER direction's k-NN relation holds
    adjacency = np.maximum(adjacency, adjacency.T)
    return adjacency


def normalize_adjacency(adjacency: np.ndarray) -> np.ndarray:
    """Symmetric normalization (D^-1/2 A D^-1/2) with added self-loops
    (A + I), the standard GCN propagation rule (Kipf & Welling 2017) --
    used here instead of the magnetic Laplacian's complex-valued
    normalization, since the structural graph carries no phase
    information (unlike PLV-derived complex edge weights) to justify
    the added complex-arithmetic cost.
    """
    n = adjacency.shape[0]
    A_hat = adjacency + np.eye(n, dtype=adjacency.dtype)
    degree = A_hat.sum(axis=1)
    d_inv_sqrt = np.zeros_like(degree)
    np.power(degree, -0.5, out=d_inv_sqrt, where=degree > 0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ A_hat @ D_inv_sqrt
