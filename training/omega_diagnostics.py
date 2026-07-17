"""
grn_balladeer.training.omega_diagnostics
============================================
Module 9 (used from Week 4 onward as a safeguard) — detects the
degenerate failure mode where omega collapses to near-identical values
across nodes, which trivially satisfies the ratio=1.0 (unison) entry in
CONSONANCE_RATIOS and drives harmonic_loss near zero WITHOUT any real
harmonic structure being learned (confirmed empirically on an untrained
GRNEncoder: std(omega)=0.0004, harmonic_loss=2.28e-05, all frontal
ratios ~1.0 — see Week 4 Notion notes).

This does not replace determine_rule_direction or the harmonic loss
itself — it is a training-time sanity check to log/flag alongside them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OmegaCollapseReport:
    std_omega: float
    is_collapsed: bool
    threshold: float


def check_omega_collapse(omega, threshold: float = 0.01) -> OmegaCollapseReport:
    """omega: (n_nodes,) tensor. Flags collapse if std(omega) < threshold.
    threshold=0.01 is a WORKING DEFAULT (the untrained-head reference
    case measured std=0.0004, well below it) — not yet tuned against a
    trained model's expected healthy std(omega) range, since no training
    run exists yet. Revisit once Module 9's first real training run
    gives a baseline of what a non-degenerate std(omega) looks like."""
    std_val = omega.std().item()
    return OmegaCollapseReport(std_omega=std_val, is_collapsed=std_val < threshold, threshold=threshold)
