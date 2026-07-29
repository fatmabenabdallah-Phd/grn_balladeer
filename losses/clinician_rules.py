"""
grn_balladeer.losses.clinician_rules
========================================
Bridges clinician domain knowledge (expressed in natural language) to
GRN's neurosymbolic loss framework, generalizing the existing
symbolic_implication_loss (a single, fixed frontal-electrode cluster,
hardcoded at development time) to an arbitrary, growing set of
clinician-authored rules, each independently ablatable and citable.

DESIGN PRINCIPLE -- structured schema with mandatory human confirmation,
not free-text parsing: a clinician's natural-language rule is proposed
as a structured JSON schema (electrode groups, expected relation,
direction, rationale), NOT silently converted into a loss term. The
clinician must review and confirm (or correct) this structured
proposal before it is ever used in training. This mirrors the standard
this project already applies to itself: every citation added to
references.bib was independently verified against a primary source
before being trusted, not assumed correct from a first pass. A
silently-accepted mistranslation of a clinical rule is a much
higher-stakes failure mode than a wrong citation, so the same
verify-before-trust discipline applies here, with a human in the loop
instead of a second search.

Pipeline:
  1. propose_rule_from_text(text) -- calls an LLM (Anthropic API) to
     draft a structured rule from the clinician's free-text statement.
     Returns status="PENDING_CLINICIAN_CONFIRMATION" -- NEVER auto-used.
  2. validate_clinician_rule(rule) -- checks the schema is well-formed
     (electrode names real, groups non-empty, ratio set sane) BEFORE
     showing it to the clinician for confirmation, so obviously broken
     proposals never reach the human review step.
  3. confirm_rule(rule) -- the clinician (or a researcher acting as
     their proxy) explicitly flips status to "CONFIRMED" after review.
     Only confirmed rules may be passed to clinician_rule_loss.
  4. clinician_rule_loss(omega, rule, ch_names) -- generalizes
     symbolic_implication_loss's frontal-pair logic to the rule's own
     electrode groups, computed the same way L_symb already is.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class ClinicianRule:
    """A structured, human-reviewable representation of one clinician
    domain-knowledge rule. See module docstring: never used in training
    until status == "CONFIRMED"."""

    rule_id: str
    electrode_group_a: List[str]
    electrode_group_b: List[str]
    direction: str  # "lower_consonance_for_positive_class" or "higher_consonance_for_positive_class"
    rationale_text: str
    expected_ratio_set: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0, 4.0])
    citation: Optional[str] = None
    status: str = "PENDING_CLINICIAN_CONFIRMATION"  # or "CONFIRMED" or "REJECTED"

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "electrode_group_a": self.electrode_group_a,
            "electrode_group_b": self.electrode_group_b,
            "direction": self.direction,
            "rationale_text": self.rationale_text,
            "expected_ratio_set": self.expected_ratio_set,
            "citation": self.citation,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ClinicianRule":
        return cls(**d)


VALID_DIRECTIONS = {"lower_consonance_for_positive_class", "higher_consonance_for_positive_class"}


def validate_clinician_rule(rule: ClinicianRule, valid_channels: List[str]) -> List[str]:
    """Checks a proposed rule for structural problems BEFORE it is shown
    to the clinician for confirmation. Returns a list of human-readable
    problem descriptions (empty list = no problems found). This is a
    syntactic/schema check only -- it cannot verify that the rule is
    clinically correct, only that it is well-formed enough to present
    for review.
    """
    problems = []

    if not rule.electrode_group_a:
        problems.append("electrode_group_a is empty.")
    if not rule.electrode_group_b:
        problems.append("electrode_group_b is empty.")

    unknown_a = [ch for ch in rule.electrode_group_a if ch not in valid_channels]
    if unknown_a:
        problems.append(f"electrode_group_a contains unknown channel(s): {unknown_a}")
    unknown_b = [ch for ch in rule.electrode_group_b if ch not in valid_channels]
    if unknown_b:
        problems.append(f"electrode_group_b contains unknown channel(s): {unknown_b}")

    overlap = set(rule.electrode_group_a) & set(rule.electrode_group_b)
    if overlap:
        problems.append(
            f"electrode_group_a and electrode_group_b overlap: {sorted(overlap)} -- "
            f"a channel cannot be in both groups of the same rule."
        )

    if rule.direction not in VALID_DIRECTIONS:
        problems.append(
            f"direction '{rule.direction}' not recognized -- must be one of {VALID_DIRECTIONS}."
        )

    if not rule.expected_ratio_set or any(r <= 0 for r in rule.expected_ratio_set):
        problems.append("expected_ratio_set must be a non-empty list of positive numbers.")

    if not rule.rationale_text.strip():
        problems.append(
            "rationale_text is empty -- every rule must carry a stated clinical "
            "justification, even a brief one, for the same reason every citation "
            "in this project's bibliography carries a verification note."
        )

    return problems


CLINICIAN_RULE_SYSTEM_PROMPT = """You translate a clinician's free-text \
statement about expected EEG connectivity patterns in ADHD into a \
structured JSON rule. You do NOT decide whether the rule is correct; \
you only translate it faithfully into the schema below. If the \
statement is ambiguous, prefer a narrower, more literal interpretation \
over a broader guess, and note the ambiguity in "rationale_text".

Respond with ONLY a JSON object (no markdown fences, no preamble), matching:
{
  "electrode_group_a": ["<10-20 channel names>"],
  "electrode_group_b": ["<10-20 channel names>"],
  "direction": "lower_consonance_for_positive_class" | "higher_consonance_for_positive_class",
  "rationale_text": "<brief paraphrase of the clinician's stated reasoning>",
  "expected_ratio_set": [1, 2, 3, 4]
}

Valid channel names (this project's 30-channel CGX montage): {valid_channels}

Clinician's statement: "{clinician_text}"
"""


def propose_rule_from_text(
    clinician_text: str,
    valid_channels: List[str],
    rule_id: str,
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-6",
) -> ClinicianRule:
    """Calls the Anthropic API to draft a structured ClinicianRule from
    a clinician's free-text statement. The returned rule ALWAYS has
    status="PENDING_CLINICIAN_CONFIRMATION" -- this function proposes,
    it never confirms. The clinician (or a researcher relaying their
    review) must call confirm_rule() explicitly before the rule can be
    used in clinician_rule_loss.

    Requires an Anthropic API key, either passed directly or read from
    the ANTHROPIC_API_KEY environment variable.
    """
    import requests

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "propose_rule_from_text: no Anthropic API key found. Pass api_key= "
            "or set the ANTHROPIC_API_KEY environment variable."
        )

    prompt = CLINICIAN_RULE_SYSTEM_PROMPT.format(
        valid_channels=", ".join(valid_channels),
        clinician_text=clinician_text,
    )

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    response.raise_for_status()
    raw_text = response.json()["content"][0]["text"].strip()
    parsed = json.loads(raw_text)

    return ClinicianRule(
        rule_id=rule_id,
        electrode_group_a=parsed["electrode_group_a"],
        electrode_group_b=parsed["electrode_group_b"],
        direction=parsed["direction"],
        rationale_text=parsed["rationale_text"] + " [MACHINE-DRAFTED FROM: \"" + clinician_text + "\" -- NOT YET CLINICIAN-CONFIRMED]",
        expected_ratio_set=[float(r) for r in parsed.get("expected_ratio_set", [1, 2, 3, 4])],
        citation=None,
        status="PENDING_CLINICIAN_CONFIRMATION",
    )


def confirm_rule(rule: ClinicianRule, valid_channels: List[str]) -> ClinicianRule:
    """Explicitly marks a rule CONFIRMED after human review. Re-validates
    the schema one more time (defense in depth: a rule should never
    reach CONFIRMED status if it fails basic structural checks, even if
    a caller skipped validate_clinician_rule earlier)."""
    problems = validate_clinician_rule(rule, valid_channels)
    if problems:
        raise ValueError(
            f"confirm_rule: cannot confirm rule '{rule.rule_id}', it still has "
            f"unresolved problems: {problems}"
        )
    rule.status = "CONFIRMED"
    return rule


def clinician_rule_loss(
    omega: torch.Tensor,
    rule: ClinicianRule,
    ch_names: List[str],
    positive_class_confidence: torch.Tensor,
) -> torch.Tensor:
    """Computes a loss term for one CONFIRMED clinician rule, generalizing
    symbolic_implication_loss's frontal-cluster logic to the rule's own
    electrode_group_a x electrode_group_b cross-product of pairs.

    omega: (n_nodes,) resonance frequencies from GRNEncoder's resonance head.
    rule: a ClinicianRule with status=="CONFIRMED" (raises otherwise).
    ch_names: the ordered channel list matching omega's node ordering.
    positive_class_confidence: scalar tensor, this epoch's predicted
    probability of the positive (ADHD) class -- same role "confidence_i"
    plays in the existing symbolic_implication_loss.
    """
    if rule.status != "CONFIRMED":
        raise ValueError(
            f"clinician_rule_loss: rule '{rule.rule_id}' has status "
            f"'{rule.status}', not 'CONFIRMED' -- refusing to use an "
            f"unconfirmed rule in training."
        )

    from grn_balladeer.losses.harmonic_loss import compute_consonance_degree
    from grn_balladeer.losses.symbolic_loss import symbolic_implication_loss

    idx_a = [ch_names.index(ch) for ch in rule.electrode_group_a]
    idx_b = [ch_names.index(ch) for ch in rule.electrode_group_b]

    pairs_i, pairs_j = [], []
    for a in idx_a:
        for b in idx_b:
            pairs_i.append(a)
            pairs_j.append(b)
    pairs_i = torch.tensor(pairs_i, dtype=torch.long, device=omega.device)
    pairs_j = torch.tensor(pairs_j, dtype=torch.long, device=omega.device)

    omega_a, omega_b = omega[pairs_i], omega[pairs_j]
    mu_ab = compute_consonance_degree(omega_a, omega_b)

    # direction: for "lower_consonance_for_positive_class", the implication
    # is (positive_class -> low consonance), matching symbolic_implication_loss's
    # "direct" mode; for the opposite direction, we invert mu before passing
    # it in the same way, keeping the underlying fuzzy-implication mechanics
    # (and its literature grounding) unchanged rather than duplicating them.
    if rule.direction == "lower_consonance_for_positive_class":
        return symbolic_implication_loss(mu_ab, positive_class_confidence, direction="direct")
    else:
        return symbolic_implication_loss(1.0 - mu_ab, positive_class_confidence, direction="direct")
