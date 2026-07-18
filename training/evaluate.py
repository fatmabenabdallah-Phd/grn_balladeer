"""
grn_balladeer.training.evaluate
===================================
Module 9 — evaluate() for the GRN model itself (not the SVM/RF
baselines, which already have their own evaluate()/evaluate_disaggregated()
in eval/baselines.py). Reuses those same metric functions rather than
duplicating accuracy/F1/AUC logic — only the "get predictions out of a
GRN model" part is new here.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from grn_balladeer.eval.baselines import EvalResult, evaluate, evaluate_disaggregated
from grn_balladeer.model.classification_head import ClassificationHead, global_pool, split_real_imag
from grn_balladeer.model.grn_encoder import GRNEncoder


@torch.no_grad()
def evaluate_model(
    encoder: GRNEncoder,
    cls_head: ClassificationHead,
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    labels: np.ndarray,
    sex: Optional[np.ndarray] = None,
    age_bin: Optional[np.ndarray] = None,
) -> dict:
    """Runs the GRN (encoder + classification head only — no gradient,
    no resonance head needed for evaluation-only accuracy/F1/AUC) over
    `batch`, then computes global + disaggregated (sex, age_bin) metrics
    via eval.baselines' EvalResult/evaluate/evaluate_disaggregated.

    Returns {'global': EvalResult, 'by_sex': {...}, 'by_age_bin': {...}}
    (the last two omitted if the corresponding array wasn't provided).
    """
    encoder.eval()
    cls_head.eval()

    probas = []
    for X_i, L_norm_i in batch:
        h_i = encoder(X_i, L_norm_i)
        h_real_i = split_real_imag(h_i)
        pooled_i = global_pool(h_real_i)
        logits_i = cls_head(pooled_i.unsqueeze(0))
        probas.append(torch.softmax(logits_i, dim=-1)[0, 1].item())

    y_proba = np.array(probas)
    y_pred = (y_proba >= 0.5).astype(int)

    results = {"global": evaluate(labels, y_pred, y_proba)}
    if sex is not None:
        results["by_sex"] = evaluate_disaggregated(labels, y_pred, y_proba, sex)
    if age_bin is not None:
        results["by_age_bin"] = evaluate_disaggregated(labels, y_pred, y_proba, age_bin)

    encoder.train()
    cls_head.train()
    return results
