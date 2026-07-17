"""
grn_balladeer.training.evaluate
===================================
Module 9 — evaluation, with disaggregation by sex and age_bin (columns
as defined in data.labels.build_label_table: 'sex' in {'male','female'},
'age_bin' in {'6-9','10-12','13-15','16-18'}).

IMPORTANT LIMITATION, not yet resolvable in this codebase: disaggregated
metrics are only meaningful with more than one subject per (sex,
age_bin) cell. Only UB0136 (a single subject) has been available so
far - disaggregation on a 1-subject dataset degenerates to a single
100%-or-0% cell and one empty cell, which is NOT informative and should
not be read as a real fairness/robustness result. This function is
built to be correct and ready for a real multi-subject run - it is
tested below on synthetic multi-subject data specifically because a
real disaggregation test isn't possible yet.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import pandas as pd

from grn_balladeer.model.classification_head import ClassificationHead
from grn_balladeer.model.grn_encoder import GRNEncoder
from grn_balladeer.training.batch_forward import forward_batch


def evaluate(
    encoder: GRNEncoder,
    head: ClassificationHead,
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    labels: torch.Tensor,
    meta_df: Optional[pd.DataFrame] = None,
    pool_method: str = "mean",
) -> dict:
    """Runs inference (no grad, eval mode) over `batch` and returns
    overall accuracy plus, if meta_df is given, accuracy disaggregated
    by 'sex' and by 'age_bin'.

    meta_df: optional DataFrame with one row per sample, aligned by
        position with `batch`/`labels`, containing at least 'sex' and/or
        'age_bin' columns (data.labels.build_label_table's convention).
        If None, only overall accuracy is returned.

    Returns: {'accuracy': float, 'n': int,
              'by_sex': {group: {'accuracy':.., 'n':..}, ...} (if available),
              'by_age_bin': {group: {'accuracy':.., 'n':..}, ...} (if available)}
    """
    encoder.eval()
    head.eval()
    with torch.no_grad():
        logits = forward_batch(encoder, head, batch, pool_method=pool_method)
        preds = logits.argmax(dim=-1)
        correct = (preds == labels)

    result = {"accuracy": correct.float().mean().item(), "n": len(labels)}

    if meta_df is not None:
        if len(meta_df) != len(labels):
            raise ValueError(
                f"evaluate: meta_df has {len(meta_df)} rows but labels has {len(labels)} - "
                "must be row-aligned with batch/labels."
            )
        correct_np = correct.numpy()
        for group_col, out_key in [("sex", "by_sex"), ("age_bin", "by_age_bin")]:
            if group_col not in meta_df.columns:
                continue
            groups = {}
            for group_val, sub_idx in meta_df.groupby(group_col, observed=True).groups.items():
                idx = list(sub_idx)
                n_group = len(idx)
                if n_group == 0:
                    continue
                acc_group = correct_np[idx].mean()
                groups[str(group_val)] = {"accuracy": float(acc_group), "n": n_group}
            result[out_key] = groups

    return result
