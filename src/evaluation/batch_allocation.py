"""Count top-K selections in a replicated batch without reading outcomes.

Research design: docs/cas_q2/CONFIRMATION_DESIGN.md. This arithmetic kernel
does not sample questions, fit a model, calculate intervals or authorize a run.
"""
from __future__ import annotations

import numpy as np


def weighted_top_k(order: np.ndarray, row_weights: np.ndarray, cap: int) -> np.ndarray:
    """Return selected copy counts per original row.

    ``order`` lists unique eligible row indices in frozen score/canonical-tie
    order. ``row_weights`` contains nonnegative integer resampling counts for
    every row, including ineligible ones. The caller groups question siblings
    and derives ``cap`` from the entire batch, not just the eligible rows.
    """
    weights = np.asarray(row_weights)
    indices = np.asarray(order)
    if weights.ndim != 1 or weights.dtype.kind not in "iu":
        raise ValueError("row weights must be a one-dimensional integer array")
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise ValueError("order must be a one-dimensional integer array")
    if type(cap) is not int or cap < 0:
        raise ValueError("cap must be a nonnegative Python integer")
    limit = np.iinfo(np.int64).max
    if cap > limit or np.any(weights < 0):
        raise ValueError("negative weight or unsupported cap")
    if weights.size and int(weights.max()) > limit // weights.size:
        raise ValueError("weights may overflow integer accumulation")
    if indices.size and (np.any(indices < 0) or np.any(indices >= weights.size)):
        raise ValueError("order contains an out-of-range row")
    if np.unique(indices).size != indices.size:
        raise ValueError("order contains a duplicated row index")
    weights = weights.astype(np.int64, copy=False)
    indices = indices.astype(np.int64, copy=False)
    selected = np.zeros(weights.size, dtype=np.int64)
    ordered_weights = weights[indices]
    before = np.cumsum(ordered_weights) - ordered_weights
    selected[indices] = np.minimum(ordered_weights, np.maximum(cap - before, 0))
    return selected
