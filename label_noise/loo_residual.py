# loo_residual.py
"""
Label-consistency scoring (HANDOFF §4.3): per support frame, how far is the
frame's label from what the model predicts for it?

Two scores (the two the Phase-1 JSONL records, §5):

  1. LOO angular residual (primary, §4.3.1)
       Adapt on the K-1 other support frames, predict the held-out frame, measure
       the angular error (degrees) between the prediction and the held-out LABEL.
       A corrupted label sits far from the manifold the K-1 frames pull the model
       toward, so its residual is large.

  2. Source-model residual (baseline, §4.3.2)
       Same angular error but with NO adaptation — straight from the loaded
       backbone. Tests whether adaptation is needed at all.

The contamination confound (§4.4): at pc>0 the K-1 adaptation set is itself
partly corrupted, which inflates clean-frame residuals. `loo_residuals_purified`
implements the one-round purify-and-rescore probe.

Scoring is backbone-agnostic: it calls the injected `adapt_fn` / `predict_fn`
(defaulting to backbone_api.adapt / backbone_api.predict), so the real ResNet and
the SyntheticBackbone run through the identical code path.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gaze_math as gm
import backbone_api as api

__all__ = [
    "source_residuals",
    "loo_residuals",
    "loo_residuals_purified",
    "roc_auc",
]


def _index(images, idx):
    """Subset rows of images whether images is a torch tensor or numpy array."""
    if hasattr(images, "shape") and not isinstance(images, np.ndarray):
        return images[idx]          # torch tensor: fancy indexing with a list works
    return np.asarray(images)[idx]


def source_residuals(model, images, labels, predict_fn=api.predict):
    """(K,) angular residuals (deg) with no adaptation (HANDOFF §4.3.2)."""
    labels = np.asarray(labels, dtype=np.float64).reshape(-1, 2)
    k = labels.shape[0]
    preds = predict_fn(model, images)               # (K, 3)
    return np.array([gm.angular_error_vec_deg(preds[i], gm.pitchyaw_to_vector(labels[i]))
                     for i in range(k)], dtype=np.float64)


def loo_residuals(
    base_model, images, labels,
    exclude=None,
    steps=api.DEFAULT_STEPS, lr=api.DEFAULT_LR,
    active_layers=api.DEFAULT_ACTIVE_LAYERS,
    adapt_fn=api.adapt, predict_fn=api.predict,
):
    """
    (K,) LOO angular residuals (deg), primary score (HANDOFF §4.3.1).

    For each frame i: adapt on the other frames (minus any in `exclude`), then
    predict frame i and measure angular error to label_i.

    `exclude` (set of indices) is never used as a support/donor frame in ANY fold
    — the hook the purification variant uses to drop a suspected-bad frame. An
    excluded frame is still scored.
    """
    labels = np.asarray(labels, dtype=np.float64).reshape(-1, 2)
    k = labels.shape[0]
    exclude = set() if exclude is None else set(exclude)
    res = np.full(k, np.nan, dtype=np.float64)

    for i in range(k):
        support = [j for j in range(k) if j != i and j not in exclude]
        if len(support) == 0:
            continue
        adapted = adapt_fn(
            base_model, _index(images, support), labels[support],
            steps=steps, lr=lr, active_layers=active_layers,
        )
        pred = predict_fn(adapted, _index(images, [i]))          # (1, 3)
        res[i] = gm.angular_error_vec_deg(pred[0], gm.pitchyaw_to_vector(labels[i]))
    return res


def loo_residuals_purified(
    base_model, images, labels,
    steps=api.DEFAULT_STEPS, lr=api.DEFAULT_LR,
    active_layers=api.DEFAULT_ACTIVE_LAYERS,
    adapt_fn=api.adapt, predict_fn=api.predict,
):
    """
    One round of purification (HANDOFF §4.4): score with plain LOO, drop the
    single highest-residual frame from every fold's adaptation set, and rescore.

    Returns (rescored_residuals, dropped_index). The dropped frame is still scored
    in the rescored pass (it just no longer contaminates other folds).
    """
    first = loo_residuals(base_model, images, labels, steps=steps, lr=lr,
                          active_layers=active_layers,
                          adapt_fn=adapt_fn, predict_fn=predict_fn)
    if np.all(np.isnan(first)):
        return first, None
    worst = int(np.nanargmax(first))
    rescored = loo_residuals(base_model, images, labels, exclude={worst},
                             steps=steps, lr=lr, active_layers=active_layers,
                             adapt_fn=adapt_fn, predict_fn=predict_fn)
    return rescored, worst


def roc_auc(scores, positives):
    """
    ROC-AUC of `scores` as a detector of the boolean `positives` (higher score =>
    more likely positive). Rank/Mann-Whitney-U formulation, ties handled by mean
    rank. Returns np.nan if only one class is present.
    """
    scores = np.asarray(scores, dtype=np.float64)
    positives = np.asarray(positives, dtype=bool)
    ok = ~np.isnan(scores)
    scores, positives = scores[ok], positives[ok]
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum + 1) / 2.0     # 1-based average rank per unique value
    ranks = avg[inv]
    sum_pos = ranks[positives].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)
