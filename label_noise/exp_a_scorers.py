# exp_a_scorers.py
"""Experiment A — six per-frame corruption scorers (higher score = more likely
corrupted), evaluated identically to the Phase-1 LOO residual.

  1. small_loss   — adapt on all K, per-frame adaptation residual (deg).
  2. dividemix    — 2-component GMM on the K adaptation losses; posterior of the
                    high-loss component. Returns a `failed` flag (a result, §1.2).
  3. coteaching   — two backbones (different seeds), each adapted on the support
                    set; per-frame disagreement (deg) between their predictions.
  4. suge         — feature-space neighbour-projected label vs stated label (deg).
  5. loo          — leave-one-out residual (ours).  [reused from loo_residual]
  6. source       — source-model residual, no adaptation (ours). [reused]

Backbone-agnostic: uses injected adapt_fn / predict_fn / feature_fn (defaulting
to backbone_api). Labels are (pitch,yaw) radians, predictions are unit vectors
(gaze_math), matching loo_residual — no axis bridge.
"""
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gaze_math as gm
import backbone_api as api
import loo_residual as lr

__all__ = ["small_loss_scores", "dividemix_scores", "coteaching_scores",
           "suge_scores", "score_all", "SCORERS"]

SCORERS = ["small_loss", "dividemix", "coteaching", "suge", "loo", "source"]


def _resid_deg(pred_vecs, labels):
    """Per-frame angular error (deg) between predicted vectors and (pitch,yaw) labels."""
    labels = np.asarray(labels, float).reshape(-1, 2)
    return np.array([gm.angular_error_vec_deg(pred_vecs[i], gm.pitchyaw_to_vector(labels[i]))
                     for i in range(len(labels))], float)


# ---------------------------------------------------------------------------
# 1. small-loss: adaptation residual on each support frame after adapting on all K
# ---------------------------------------------------------------------------
def small_loss_scores(base_model, images, labels, steps=api.DEFAULT_STEPS,
                      lr_=api.DEFAULT_LR, adapt_fn=api.adapt, predict_fn=api.predict):
    adapted = adapt_fn(base_model, images, labels, steps=steps, lr=lr_)
    preds = predict_fn(adapted, images)
    return _resid_deg(preds, labels)


# ---------------------------------------------------------------------------
# 2. DivideMix: 2-component 1-D GMM (EM) on the losses; posterior of high-loss comp
#    (dependency-free so we own the failure detection)
# ---------------------------------------------------------------------------
def _gmm2_em(x, iters=100, tol=1e-6):
    x = np.asarray(x, float)
    n = len(x)
    if n < 4 or np.ptp(x) < 1e-9:
        return None, "degenerate_input"
    # init: split at median
    med = np.median(x)
    mu = np.array([x[x <= med].mean(), x[x > med].mean() if (x > med).any() else med + 1e-3])
    var = np.array([max(x.var(), 1e-6)] * 2)
    w = np.array([0.5, 0.5])
    prev = None
    for _ in range(iters):
        # E-step
        g = np.stack([w[k] * np.exp(-0.5 * (x - mu[k]) ** 2 / var[k]) / np.sqrt(2 * np.pi * var[k])
                      for k in range(2)], 1)
        s = g.sum(1, keepdims=True)
        if np.any(s <= 1e-300):
            return None, "underflow"
        r = g / s
        # M-step
        Nk = r.sum(0)
        if np.any(Nk < 1e-6):
            return None, "empty_component"
        w = Nk / n
        mu = (r * x[:, None]).sum(0) / Nk
        var = (r * (x[:, None] - mu[None, :]) ** 2).sum(0) / Nk
        var = np.maximum(var, 1e-6)
        ll = np.log(s).sum()
        if prev is not None and abs(ll - prev) < tol:
            break
        prev = ll
    if abs(mu[0] - mu[1]) < 1e-3 or w.min() < 0.05:
        return None, "collapsed"
    hi = int(np.argmax(mu))                    # high-loss component
    return r[:, hi], "ok"


def dividemix_scores(losses):
    """Returns (scores, failed_bool, reason). `losses` = the K per-frame losses
    (e.g. small_loss residuals). On failure, falls back to the raw loss ranking."""
    losses = np.asarray(losses, float)
    post, reason = _gmm2_em(losses)
    if post is None:
        return losses.copy(), True, reason   # degrade to raw loss; flagged failed
    return post, False, reason


# ---------------------------------------------------------------------------
# 3. co-teaching: disagreement between two backbones each adapted on the support
# ---------------------------------------------------------------------------
def coteaching_scores(model_a, model_b, images, labels, steps=api.DEFAULT_STEPS,
                      lr_=api.DEFAULT_LR, adapt_fn=api.adapt, predict_fn=api.predict):
    a = adapt_fn(model_a, images, labels, steps=steps, lr=lr_)
    b = adapt_fn(model_b, images, labels, steps=steps, lr=lr_)
    pa, pb = predict_fn(a, images), predict_fn(b, images)
    return np.array([gm.angular_error_vec_deg(pa[i], pb[i]) for i in range(len(pa))], float)


# ---------------------------------------------------------------------------
# 4. SUGE-style neighbour consistency: feature-space neighbours' stated labels
#    -> weighted-projected label; score = angle to this frame's stated label.
#    At K=9 the neighbourhood is at most 8 frames (noted in the write-up, §1.4).
# ---------------------------------------------------------------------------
def suge_scores(feats, labels, temp=0.5):
    feats = np.asarray(feats, float)
    labels = np.asarray(labels, float).reshape(-1, 2)
    K = len(labels)
    fn = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    label_vecs = gm.pitchyaw_to_vector(labels)             # (K,3)
    out = np.zeros(K, float)
    for i in range(K):
        sims = fn @ fn[i]                                  # cosine similarity
        sims[i] = -np.inf                                  # exclude self
        w = np.exp((sims - np.nanmax(sims[np.isfinite(sims)])) / temp)
        w[~np.isfinite(w)] = 0.0
        if w.sum() <= 0:
            out[i] = np.nan; continue
        w = w / w.sum()
        proj = (w[:, None] * label_vecs).sum(0)            # weighted neighbour label
        proj = proj / (np.linalg.norm(proj) + 1e-9)
        out[i] = gm.angular_error_vec_deg(proj, label_vecs[i])
    return out


# ---------------------------------------------------------------------------
# Convenience: compute all six scorers for one support set.
# ---------------------------------------------------------------------------
def score_all(base_model, images, labels, model_b=None, feats=None,
              steps=api.DEFAULT_STEPS, lr_=api.DEFAULT_LR,
              adapt_fn=api.adapt, predict_fn=api.predict):
    sl = small_loss_scores(base_model, images, labels, steps, lr_, adapt_fn, predict_fn)
    dm, dm_failed, dm_reason = dividemix_scores(sl)
    src = lr.source_residuals(base_model, images, labels, predict_fn=predict_fn)
    loo = lr.loo_residuals(base_model, images, labels, steps=steps, lr=lr_,
                           adapt_fn=adapt_fn, predict_fn=predict_fn)
    ct = (coteaching_scores(base_model, model_b, images, labels, steps, lr_, adapt_fn, predict_fn)
          if model_b is not None else np.full(len(labels), np.nan))
    sg = suge_scores(feats, labels) if feats is not None else np.full(len(labels), np.nan)
    return {"small_loss": sl, "dividemix": dm, "coteaching": ct, "suge": sg,
            "loo": loo, "source": src, "_dividemix_failed": dm_failed, "_dividemix_reason": dm_reason}
