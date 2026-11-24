# corruption.py
"""
Label-corruption simulator for Phase 1 (HANDOFF §4.2).

Two models, both operating on a subject's K support labels carried as
(pitch, yaw) radians arrays (see gaze_math):

  * random_offset (controlled) — rotate a label by EXACTLY delta degrees at a
    random azimuth, via gaze_math.rotate_by_angle. Gives a clean, interpretable
    severity axis. This replaces the earlier per-component Gaussian/uniform
    jitter, whose angular magnitude depended on the operating point.

  * structured_swap (realistic) — replace a frame's label with another frame's
    label drawn from the SAME subject's pool. This is what an attention lapse
    actually produces: the gaze stays on the manifold and is plausible; only the
    pairing is wrong. The realised angular offset per corrupted frame is recorded
    — that distribution anchors the severity axis to reality (§4.5 histogram).

Every corrupted frame's *realised* offset (degrees) is returned, so run_phase1
logs achieved severity, not the requested one.

`apply_corruption` is the driver: it draws the corruption mask at rate pc and
dispatches to a model. The support labels themselves are never mutated (a copy is
returned).
"""
import numpy as np

import gaze_math as gm

MODELS = ("random_offset", "structured_swap")

__all__ = [
    "MODELS",
    "random_offset",
    "structured_swap",
    "draw_corruption_mask",
    "apply_corruption",
]


def draw_corruption_mask(k, pc, rng, min_clean=1):
    """
    Boolean mask (length k) of which support frames are corrupted.

    Bernoulli(pc) per frame (so pc=0.4, K=9 -> ~3-4 corrupted, matching §4.4),
    but constrained to leave at least `min_clean` clean frames — otherwise a fold
    can have no usable label at all and LOO scoring is undefined.
    """
    mask = rng.random_sample(k) < pc
    n_clean = int((~mask).sum())
    if n_clean < min_clean:
        # un-corrupt random frames until min_clean clean remain
        corrupted_idx = np.where(mask)[0]
        rng.shuffle(corrupted_idx)
        need = min_clean - n_clean
        for i in corrupted_idx[:need]:
            mask[i] = False
    return mask


def random_offset(labels, mask, delta_deg, rng):
    """
    Rotate each masked label by exactly `delta_deg` at a random azimuth.

    Returns (new_labels, realised_offset_deg) where realised_offset_deg[i] is the
    achieved angular offset for corrupted frames and 0.0 for clean frames.
    """
    labels = np.asarray(labels, dtype=np.float64).copy()
    k = labels.shape[0]
    offsets = np.zeros(k, dtype=np.float64)
    for i in range(k):
        if mask[i]:
            new_py, realised = gm.rotate_by_angle(labels[i], delta_deg, rng=rng)
            labels[i] = new_py
            offsets[i] = realised
    return labels, offsets


def structured_swap(labels, mask, pool_labels, rng):
    """
    Replace each masked label with a donor label from the same subject's pool.

    Parameters
    ----------
    labels : (K, 2) support labels (pitch, yaw) radians.
    mask : (K,) bool, which frames to corrupt.
    pool_labels : (M, 2) the subject's full frame-label pool (the manifold to
                  draw plausible donors from). Typically the subject's all-frames
                  labels; may include the support labels.
    rng : numpy RandomState.

    Returns (new_labels, realised_offset_deg). The donor is chosen to differ
    materially from the frame's own label (angular offset > ~0.5 deg) so a "swap"
    is not a no-op; if the pool is degenerate, the largest available offset is used.
    """
    labels = np.asarray(labels, dtype=np.float64).copy()
    pool = np.asarray(pool_labels, dtype=np.float64)
    k = labels.shape[0]
    offsets = np.zeros(k, dtype=np.float64)
    for i in range(k):
        if not mask[i]:
            continue
        orig = labels[i].copy()
        cand_off = gm.angular_error_deg(
            np.repeat(orig[None, :], pool.shape[0], axis=0), pool
        )
        cand_off = np.atleast_1d(cand_off)
        usable = np.where(cand_off > 0.5)[0]
        if usable.size == 0:
            j = int(np.argmax(cand_off))          # degenerate pool: best we can do
        else:
            j = int(rng.choice(usable))
        labels[i] = pool[j]
        offsets[i] = float(cand_off[j])
    return labels, offsets


def apply_corruption(labels, model, rng, pc=None, mask=None,
                     delta_deg=None, pool_labels=None, min_clean=1):
    """
    Draw a corruption mask (or use the supplied one) and apply a corruption model.

    Parameters
    ----------
    labels : (K, 2) clean support labels (pitch, yaw) radians.
    model : "random_offset" or "structured_swap".
    rng : numpy RandomState.
    pc : corruption rate; used to draw the mask if `mask` is None.
    mask : optional explicit (K,) bool mask (overrides pc).
    delta_deg : required for "random_offset".
    pool_labels : required for "structured_swap".

    Returns dict:
        corrupted_labels : (K, 2)
        is_corrupted     : (K,) bool
        actual_offset_deg: (K,) float  (0.0 on clean frames)
        model, pc, delta_deg
    """
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}; expected one of {MODELS}.")
    labels = np.asarray(labels, dtype=np.float64)
    k = labels.shape[0]

    if mask is None:
        if pc is None:
            raise ValueError("apply_corruption needs pc or an explicit mask.")
        mask = draw_corruption_mask(k, pc, rng, min_clean=min_clean)
    mask = np.asarray(mask, dtype=bool)

    if model == "random_offset":
        if delta_deg is None:
            raise ValueError("random_offset requires delta_deg.")
        new_labels, offsets = random_offset(labels, mask, delta_deg, rng)
    else:  # structured_swap
        if pool_labels is None:
            raise ValueError("structured_swap requires pool_labels.")
        new_labels, offsets = structured_swap(labels, mask, pool_labels, rng)

    return {
        "corrupted_labels": new_labels,
        "is_corrupted": mask,
        "actual_offset_deg": offsets,
        "model": model,
        "pc": pc,
        "delta_deg": delta_deg,
    }
