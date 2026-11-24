#!/usr/bin/env python3
# corruption_temporal.py -- corruption models with COHERENT displacement.
"""WHY THIS EXISTS

Every corruption model in corruption.py displaces each corrupted frame
INDEPENDENTLY: random_offset draws a fresh azimuth per frame, structured_swap
draws a fresh donor per frame. Experiment B's d_coord replaces a label with the
next target's, which is a specific misalignment scored for DAMAGE, not a shared
displacement direction, and the detectability floor has never been measured
under anything but independent isotropic displacement.

Real attention lapses are not independent. A subject who disengages does not
look in a fresh random direction at each calibration point -- they rest
somewhere, and several points in a row inherit a SIMILAR error. That matters for
detection in a specific way: coherently displaced frames pull the adaptation
TOWARD themselves, shrinking their own residuals, so a residual scorer should
find them harder to flag. The prediction is that coherence RAISES the floor,
which would mean c is constant across models but not across corruption
processes.

THE COHERENCE PARAMETER

    azimuth_i = theta_common + (1 - kappa) * eps_i,   eps_i ~ U(-pi, pi)

  kappa = 0  -> azimuths uniform and independent == corruption.random_offset
  kappa = 1  -> every corrupted frame displaced in the SAME direction
  in between -> partial coherence

theta_common and the eps_i are drawn from a kappa-INDEPENDENT stream, so a sweep
over kappa reuses identical masks and identical random numbers and differs only
in coherence. The comparison is paired by construction.

Magnitude is still exactly delta_deg per frame (gaze_math.rotate_by_angle with
an explicit azimuth), so the severity axis is unchanged and floors are directly
comparable to the published ones.

corruption.py is untouched; nothing here changes any existing result.
"""
import hashlib

import numpy as np

import gaze_math as gm
from corruption import draw_corruption_mask

__all__ = ["rng_for", "coherent_offset", "cluster_mask", "apply_coherent"]


def rng_for(*parts):
    """SHA-256 keyed RandomState -- reproducible across processes and machines,
    matching the convention the rest of the project uses (never Python's
    per-process-salted hash)."""
    key = "|".join(str(p) for p in parts).encode()
    return np.random.RandomState(
        int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2 ** 32))


def coherent_offset(labels, mask, delta_deg, kappa, theta_common, eps):
    """Rotate each masked label by exactly delta_deg, at a partially shared azimuth.

    `theta_common` (scalar) and `eps` (length-K) must be drawn by the caller from
    a kappa-independent stream so that a kappa sweep stays paired.
    """
    labels = np.asarray(labels, dtype=np.float64).copy()
    offsets = np.zeros(len(labels), dtype=np.float64)
    for i in range(len(labels)):
        if not mask[i]:
            continue
        az = float(theta_common + (1.0 - kappa) * eps[i]) % (2.0 * np.pi)
        new_py, realised = gm.rotate_by_angle(labels[i], delta_deg, azimuth_rad=az)
        labels[i] = new_py
        offsets[i] = realised
    return labels, offsets


def cluster_mask(labels, n_corrupt, rng):
    """Re-place `n_corrupt` corrupted frames as a LOCAL CLUSTER on the gaze
    manifold: a random seed frame plus its nearest neighbours in label space.

    Index contiguity would be meaningless here -- a ridge fit sees nine
    unordered points, so permuting the support set cannot change any residual.
    What a real lapse actually does is span CONSECUTIVE calibration targets,
    which sit near each other on screen, so it removes a contiguous REGION of
    the calibration manifold rather than scattered points. That the model does
    see, because features track gaze direction.
    """
    labels = np.asarray(labels, dtype=np.float64)
    k = len(labels)
    i = int(rng.randint(k))
    d = gm.angular_error_deg(np.repeat(labels[i][None, :], k, axis=0), labels)
    mask = np.zeros(k, dtype=bool)
    mask[np.argsort(np.atleast_1d(d))[:n_corrupt]] = True
    return mask


def apply_coherent(labels, sid, delta_deg, pc, seed, kappa,
                   locality="scattered", min_clean=1):
    """Driver. Mask and azimuth randomness are keyed WITHOUT kappa, so every
    kappa level sees the same corrupted frames displaced by the same magnitude
    in progressively more aligned directions."""
    labels = np.asarray(labels, dtype=np.float64)
    k = len(labels)
    mask = draw_corruption_mask(
        k, pc, rng_for(sid, "mask", delta_deg, pc, seed), min_clean=min_clean)
    if locality == "clustered":
        # same NUMBER of corrupted frames, re-placed as a local cluster, so the
        # two localities are paired on corruption count.
        mask = cluster_mask(labels, int(mask.sum()),
                            rng_for(sid, "clust", delta_deg, pc, seed))
    elif locality != "scattered":
        raise ValueError(f"locality must be scattered|clustered, got {locality!r}")
    r = rng_for(sid, "azim", delta_deg, pc, seed)
    theta_common = r.uniform(0.0, 2.0 * np.pi)
    eps = r.uniform(-np.pi, np.pi, size=k)
    new_labels, offsets = coherent_offset(
        labels, mask, delta_deg, kappa, theta_common, eps)
    return {"corrupted_labels": new_labels,
            "is_corrupted": mask,
            "actual_offset_deg": offsets,
            "kappa": float(kappa),
            "locality": locality,
            "delta_deg": float(delta_deg),
            "pc": float(pc)}
