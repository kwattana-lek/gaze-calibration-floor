#!/usr/bin/env python3
# exp_coherence.py -- does the detectability floor depend on corruption COHERENCE?
"""THE QUESTION

The published floor (c = delta_floor@0.7 / e_hat ~ 1.03, invariant across two
architectures and two adaptation depths) was measured under corruption that
displaces every corrupted frame in an INDEPENDENT random direction. Real
attention lapses are not independent: a disengaged subject rests somewhere, and
several calibration points inherit a similar error.

Coherently displaced frames pull the adaptation toward themselves, shrinking
their own residuals, so a residual scorer should find them harder to flag.

  Prediction: delta_floor@0.7 RISES with coherence kappa.
  If it does, c is invariant across MODELS but not across corruption PROCESSES,
  which is a scope limit on the published constant.

DESIGN. Fully paired: mask and azimuth randomness are keyed without kappa, and
the per-frame azimuth offsets are scaled by (1 - kappa), so every kappa level
uses identical corrupted frames displaced by identical magnitudes -- only the
alignment of the directions changes. kappa = 0 reduces to the published model.

ARM. Frozen UniGaze ViT-H/14 + ridge head on cached features, lambda = 1e-2:
the arm where c = 1.162 was measured. Closed form, so the whole sweep is seconds.

SPEED. The ridge hat matrix depends only on the features and lambda, never on
the labels, so each subject's leave-one-out functionals are precomputed once and
reused across every cell. Verified exact against unigaze_arm.residuals.

WRITES only its own records file.
"""
import argparse, json, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gaze_math as gm                      # noqa: E402
import corruption_temporal as ct            # noqa: E402
from unigaze_arm import K, SUBJ, FEAT_DIR, residuals, roc_auc   # noqa: E402

OUT = os.path.join(_HERE, "coherence_records.json")
RESNET_DIR = os.path.join(_HERE, "resnet_frozen_features")


def load_feats(arm, level, feat_dir=None, subjects=None):
    """Cached 9-frame support features. Both arms store F (9, d) and Y (9, 2);
    UniGaze is flat and 1280-d, the frozen ResNet is split by accuracy level and
    512-d.

    feat_dir/subjects default to the dataset unigaze_arm selected, so MPIIFaceGaze
    behaviour is unchanged and a second corpus needs no edit here."""
    root = feat_dir or FEAT_DIR
    out = {}
    for s in (subjects or SUBJ):
        f = (os.path.join(root, f"{s}.npz") if arm == "unigaze"
             else os.path.join(RESNET_DIR, level, f"{s}.npz"))
        if not os.path.exists(f):
            sys.exit(f"ABORT: no cached features at {f}")
        z = np.load(f)
        out[s] = (z["F"][:K].astype(np.float64), z["Y"][:K].astype(np.float64))
    return out
DELTAS_BASE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20]
DELTAS_EXT = [24, 28, 32, 36, 40, 45, 50]   # for cells censored at the top
KAPPAS = [0.0, 0.25, 0.5, 0.75, 1.0]
TARGET = 0.7


def loo_functionals(F, lam):
    """Row i: the fixed linear map from the other n-1 labels to the LOO prediction
    at i. Depends only on features and lambda, never on the labels.

    unigaze_arm.ridge_fit_predict centres BOTH X and Y on the training fold:

        pred = (x_i - mu) Xc^T (G + lam I)^-1 Yc + ybar

    Writing b = (G + lam I)^-1 (x_i - mu) Xc^T and expanding Yc = Y - ybar,
    with ybar = (1/(n-1)) 1^T Y, the label-centring folds back in as a rank-one
    correction and the whole map is still linear in Y:

        a_i = b + (1 - sum(b)) / (n - 1)

    Dropping that correction is a ~0.19 deg error per frame -- the exactness
    check below exists because an earlier version of this function did."""
    n = len(F)
    out = []
    for i in range(n):
        m = np.arange(n) != i
        Xtr = F[m]
        mu = Xtr.mean(0, keepdims=True)
        Xc = Xtr - mu
        G = Xc @ Xc.T
        k = ((F[i:i + 1] - mu) @ Xc.T)[0]
        b = np.linalg.solve(G + lam * np.eye(n - 1), k)
        out.append((m, b + (1.0 - b.sum()) / (n - 1)))
    return out


def loo_resid(funcs, labels):
    pred = np.stack([a @ labels[m] for m, a in funcs])
    return gm.angular_error_deg(pred, labels)


def crossing(deltas, aucs, target):
    """First delta where AUC crosses target, linear between grid points, and only
    if the crossing is SUSTAINED to the top of the grid (the guard added after
    the high-end truncation artifact)."""
    pairs = sorted(zip(deltas, aucs))
    if pairs[-1][1] < target:
        return None, "censored_hi"
    for (d0, a0), (d1, a1) in zip(pairs, pairs[1:]):
        if a0 < target <= a1 and a1 != a0:
            return d0 + (target - a0) * (d1 - d0) / (a1 - a0), "ok"
    return None, "no_crossing"


DELTAS = DELTAS_BASE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc", type=float, default=0.2)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--arm", choices=("unigaze", "resnet"), default="unigaze")
    ap.add_argument("--level", choices=("L0", "L1", "L2"), default="L0",
                    help="accuracy level; resnet arm only")
    ap.add_argument("--extend", action="store_true",
                    help="append the extended delta grid, for cells whose "
                         "crossing is censored at the top of the base grid")
    a = ap.parse_args()
    global DELTAS
    DELTAS = DELTAS_BASE + (DELTAS_EXT if a.extend else [])

    feats = load_feats(a.arm, a.level)
    funcs = {s: loo_functionals(feats[s][0], a.lam) for s in SUBJ}

    # exactness check against the arm's own implementation
    s0 = SUBJ[0]
    F0, Y0 = feats[s0]
    ref = residuals(F0, Y0, a.lam)[1]
    got = loo_resid(funcs[s0], Y0)
    dev = float(np.max(np.abs(ref - got)))
    print(f"precompute vs unigaze_arm.residuals: max |diff| = {dev:.3e}")
    if dev > 1e-9:
        sys.exit("ABORT: precomputed LOO does not reproduce the arm. Do not trust the sweep.")

    rows = []
    for kap in KAPPAS:
        for d in DELTAS:
            sc, pos, per_subj = [], [], {}
            for s in SUBJ:
                F, Y = feats[s]
                ss, sp = [], []
                for seed in range(a.seeds):
                    c = ct.apply_coherent(Y, s, d, a.pc, seed, kap)
                    r = loo_resid(funcs[s], c["corrupted_labels"])
                    ss.extend(r.tolist()); sp.extend(c["is_corrupted"].tolist())
                sc.extend(ss); pos.extend(sp)
                per_subj[s] = roc_auc(ss, sp)
            rows.append({"kappa": kap, "delta": d,
                         "auc": roc_auc(sc, pos),
                         "n_corrupted": int(np.sum(pos)), "n": len(pos),
                         "per_subject_auc": per_subj})
        print(f"  kappa={kap} done", flush=True)

    arm_name = ("Frozen UniGaze ViT-H/14" if a.arm == "unigaze"
                else f"Frozen ResNet-18 {a.level}")
    print(f"\n{arm_name} + ridge, lambda={a.lam:g}, p_c={a.pc:g}, "
          f"{a.seeds} seeds x {len(SUBJ)} subjects.\n")
    print(f"{'delta':>7}" + "".join(f"{'k='+str(k):>9}" for k in KAPPAS))
    print("-" * (7 + 9 * len(KAPPAS)))
    for d in DELTAS:
        line = f"{d:>7}"
        for k in KAPPAS:
            line += f"{[r for r in rows if r['kappa']==k and r['delta']==d][0]['auc']:>9.3f}"
        print(line)

    print(f"\n{'kappa':>7} {'floor@0.7':>11} {'status':>13} {'vs kappa=0':>12}")
    print("-" * 46)
    base = None
    floors = {}
    for k in KAPPAS:
        g = [r for r in rows if r["kappa"] == k]
        fl, st = crossing([r["delta"] for r in g], [r["auc"] for r in g], TARGET)
        floors[k] = fl
        if k == 0.0:
            base = fl
        rel = "" if (fl is None or base is None) else f"{fl-base:>+12.2f}"
        print(f"{k:>7} {('%.2f' % fl) if fl else '--':>11} {st:>13} {rel}")

    # bootstrap over subjects on the kappa=1 vs kappa=0 floor difference
    rs = np.random.RandomState(0)
    diffs = []
    for _ in range(2000):
        pick = rs.choice(SUBJ, len(SUBJ), True)
        f = {}
        for k in (0.0, 1.0):
            g = [r for r in rows if r["kappa"] == k]
            au = [float(np.mean([r["per_subject_auc"][s] for s in pick])) for r in g]
            f[k] = crossing([r["delta"] for r in g], au, TARGET)[0]
        if f[0.0] and f[1.0]:
            diffs.append(f[1.0] - f[0.0])
    if diffs:
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        print(f"\nfloor(kappa=1) - floor(kappa=0): median {np.median(diffs):+.2f} deg, "
              f"95% CI [{lo:+.2f}, {hi:+.2f}]  ({len(diffs)}/2000 usable)")

    json.dump({"pc": a.pc, "lam": a.lam, "seeds": a.seeds,
               "floors": {str(k): floors[k] for k in KAPPAS}, "rows": rows},
              open(OUT, "w"), indent=2)
    print(f"\nwrote {os.path.basename(OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
