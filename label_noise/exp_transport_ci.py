#!/usr/bin/env python3
# exp_transport_ci.py -- subject-level bootstrap CIs for the transport comparison.
"""WHY THIS EXISTS

exp_transport.py reports ten point estimates and no uncertainty. The headline --
"c transports, ratio 0.91x..1.03x" -- is not interpretable without knowing how
wide those intervals are: a ratio band of 0.91-1.03 means one thing if each c is
+-0.02 and nothing at all if each is +-0.3. A reviewer will ask, and should.

The resampling unit is the SUBJECT, not the frame. Frames within a subject share
a support set, a head pose distribution and a person, so resampling frames would
treat 9 correlated residuals as 9 independent ones and produce intervals that are
far too narrow. Subjects are the level at which the design replicates.

For RT-GENE the unit is the PARTICIPANT, not the session: s014 and s015 repeat
s002 and s005 (Zenodo doi 10.5281/zenodo.2529036), so the 16-session list has two
people who would be drawn as if independent. Use the 14-participant list; the
script warns if a repeat session is passed.

Both the floor and ehat are recomputed inside each replicate from the SAME
resampled subjects, because c is their ratio and holding ehat fixed while
resampling the floor would understate the spread of c.

The expensive part -- per-subject LOO residuals under corruption -- is computed
once and reused across replicates. Only pooling, ROC and the crossing are redone,
which is what makes B=2000 affordable.

Usage:
  python3 exp_transport_ci.py --subj-b s000,...,s016 --feat-b rtgene_unigaze_features
"""
import argparse, json, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import corruption_temporal as ct          # noqa: E402
import exp_coherence as E                 # noqa: E402
from exp_transport import load, LAM, PC, SEEDS, GRID   # noqa: E402

try:
    from prepare_rtgene_labels import REPEAT_SESSIONS
except Exception:
    REPEAT_SESSIONS = {}

OUT = os.path.join(_HERE, "transport_ci_summary.json")


def auc_fast(scores, pos):
    """Rank-based ROC-AUC. Same statistic as unigaze_arm.roc_auc, kept local so
    the bootstrap inner loop has no attribute lookups across modules."""
    ok = ~np.isnan(scores)
    s, p = scores[ok], pos[ok]
    a, b = int(p.sum()), int(p.size - p.sum())
    if a == 0 or b == 0:
        return np.nan
    # Vectorised mid-rank, identical to unigaze_arm.roc_auc. The obvious
    # explicit tie loop is ~20x slower and dominates a B=2000 bootstrap.
    u, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    cs = np.cumsum(cnt)
    r = ((cs - cnt + cs + 1) / 2.0)[inv]
    return float((r[p].sum() - a * (a + 1) / 2.0) / (a * b))


def precompute(feats, subj):
    """Per-subject clean residuals, and corrupted residuals for every (kappa, delta)."""
    funcs = {s: E.loo_functionals(feats[s][0], LAM) for s in subj}
    clean = {s: np.asarray(E.loo_resid(funcs[s], feats[s][1]), float) for s in subj}
    cells = {}
    for kap in E.KAPPAS:
        for d in GRID:
            sc, lb = {}, {}
            for s in subj:
                v, p = [], []
                for sd in range(SEEDS):
                    c = ct.apply_coherent(feats[s][1], s, d, PC, sd, kap)
                    v.extend(np.atleast_1d(
                        E.loo_resid(funcs[s], c["corrupted_labels"])).tolist())
                    p.extend(c["is_corrupted"].tolist())
                sc[s], lb[s] = np.asarray(v, float), np.asarray(p, bool)
            cells[(kap, d)] = (sc, lb)
    return clean, cells


def one_replicate(draw, clean, cells):
    e = float(np.mean([clean[s].mean() for s in draw]))
    out = {}
    for kap in E.KAPPAS:
        aucs = []
        for d in GRID:
            sc, lb = cells[(kap, d)]
            aucs.append(auc_fast(np.concatenate([sc[s] for s in draw]),
                                 np.concatenate([lb[s] for s in draw])))
        fl = E.crossing(GRID, aucs, 0.7)[0]
        out[kap] = (fl, (fl / e if fl else None))
    return e, out


def pct(v, lo=2.5, hi=97.5):
    v = [x for x in v if x is not None and np.isfinite(x)]
    if len(v) < 2:
        return None
    return [float(np.percentile(v, lo)), float(np.percentile(v, hi))]


def run_arm(feat_dir, subj, label, B, rng):
    dup = [s for s in subj if s in REPEAT_SESSIONS]
    if dup:
        print(f"  WARNING [{label}]: {', '.join(dup)} repeat "
              f"{', '.join(REPEAT_SESSIONS[s] for s in dup)} -- the bootstrap will "
              f"draw the same person as two independent subjects.", file=sys.stderr)
    feats = load(feat_dir, subj)
    clean, cells = precompute(feats, subj)
    e0, pt = one_replicate(subj, clean, cells)
    reps = []
    n = len(subj)
    for _ in range(B):
        draw = [subj[i] for i in rng.integers(0, n, n)]
        reps.append(one_replicate(draw, clean, cells))
    print(f"\n{label}: {n} subjects, clean-fit e_hat {e0:.2f} deg, B={B}")
    print(f"  {'kappa':>7} {'floor':>8} {'c':>8}   {'95% CI on c':>18}")
    res = {}
    for kap in E.KAPPAS:
        cs = [r[1][kap][1] for r in reps]
        ci = pct(cs)
        fl, c = pt[kap]
        print(f"  {kap:>7} {(f'{fl:.2f}' if fl else '--'):>8} "
              f"{(f'{c:.3f}' if c else '--'):>8}   "
              f"{(f'[{ci[0]:.3f}, {ci[1]:.3f}]' if ci else '--'):>18}")
        res[str(kap)] = {"floor": fl, "c": c, "ci95": ci,
                         "n_valid": int(sum(x is not None for x in cs))}
    return {"label": label, "n": n, "e_clean": e0, "subjects": subj, "B": B,
            "by_kappa": res}, reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-a", default="unigaze_features")
    ap.add_argument("--subj-a", default=",".join(f"p{i:02d}" for i in range(15)))
    ap.add_argument("--feat-b", default="rtgene_unigaze_features")
    ap.add_argument("--subj-b", required=True)
    ap.add_argument("--label-a", default="MPIIFaceGaze")
    ap.add_argument("--label-b", default="RT-GENE")
    ap.add_argument("-B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    A, repsA = run_arm(a.feat_a, [s for s in a.subj_a.split(",") if s], a.label_a, a.B, rng)
    B_, repsB = run_arm(a.feat_b, [s for s in a.subj_b.split(",") if s], a.label_b, a.B, rng)

    print(f"\n  {'kappa':>7} {'ratio':>8}   {'95% CI on ratio':>20}  {'excludes 1?':>12}")
    print("  " + "-" * 54)
    ratios = {}
    for kap in E.KAPPAS:
        ca = A["by_kappa"][str(kap)]["c"]
        cb = B_["by_kappa"][str(kap)]["c"]
        pt = cb / ca if (ca and cb) else None
        # The two corpora have disjoint subjects, so replicates are independent
        # and are paired only by index -- that is a valid draw from the product
        # of the two sampling distributions, not a paired design.
        rs = [rb[1][kap][1] / ra[1][kap][1]
              for ra, rb in zip(repsA, repsB)
              if ra[1][kap][1] and rb[1][kap][1]]
        ci = pct(rs)
        flag = "" if not ci else ("YES" if (ci[0] > 1 or ci[1] < 1) else "no")
        print(f"  {kap:>7} {(f'{pt:.3f}' if pt else '--'):>8}   "
              f"{(f'[{ci[0]:.3f}, {ci[1]:.3f}]' if ci else '--'):>20}  {flag:>12}")
        ratios[str(kap)] = {"ratio": pt, "ci95": ci, "excludes_1": flag == "YES"}

    # ---- within-corpus kappa contrast -------------------------------------
    # The paper's central claim is that coherence moves the floor, i.e. that
    # c(kappa=1) > c(kappa=0) WITHIN a corpus. Comparing the two marginal CIs by
    # eye is the wrong test and an underpowered one: the same subjects enter both
    # endpoints, so the contrast must be formed INSIDE each replicate, where the
    # subject draw cancels. This is paired and is much tighter than the overlap
    # of the marginals suggests.
    k0, k1 = E.KAPPAS[0], E.KAPPAS[-1]
    print(f"\n  --- within-corpus contrast c(kappa={k1}) - c(kappa={k0}), paired ---")
    print(f"  {'corpus':>28} {'delta c':>9}   {'95% CI':>20}  {'excludes 0?':>12}")
    print("  " + "-" * 74)
    contrasts = {}
    for arm_res, reps in ((A, repsA), (B_, repsB)):
        d = [r[1][k1][1] - r[1][k0][1] for r in reps
             if r[1][k1][1] and r[1][k0][1]]
        ci = pct(d)
        ptd = (arm_res["by_kappa"][str(k1)]["c"] or 0) - (arm_res["by_kappa"][str(k0)]["c"] or 0)
        flag = "" if not ci else ("YES" if (ci[0] > 0 or ci[1] < 0) else "no")
        print(f"  {arm_res['label']:>28} {ptd:>9.3f}   "
              f"{(f'[{ci[0]:.3f}, {ci[1]:.3f}]' if ci else '--'):>20}  {flag:>12}")
        contrasts[arm_res["label"]] = {"delta_c": ptd, "ci95": ci,
                                       "excludes_0": flag == "YES"}

    print("\n  Reading: a CI that CONTAINS 1 at every kappa is the transport result --")
    print("  no detectable difference in c between corpora. A CI that excludes 1 at")
    print("  some kappa is a real difference and must be reported as one, however")
    print("  small. Width matters as much as coverage: intervals this is run on are")
    print("  subject-level, so they are wide by construction and honestly so.")

    json.dump({"arm_a": A, "arm_b": B_, "ratio": ratios,
               "kappa_contrast": contrasts}, open(a.out, "w"), indent=2)
    print(f"\nwrote {os.path.basename(a.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
