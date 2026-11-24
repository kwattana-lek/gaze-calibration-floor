#!/usr/bin/env python3
# exp_interaction_ci.py -- does the coherence x locality INTERACTION transport?
"""WHY THIS EXISTS

exp_transport.py answered whether c(kappa) transports, but it runs at
locality="scattered" (apply_coherent's default, never overridden). So what it
established is that COHERENCE ALONE transports. The paper's headline is not that:
it is the INTERACTION -- locality alone is a flat null, coherence alone moves the
floor moderately, and together they more than double it. That is the claim that
makes the paper about attention lapses rather than about coherence, and it had
not been tested on a second corpus.

This runs the full 2x2xkappa design on both corpora from one code path, rather
than splicing exp_transport's scattered row against exp_prior_detector's
clustered row -- those agree numerically on MPIIFaceGaze, but a manuscript table
should not be assembled from two scripts' incidental outputs.

THE PAIRING. The interaction is a within-subject contrast: the same nine support
features are corrupted two ways. So each bootstrap replicate draws ONE subject
sample and evaluates every (locality, kappa) cell on it. Resampling the two
localities independently would inflate the interaction's variance by treating a
paired comparison as unpaired, and would be wrong in the conservative direction
rather than the honest one.

Statistics reported, each with a percentile CI over subjects:
  main effect of coherence  floor(S,k=1) - floor(S,k=0)
  main effect of locality   floor(C,k=0) - floor(S,k=0)
  interaction (additive)    [floor(C,k1)-floor(C,k0)] - [floor(S,k1)-floor(S,k0)]
  lapse ratio               floor(C,k=1) / floor(S,k=0)   <- the abstract's 2.07x

Usage:
  python3 exp_interaction_ci.py --subj-b s000,... --feat-b rtgene_unigaze_features
"""
import argparse, json, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import corruption_temporal as ct                    # noqa: E402
import exp_coherence as E                           # noqa: E402
from exp_transport import load, PC, SEEDS, GRID     # noqa: E402
from exp_transport_ci import auc_fast, pct          # noqa: E402

try:
    from prepare_rtgene_labels import REPEAT_SESSIONS
except Exception:
    REPEAT_SESSIONS = {}

LOCS = ("scattered", "clustered")
OUT = os.path.join(_HERE, "interaction_ci_summary.json")


def precompute(feats, subj, lam):
    funcs = {s: E.loo_functionals(feats[s][0], lam) for s in subj}
    cells = {}
    for loc in LOCS:
        for kap in E.KAPPAS:
            for d in GRID:
                sc, lb = {}, {}
                for s in subj:
                    v, p = [], []
                    for sd in range(SEEDS):
                        c = ct.apply_coherent(feats[s][1], s, d, PC, sd, kap,
                                              locality=loc)
                        v.extend(np.atleast_1d(
                            E.loo_resid(funcs[s], c["corrupted_labels"])).tolist())
                        p.extend(c["is_corrupted"].tolist())
                    sc[s], lb[s] = np.asarray(v, float), np.asarray(p, bool)
                cells[(loc, kap, d)] = (sc, lb)
    return cells


def floors_for(draw, cells):
    """All (locality, kappa) floors on ONE subject draw -- the pairing."""
    out = {}
    for loc in LOCS:
        for kap in E.KAPPAS:
            aucs = []
            for d in GRID:
                sc, lb = cells[(loc, kap, d)]
                aucs.append(auc_fast(np.concatenate([sc[s] for s in draw]),
                                     np.concatenate([lb[s] for s in draw])))
            out[(loc, kap)] = E.crossing(GRID, aucs, 0.7)[0]
    return out


def stats_from(f):
    k0, k1 = E.KAPPAS[0], E.KAPPAS[-1]
    S0, S1 = f[("scattered", k0)], f[("scattered", k1)]
    C0, C1 = f[("clustered", k0)], f[("clustered", k1)]
    if not all((S0, S1, C0, C1)):
        return {}
    return {"coherence_main": S1 - S0,
            "locality_main": C0 - S0,
            "interaction": (C1 - C0) - (S1 - S0),
            "lapse_ratio": C1 / S0}


def run(feat_dir, subj, label, lam, B, rng):
    dup = [s for s in subj if s in REPEAT_SESSIONS and REPEAT_SESSIONS[s] in subj]
    if dup:
        sys.exit("ABORT: " + ", ".join(f"{s} repeats {REPEAT_SESSIONS[s]}" for s in dup)
                 + " -- pass the unique-participant list.")
    cells = precompute(load(feat_dir, subj), subj, lam)
    pt = floors_for(subj, cells)
    reps = [floors_for([subj[i] for i in rng.integers(0, len(subj), len(subj))], cells)
            for _ in range(B)]

    print(f"\n{label}: {len(subj)} subjects, lambda={lam:g}, B={B}")
    print(f"  {'locality':>10}" + "".join(f"{'k='+str(k):>9}" for k in E.KAPPAS))
    print("  " + "-" * (10 + 9 * len(E.KAPPAS)))
    for loc in LOCS:
        print(f"  {loc:>10}" + "".join(
            f"{pt[(loc,k)]:>9.2f}" if pt[(loc, k)] else f"{'none':>9}"
            for k in E.KAPPAS))
    print(f"  {'difference':>10}" + "".join(
        f"{pt[('clustered',k)]-pt[('scattered',k)]:>+9.2f}"
        if (pt[('clustered', k)] and pt[('scattered', k)]) else f"{'--':>9}"
        for k in E.KAPPAS))

    p0 = stats_from(pt)
    res = {}
    print(f"\n  {'statistic':>16} {'value':>9}   {'95% CI':>20}  {'excl. null?':>12}")
    print("  " + "-" * 62)
    for name, null in (("coherence_main", 0.0), ("locality_main", 0.0),
                       ("interaction", 0.0), ("lapse_ratio", 1.0)):
        vals = [s[name] for s in (stats_from(r) for r in reps) if name in s]
        ci = pct(vals)
        v = p0.get(name)
        flag = "" if not ci else ("YES" if (ci[0] > null or ci[1] < null) else "no")
        med = float(np.median(vals)) if vals else None
        frac = len(vals) / max(1, len(reps))
        warn = "" if frac > 0.98 else f"  <-- only {frac:.0%} of replicates valid"
        print(f"  {name:>16} {(f'{v:.3f}' if v is not None else '--'):>9}   "
              f"{(f'[{ci[0]:.3f}, {ci[1]:.3f}]' if ci else '--'):>20}  {flag:>12}{warn}")
        res[name] = {"point": v, "ci95": ci, "boot_median": med,
                     "null": null, "excludes_null": flag == "YES",
                     "n_valid": len(vals)}
    censored = {f"{l}|{k}": sum(1 for r in reps if not r[(l, k)]) for l in LOCS
                for k in E.KAPPAS}
    bad = {k: v for k, v in censored.items() if v > 0.02 * B}
    if bad:
        print(f"\n  CENSORING: replicates with no 0.7 crossing, by cell "
              f"(grid top {max(GRID):g}):")
        for k, v in sorted(bad.items(), key=lambda x: -x[1]):
            print(f"    {k:>20}  {v}/{B}  ({v/B:.0%})")
        print("  A cell censored in a meaningful share of resamples makes every CI "
              "built\n  on it a SELECTED subset -- the replicates that survived are "
              "the ones whose\n  floor happened to fit under the grid top. Extend the "
              "grid with --extra-deltas\n  and rerun BOTH corpora before quoting "
              "anything from this table.")
    return {"label": label, "n": len(subj), "lam": lam, "B": B, "subjects": subj,
            "grid_top": max(GRID), "censored_replicates": censored,
            "floors": {f"{l}|{k}": pt[(l, k)] for l in LOCS for k in E.KAPPAS},
            "stats": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-a", default="unigaze_features")
    ap.add_argument("--subj-a", default=",".join(f"p{i:02d}" for i in range(15)))
    ap.add_argument("--feat-b", default="rtgene_unigaze_features")
    ap.add_argument("--subj-b", required=True)
    ap.add_argument("--label-a", default="MPIIFaceGaze")
    ap.add_argument("--label-b", default="RT-GENE")
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("-B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--extra-deltas", default="",
                    help="comma-separated deltas appended to the grid. Needed when a "
                         "floor lands near the grid top: the crossing is then CENSORED "
                         "in resamples that push it past the top, and those replicates "
                         "drop out, so the CI is computed on a subset selected by the "
                         "outcome. Both corpora must use the SAME grid.")
    a = ap.parse_args()

    global GRID
    if a.extra_deltas:
        GRID = sorted(set(GRID) | {float(x) for x in a.extra_deltas.split(",") if x})
        print(f"grid extended to top {max(GRID):g} ({len(GRID)} points)")

    rng = np.random.default_rng(a.seed)
    A = run(a.feat_a, [s for s in a.subj_a.split(",") if s], a.label_a, a.lam, a.B, rng)
    B_ = run(a.feat_b, [s for s in a.subj_b.split(",") if s], a.label_b, a.lam, a.B, rng)

    print(f"\n  {'statistic':>16} {a.label_a[:14]:>15} {a.label_b[:14]:>15}   overlap?")
    print("  " + "-" * 62)
    for name in ("coherence_main", "locality_main", "interaction", "lapse_ratio"):
        ca, cb = A["stats"][name]["ci95"], B_["stats"][name]["ci95"]
        ov = "--" if not (ca and cb) else ("yes" if (ca[0] <= cb[1] and cb[0] <= ca[1])
                                           else "NO -- differs")
        print(f"  {name:>16} {A['stats'][name]['point']:>15.3f} "
              f"{B_['stats'][name]['point']:>15.3f}   {ov}")
    print("\n  Overlap of two marginal CIs is a weak check, not a test: the corpora "
          "have\n  disjoint subjects, so no paired contrast exists between them. It is "
          "read as\n  'not shown to differ', never as 'shown to agree'.")

    json.dump({"arm_a": A, "arm_b": B_}, open(a.out, "w"), indent=2)
    print(f"\nwrote {os.path.basename(a.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
