#!/usr/bin/env python3
# exp_real_lapse_timesplit.py -- POST-HOC: why do real lapses score below their synthetic twins?
"""POST-HOC, and labelled so wherever it is reported. Written AFTER
real_lapse_summary.json was seen; it tests an explanation for one of its results.

THE RESULT BEING EXPLAINED
exp_real_lapse.py recorded in advance the expectation that early post-jump lapses
would be detected no worse than synthetic corruption of matched displacement. They
are detected WORSE, by 0.03-0.08 AUC in every bin above the floor, intervals
excluding zero. In the late window (200-400 ms) the deficit grows to ~0.28.

THE EXPLANATION ON TRIAL
LapseDeg assumes the eyes are still on the previous target. Saccade latency is
~150-250 ms but not constant: some saccades launch early, and an eye already on the
new target makes the label CORRECT. Such frames sit in the "lapse" class with a
clean-frame residual and pull the AUC down. If that is the mechanism, the deficit
must grow monotonically with time since the jump, and should be near zero in the
first ~100 ms, before almost any saccade can have launched.

If the deficit is as large at 0-100 ms as later, the explanation is wrong and
something else separates real from simulated corruption (e.g. the frame's image
differing, not just its label) -- which would matter more.

Same protocol, same cached features, same seeding scheme as exp_real_lapse.py.
Positives are restricted to LapseDeg >= 8 deg (clear of the ~5 deg floor, where both
arms are at chance and a difference cannot show). Writes only
real_lapse_timesplit_records.jsonl and real_lapse_timesplit_summary.json.

Usage:  python3 exp_real_lapse_timesplit.py [--trials 300] [--boot 2000]
"""
import argparse, json, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import exp_real_lapse as X                                   # noqa: E402

WINDOWS = [(0, 3, "0-100 ms"), (3, 6, "100-200 ms"), (6, 9, "200-300 ms"), (9, 12, "300-400 ms")]
MIN_DEG = 8.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    import gaze_math as gm
    from unigaze_arm import K, residuals

    recs = []
    for s in X.SUBJ:
        zc = np.load(os.path.join(X.CLEAN_DIR, f"{s}.npz"))
        zl = np.load(os.path.join(X.LAPSE_DIR, f"{s}.npz"))
        Fs, Ys = zc["F"][:K].astype(np.float64), zc["Y"][:K].astype(np.float64)
        pj = np.char.startswith(zl["kind"].astype(str), "PJ")
        for lo, hi, name in WINDOWS:
            pool = np.where(pj & (zl["since"] >= lo) & (zl["since"] < hi) & (zl["lapse"] >= MIN_DEG))[0]
            if len(pool) < X.NCORR:
                continue
            for t in range(a.trials):
                r = X.rng_for("timesplit", s, name, t)
                pos = r.choice(K, X.NCORR, replace=False)
                src = r.choice(pool, X.NCORR, replace=False)
                F, Y, Yt = Fs.copy(), Ys.copy(), Ys.copy()
                F[pos], Y[pos] = zl["F"][src], zl["Y"][src]
                for p, d in zip(pos, zl["lapse"][src]):
                    Yt[p] = gm.rotate_by_angle(Ys[p], float(d), rng=r)[0]
                loo, tw = residuals(F, Y, X.LAM)[1], residuals(Fs, Yt, X.LAM)[1]
                for i in range(K):
                    hit = i in pos
                    recs.append({"subject": s, "kind": name, "trial": t, "pos": i, "is_lapse": bool(hit),
                                 "lapse_deg": float(zl["lapse"][src][list(pos).index(i)]) if hit else None,
                                 "loo": float(loo[i]), "loo_twin": float(tw[i])})
        X.log(f"  {s} done")
    with open(os.path.join(_HERE, "real_lapse_timesplit_records.jsonl"), "w") as fh:
        fh.write("\n".join(json.dumps(r) for r in recs) + "\n")

    out = {"status": "POST-HOC", "min_lapse_deg": MIN_DEG, "trials": a.trials, "windows": {}}
    print(f"\nPOST-HOC. Positives: post-jump NL frames with LapseDeg >= {MIN_DEG:g} deg.")
    print(f"  {'since jump':<12} {'real':>6} {'95% CI':<16} {'synthetic':>9} {'95% CI':<16} {'real-synth':>10} {'95% CI':<17} {'median deg':>10}")
    for lo, hi, name in WINDOWS:
        rn, sy = (X.auc_ci(recs, name, k, boot=a.boot) for k in ("loo", "loo_twin"))
        if not rn:
            continue
        m = min(len(rn["boot"]), len(sy["boot"]))
        d = np.array(rn["boot"][:m]) - np.array(sy["boot"][:m])
        dci = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        med = float(np.median([r["lapse_deg"] for r in recs if r["kind"] == name and r["is_lapse"]]))
        out["windows"][name] = {"real": rn["auc"], "real_ci95": rn["ci95"], "synthetic": sy["auc"],
                                "synthetic_ci95": sy["ci95"], "diff": rn["auc"] - sy["auc"],
                                "diff_ci95": dci, "median_lapse_deg": med}
        print(f"  {name:<12} {rn['auc']:6.3f} [{rn['ci95'][0]:.3f}, {rn['ci95'][1]:.3f}]  "
              f"{sy['auc']:9.3f} [{sy['ci95'][0]:.3f}, {sy['ci95'][1]:.3f}]  "
              f"{rn['auc']-sy['auc']:+10.3f} [{dci[0]:+.3f}, {dci[1]:+.3f}] {med:10.1f}")
    json.dump(out, open(os.path.join(_HERE, "real_lapse_timesplit_summary.json"), "w"), indent=2)
    print("\nwrote real_lapse_timesplit_records.jsonl, real_lapse_timesplit_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
