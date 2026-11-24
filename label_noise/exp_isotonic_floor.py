#!/usr/bin/env python3
# exp_isotonic_floor.py -- the floor and c re-estimated the way the reviewer asked (major 1, 6, 7).
"""WHAT CHANGES

  Crossing.  ROC-AUC is fitted as a MONOTONE function of severity by isotonic
             regression (weighted by frames per severity) and the 0.7 / 0.9 crossings
             are read off the fitted curve. No "sustained" guard, no hand rule: a curve
             that never reaches the target is censored and reported as such.
  Unit.      Every interval is a bootstrap over SUBJECTS (B replicates). Seeds and
             frames are nested within subjects and are never resampled on their own.
  Pooling.   The pooled constant is given two ways: (a) a joint subject bootstrap in
             which cells sharing a subject pool (all MPIIFaceGaze cells; both EYEDIAP
             cells) are resampled TOGETHER, so their correlation is carried into the
             interval; (b) DerSimonian-Laird random-effects pooling using each cell's
             bootstrap variance, with tau^2 and I^2 to say how much of the between-cell
             spread exceeds what subject sampling alone would produce.

e_hat convention is the manuscript's (Sections 6, Table 5): mean over subjects of the
subject's mean clean-frame LOO residual, pooled over delta <= 10, pc = 0.2.

Reads records only. Writes logs/ISOTONIC_FLOOR_REPORT.md and isotonic_floor_summary.json.
Usage:  python3 exp_isotonic_floor.py [-B 2000] [--seed 0]
"""
import argparse, json, os, sys, collections
import numpy as np
from sklearn.isotonic import IsotonicRegression
_HERE = os.path.dirname(os.path.abspath(__file__)); _AG = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
from unigaze_arm import roc_auc                                   # noqa: E402

RT_UNIQUE = "s000,s001,s002,s003,s004,s005,s006,s007,s008,s009,s011,s012,s013,s016".split(",")
CELLS = [  # (label, group, file, filter dict, residual key)
 ("ResNet-18 L0, fine-tuned, MPII",   "MPII",    "sigma_sweep_records.jsonl",      {"sigma_level": "L0", "pc": 0.2}, "loo_residual_deg"),
 ("ResNet-18 L1, fine-tuned, MPII",   "MPII",    "sigma_sweep_records.jsonl",      {"sigma_level": "L1", "pc": 0.2}, "loo_residual_deg"),
 ("ResNet-18 L2, fine-tuned, MPII",   "MPII",    "sigma_sweep_records.jsonl",      {"sigma_level": "L2", "pc": 0.2}, "loo_residual_deg"),
 ("ResNet-18 L0, frozen+ridge, MPII", "MPII",    "resnet_frozen_records.jsonl",    {"level": "L0", "pc": 0.2, "lam": 0.01}, "loo_residual_deg"),
 ("ResNet-18 L1, frozen+ridge, MPII", "MPII",    "resnet_frozen_records.jsonl",    {"level": "L1", "pc": 0.2, "lam": 0.01}, "loo_residual_deg"),
 ("ResNet-18 L2, frozen+ridge, MPII", "MPII",    "resnet_frozen_records.jsonl",    {"level": "L2", "pc": 0.2, "lam": 0.01}, "loo_residual_deg"),
 ("UniGaze, frozen+ridge, MPII",      "MPII",    "unigaze_arm_records.jsonl",      {"pc": 0.2, "lam": 0.01}, "loo_residual_deg"),
 ("UniGaze, frozen+ridge, EYEDIAP",   "EYEDIAP", "eyediap_unigaze_arm_records.jsonl", {"pc": 0.2, "lam": 0.01}, "loo_residual_deg"),
 ("ResNet-18 L0, fine-tuned, EYEDIAP","EYEDIAP", "exp_a_records_eyediap.jsonl",    {"model": "random_offset", "pc": 0.2}, "score_loo"),
 ("UniGaze, frozen+ridge, RT-GENE",   "RTGENE",  "rtgene_unigaze_arm_records.jsonl", {"pc": 0.2, "lam": 0.01, "subject": RT_UNIQUE}, "loo_residual_deg"),
]
KCELLS = [(f"ResNet-18 L0, fine-tuned, MPII, K={k}", "MPII", "sigma_sweep_records_K.jsonl", {"K": k, "pc": 0.2}, "loo_residual_deg") for k in (5, 9, 16)]


def load_cell(path, flt, key):
    """-> {subject: {delta: (scores, labels)}}"""
    out = collections.defaultdict(lambda: collections.defaultdict(lambda: ([], [])))
    p = os.path.join(_HERE, path)
    if not os.path.exists(p):
        return None
    for l in open(p):
        r = json.loads(l)
        ok = True
        for k, v in flt.items():
            rv = r.get(k)
            if isinstance(v, list): ok &= rv in v
            elif isinstance(v, float): ok &= rv is not None and abs(rv - v) < 1e-9
            else: ok &= rv == v
            if not ok: break
        if not ok or r.get(key) is None or r.get("model", "random_offset") != "random_offset":
            continue
        d = out[r["subject"]][float(r["delta"])]
        d[0].append(float(r[key])); d[1].append(bool(r["is_corrupted"]))
    return {s: {d: (np.array(a), np.array(b)) for d, (a, b) in dd.items()} for s, dd in out.items()}


def curve_and_crossings(cell, subjects, targets=(0.7, 0.9)):
    deltas = sorted({d for s in subjects for d in cell[s]})
    aucs, ns = [], []
    for d in deltas:
        sc = np.concatenate([cell[s][d][0] for s in subjects if d in cell[s]])
        lb = np.concatenate([cell[s][d][1] for s in subjects if d in cell[s]])
        aucs.append(roc_auc(sc, lb)); ns.append(len(lb))
    x, y, w = np.array(deltas), np.array(aucs), np.array(ns, float)
    ok = np.isfinite(y)
    fit = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x[ok], y[ok], sample_weight=w[ok])
    yf = fit.predict(x)
    cross = {}
    for t in targets:
        if yf[0] >= t: cross[t] = ("below", float(x[0]))
        elif yf[-1] < t: cross[t] = ("above", float(x[-1]))
        else:
            i = int(np.argmax(yf >= t))
            x0, x1, y0, y1 = x[i-1], x[i], yf[i-1], yf[i]
            cross[t] = ("ok", float(x0 + (t - y0) / (y1 - y0) * (x1 - x0)) if y1 > y0 else float(x1))
    return x, y, yf, cross


def e_hat(cell, subjects):
    per = []
    for s in subjects:
        v = np.concatenate([cell[s][d][0][~cell[s][d][1]] for d in cell[s] if d <= 10])
        per.append(v.mean())
    return float(np.mean(per))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-B", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rs = np.random.RandomState(a.seed)
    cells = []
    for lab, grp, f, flt, key in CELLS + KCELLS:
        c = load_cell(f, flt, key)
        if c and len(c) >= 10: cells.append((lab, grp, c))
        else: print(f"  (skipping {lab}: {len(c) if c else 0} subjects on record -- incomplete)")
    # point estimates
    point = {}
    for lab, grp, c in cells:
        S = sorted(c); x, y, yf, cr = curve_and_crossings(c, S); e = e_hat(c, S)
        point[lab] = dict(group=grp, n_subj=len(S), e_hat=e, floor07=cr[0.7], floor09=cr[0.9],
                          c=(cr[0.7][1] / e if cr[0.7][0] == "ok" else None),
                          grid=[float(v) for v in x], auc=[float(v) for v in y], auc_iso=[float(v) for v in yf])
    # joint subject bootstrap: one resample per group per replicate
    groups = {g: sorted({s for lab, gg, c in cells if gg == g for s in c}) for g in {gg for _, gg, _ in cells}}
    boot = {lab: {"floor07": [], "c": [], "e": []} for lab, _, _ in cells}
    pooled_mean, pooled_cv = [], []
    for b in range(a.B):
        pick = {g: list(rs.choice(S, len(S))) for g, S in groups.items()}
        cs = []
        for lab, grp, c in cells:
            S = [s for s in pick[grp] if s in c]
            if len(S) < 3:
                f, e = np.nan, np.nan
            else:
                x, y, yf, cr = curve_and_crossings(c, S); e = e_hat(c, S)
                f = cr[0.7][1] if cr[0.7][0] == "ok" else np.nan
            boot[lab]["floor07"].append(f); boot[lab]["e"].append(e); boot[lab]["c"].append(f / e)
            if lab in {l for l, _, _, _, _ in CELLS}: cs.append(f / e)
        cs = np.array(cs)
        if np.all(np.isfinite(cs)):
            pooled_mean.append(cs.mean()); pooled_cv.append(cs.std(ddof=1) / cs.mean())
    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    for lab in boot:
        for k in ("floor07", "c", "e"):
            v = np.array(boot[lab][k]); point[lab][k + "_ci95"] = ci(v); point[lab][k + "_boot_sd"] = float(np.nanstd(v))
        point[lab]["censored_frac"] = float(np.mean(~np.isfinite(boot[lab]["floor07"])))
    # DerSimonian-Laird random effects over the ten primary cells
    prim = [lab for lab, _, _, _, _ in CELLS if lab in point and point[lab]["c"] is not None]
    yv = np.array([point[l]["c"] for l in prim]); vv = np.array([point[l]["c_boot_sd"] ** 2 for l in prim])
    w = 1 / vv; fixed = (w * yv).sum() / w.sum(); Q = (w * (yv - fixed) ** 2).sum(); df = len(yv) - 1
    C = w.sum() - (w ** 2).sum() / w.sum(); tau2 = max(0.0, (Q - df) / C); wr = 1 / (vv + tau2)
    re = (wr * yv).sum() / wr.sum(); se = np.sqrt(1 / wr.sum()); I2 = max(0.0, (Q - df) / Q) if Q > 0 else 0.0
    summary = dict(B=a.B, cells=point,
                   pooled_equal_weight=dict(mean=float(yv.mean()), cv=float(yv.std(ddof=1) / yv.mean()),
                                            mean_ci95=ci(np.array(pooled_mean)), cv_ci95=ci(np.array(pooled_cv))),
                   random_effects=dict(c=float(re), se=float(se), ci95=[float(re - 1.96 * se), float(re + 1.96 * se)],
                                       tau=float(np.sqrt(tau2)), I2=float(I2), Q=float(Q), df=int(df), n_cells=len(prim)),
                   by_group={g: float(np.mean([point[l]["c"] for l in prim if point[l]["group"] == g])) for g in groups},
                   by_adaptation={k: float(np.mean([point[l]["c"] for l in prim if k in l])) for k in ("fine-tuned", "frozen")})
    json.dump(summary, open(os.path.join(_HERE, "isotonic_floor_summary.json"), "w"), indent=2)

    L = []; w_ = L.append
    w_("# Isotonic floor estimation, subject-level bootstrap (B = %d)\n" % a.B)
    w_("Crossing = first severity at which the isotonic (monotone, frame-weighted) fit of pooled ROC-AUC reaches the target; e_hat = mean over subjects of mean clean-frame LOO residual (delta <= 10, pc = 0.2). Intervals: percentile bootstrap over subjects; cells sharing a subject pool are resampled jointly.\n")
    w_("| cell | n | e_hat [95% CI] | floor@0.7 [95% CI] | c [95% CI] | boot SD of c | censored | floor@0.9 |")
    w_("|---|---|---|---|---|---|---|---|")
    for lab, grp, c in cells:
        p = point[lab]; f7 = p["floor07"]; f9 = p["floor09"]
        fs = f"{f7[1]:.2f}" if f7[0] == "ok" else f"{f7[0]} grid ({f7[1]:g})"
        f9s = f"{f9[1]:.2f}" if f9[0] == "ok" else f"{f9[0]} {f9[1]:g}"
        cs = f"{p['c']:.3f} [{p['c_ci95'][0]:.3f}, {p['c_ci95'][1]:.3f}]" if p["c"] else "--"
        w_(f"| {lab} | {p['n_subj']} | {p['e_hat']:.2f} [{p['e_ci95'][0]:.2f}, {p['e_ci95'][1]:.2f}] | {fs} [{p['floor07_ci95'][0]:.2f}, {p['floor07_ci95'][1]:.2f}] | {cs} | {p['c_boot_sd']:.3f} | {100*p['censored_frac']:.0f}% | {f9s} |")
    pe, re_ = summary["pooled_equal_weight"], summary["random_effects"]
    w_(f"\n## Pooled constant over the {re_['n_cells']} primary cells\n")
    w_(f"- Equal-weight mean c = {pe['mean']:.3f}, joint-bootstrap 95% CI [{pe['mean_ci95'][0]:.3f}, {pe['mean_ci95'][1]:.3f}]; CV = {pe['cv']:.3f} [{pe['cv_ci95'][0]:.3f}, {pe['cv_ci95'][1]:.3f}].")
    w_(f"- Random-effects (DerSimonian-Laird) c = {re_['c']:.3f}, 95% CI [{re_['ci95'][0]:.3f}, {re_['ci95'][1]:.3f}]; between-cell tau = {re_['tau']:.3f}, I^2 = {100*re_['I2']:.0f}% (Q = {re_['Q']:.1f} on {re_['df']} df).")
    w_(f"- By dataset: " + ", ".join(f"{g} {v:.3f}" for g, v in summary["by_group"].items()) + ". By adaptation: " + ", ".join(f"{k} {v:.3f}" for k, v in summary["by_adaptation"].items()) + ".")
    w_("\nReading: tau is the between-cell spread of c AFTER subject-sampling noise is removed; I^2 is the share of the observed spread that exceeds sampling noise. A small tau with I^2 near zero says the cells are consistent with one value; a large I^2 says they are not, and the equal-weight CV would then understate the disagreement.\n")
    w_("## Isotonic fit versus raw AUC, per cell\n")
    for lab, grp, c in cells:
        p = point[lab]
        w_(f"**{lab}**  \n" + "  ".join(f"{d:g}:{r:.3f}/{f:.3f}" for d, r, f in zip(p["grid"], p["auc"], p["auc_iso"])) + "  (delta: raw/isotonic)\n")
    open(os.path.join(_AG, "logs", "ISOTONIC_FLOOR_REPORT.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:len(cells) + 12]))

if __name__ == "__main__":
    sys.exit(main())
