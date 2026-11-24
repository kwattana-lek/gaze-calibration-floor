#!/usr/bin/env python3
# unc_calib_probe.py — does undetectable label corruption break uncertainty calibration?
"""THE QUESTION

The CVPR 2026 uncertainty-calibration work for gaze fits a post-hoc calibrator
from ~50 target-domain samples and reports Coverage Probability Error (CPE), the
gap between the confidence a model claims and the coverage it actually achieves.
It assumes those samples are correctly labelled; it says nothing about
calibration-sample quality or label noise.

Our detectability floor says a label displaced by less than about 1.03 times the
model's own error cannot be flagged by any residual scorer we tested, and the
damage-ratio result says such a frame is nearly harmless to the POINT ESTIMATE.
Neither says anything about the UNCERTAINTY. A corrupted frame inflates apparent
spread, and a calibrator fitted on contaminated samples can end up confidently
wrong in precisely the regime we have been calling harmless.

  CPE degrades below the floor -> undetectable corruption is harmless for the
      mean and harmful for the uncertainty. That qualifies our own "harmless as
      well as invisible" claim, and it is the paper.
  CPE does not degrade        -> the floor also protects calibration. Weaker,
      still reportable, and it strengthens the existing framing.

PROTOCOL (mirrors the CVPR setup; this matters)

  fit  :  9 frames, CLEAN     -> personalizes the GP. This is the deployed model.
  cal  : 50 frames, CORRUPTED -> fits the post-hoc coverage calibrator. These are
                                the "~50 target-domain samples" whose cleanliness
                                the CVPR method assumes.
  eval : the rest,  CLEAN     -> measures realised coverage.

Every residual used is out-of-sample: cal and eval are both new points with
respect to the 9-frame fit. An earlier draft fitted the calibrator on
leave-one-out residuals of a 50-point fit, which fails because with n << d the
LOO distribution is not exchangeable with the distribution at genuinely new
points -- the self-test caught it as a calibrated CPE near 0.50.

Detection is scored on the same 50 calibration frames, by residual magnitude
under the personalized model, so the detectability curve and the CPE curve come
from identical corruption draws and are directly comparable. That is the whole
point: "below the floor" has to be measured at the set size the calibrator
actually uses.

WHY IT IS CHEAP

The UniGaze features are already cached (400 frames x 1280-d per subject) and a
linear-kernel GP gives closed-form predictive mean AND variance with no training
at all. The kernel scale is fitted by marginal likelihood; the raw linear kernel
carries FEATURE units rather than label units, which pins CPE at a constant and
is the second thing the self-test checks.

No scipy: isotonic regression is pool-adjacent-violators, written out below, and
the normal quantile comes from statistics.NormalDist.

Usage:  python3 unc_calib_probe.py --selftest
        python3 unc_calib_probe.py [--nfit 9] [--ncal 50] [--seeds 5]
"""
import argparse
import json
import os
import sys
from statistics import NormalDist

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

FEAT_DIR = os.path.join(_HERE, "unigaze_features")
OUT = os.path.join(_HERE, "unc_calib_records.jsonl")
SUBJ = [f"p{i:02d}" for i in range(15)]
DELTAS = [0, 1, 2, 3, 4, 5, 6, 8, 10, 14, 20]
PCS = [0.1, 0.2, 0.4]
QGRID = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
LAM_GRID = np.logspace(-4, 4, 33)
DEG = 180.0 / np.pi


# --------------------------------------------------------------- isotonic
def pava(y):
    """Pool-adjacent-violators: least-squares fit subject to y non-decreasing."""
    y = np.asarray(y, float)
    lvl, cnt = [], []
    for v in y:
        lvl.append(float(v)); cnt.append(1)
        while len(lvl) > 1 and lvl[-2] > lvl[-1]:
            v2, c2 = lvl.pop(), cnt.pop()
            v1, c1 = lvl.pop(), cnt.pop()
            lvl.append((v1 * c1 + v2 * c2) / (c1 + c2)); cnt.append(c1 + c2)
    out, p = np.empty(len(y)), 0
    for v, c in zip(lvl, cnt):
        out[p:p + c] = v; p += c
    return out


# --------------------------------------------------------------- GP head
def fit_axis(Xc, yc):
    """Linear-kernel GP hyper-parameters by marginal likelihood.

    k(x,x') = sf2 * x^T x', noise sf2*lam. lam is chosen on a grid and sf2 then
    has the profile-likelihood closed form sf2 = y^T A^-1 y / n with
    A = X X^T + lam I. Fitting the scale is not optional: the bare linear kernel
    returns a variance in FEATURE units, which makes the intervals meaningless.
    """
    G = Xc @ Xc.T
    n = len(yc)
    best = None
    for lam in LAM_GRID:
        A = G + lam * np.eye(n)
        sign, logdet = np.linalg.slogdet(A)
        if sign <= 0:
            continue
        quad = float(yc @ np.linalg.solve(A, yc))
        if quad <= 0:
            continue
        sf2 = quad / n
        nll = n * np.log(sf2) + logdet
        if best is None or nll < best[0]:
            best = (nll, float(lam), float(sf2))
    return (1.0, float(np.var(yc) + 1e-12)) if best is None else (best[1], best[2])


def gp_axis(Xc, yc, Xs, lam, sf2):
    """Predictive mean and variance (noise included) at new points, one axis."""
    n = len(Xc)
    A = Xc @ Xc.T + lam * np.eye(n)
    mean = (Xs @ Xc.T) @ np.linalg.solve(A, yc)
    ks = Xs @ Xc.T
    v = (np.einsum("ij,ij->i", Xs, Xs)
         - np.einsum("ij,ij->i", ks, np.linalg.solve(A, ks.T).T) + lam)
    return mean, sf2 * np.maximum(v, 1e-12)


# --------------------------------------------------------------- metrics
def roc_auc(scores, pos):
    s = np.asarray(scores, float); p = np.asarray(pos, bool)
    a, b = int(p.sum()), int((~p).sum())
    if a == 0 or b == 0:
        return float("nan")
    u, inv, c = np.unique(s, return_inverse=True, return_counts=True)
    cs = np.cumsum(c); avg = (cs - c + cs + 1) / 2.0
    return float((avg[inv][p].sum() - a * (a + 1) / 2) / (a * b))


def _cov(err, sd, z):
    return float(np.mean(np.abs(err) <= z * sd))


def cpe_for_axis(err_cal, sd_cal, err_ev, sd_ev):
    """Fit the coverage calibrator on the CALIBRATION residuals, score it on the
    EVALUATION residuals. Returns (CPE before calibration, CPE after)."""
    zs = [NormalDist().inv_cdf(0.5 + q / 2) for q in QGRID]
    cal_cov = pava([_cov(err_cal, sd_cal, z) for z in zs])
    raw, cal = [], []
    for q, z in zip(QGRID, zs):
        raw.append(_cov(err_ev, sd_ev, z) - q)
        j = min(int(np.searchsorted(cal_cov, q)), len(zs) - 1)
        cal.append(_cov(err_ev, sd_ev, zs[j]) - q)
    return (float(np.sqrt(np.mean(np.square(raw)))),
            float(np.sqrt(np.mean(np.square(cal)))))


# --------------------------------------------------------------- self-test
def selftest():
    """Two negative controls on well-specified Gaussian data with clean labels.

    (1) CPE must be small both before and after calibration. If it is not, any
        CPE rise under corruption is an artifact of the probe, not a finding.
    (2) The feature scale is deliberately mismatched from the label scale
        (features ~N(0,0.05), labels in radians) so that a kernel whose variance
        carries feature units fails loudly instead of quietly.
    """
    rng = np.random.RandomState(0)
    nf, nc, m, d = 9, 50, 400, 200
    w = rng.randn(d, 2) * 0.02
    s = 0.03
    X = rng.randn(nf + nc + m, d) * 0.05
    Y = X @ w + rng.randn(nf + nc + m, 2) * s
    Xf, Xc_, Xe = X[:nf], X[nf:nf + nc], X[nf + nc:]
    Yf, Yc_, Ye = Y[:nf], Y[nf:nf + nc], Y[nf + nc:]
    mux = Xf.mean(0, keepdims=True)
    Xfc, Xcc, Xec = Xf - mux, Xc_ - mux, Xe - mux
    ok = True
    for ax in (0, 1):
        ym = Yf[:, ax].mean()
        lam, sf2 = fit_axis(Xfc, Yf[:, ax] - ym)
        mu_c, v_c = gp_axis(Xfc, Yf[:, ax] - ym, Xcc, lam, sf2)
        mu_e, v_e = gp_axis(Xfc, Yf[:, ax] - ym, Xec, lam, sf2)
        raw, cal = cpe_for_axis(Yc_[:, ax] - (mu_c + ym), np.sqrt(v_c),
                                Ye[:, ax] - (mu_e + ym), np.sqrt(v_e))
        print(f"  axis {ax}: lam {lam:<9.3g} sf2 {sf2:<9.3g} "
              f"CPE raw {raw:.4f}  calibrated {cal:.4f}")
        if cal > 0.12 or raw > 0.20:
            ok = False
    mono = pava([0.5, 0.2, 0.9, 0.4])
    print(f"  PAVA monotone: {bool(np.all(np.diff(mono) >= -1e-12))} -> {np.round(mono,3)}")
    print("  SELF-TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# --------------------------------------------------------------- main
def main():
    global QGRID
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfit", type=int, default=9)
    ap.add_argument("--ncal", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--qgrid", type=float, default=None,
                    help="uniform QGRID step (e.g. 0.01). Default keeps the "
                         "10-point grid the probe shipped with, whose coarseness "
                         "contributes ~0.017 of pure artifact to calibrated CPE "
                         "(see cpe_floor_diagnostic.py).")
    ap.add_argument("--out", default=None,
                    help="records path. Default is unc_calib_records.jsonl, "
                         "which this script TRUNCATES on every run.")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    if a.qgrid:
        if not a.out:
            sys.stderr.write("ABORT: --qgrid produces records that are not "
                             "comparable to the canonical ones. Pass --out.\n")
            return 2
        QGRID = [round(float(q), 6) for q in np.arange(a.qgrid, 1.0, a.qgrid)]
    out_path = a.out or OUT
    print(f"QGRID: {len(QGRID)} levels [{QGRID[0]:g} .. {QGRID[-1]:g}]  ->  "
          f"{os.path.basename(out_path)}", flush=True)

    import corruption as corr

    rows = []
    for si, s in enumerate(SUBJ):
        f = os.path.join(FEAT_DIR, f"{s}.npz")
        if not os.path.exists(f):
            sys.exit(f"ABORT: no cached features for {s}")
        z = np.load(f)
        F, Y = z["F"].astype(np.float64), z["Y"].astype(np.float64)
        need = a.nfit + a.ncal + 50
        if len(F) < need:
            sys.exit(f"ABORT: {s} has {len(F)} frames, need {need}")

        for seed in range(a.seeds):
            rs = np.random.RandomState(1000 * si + seed)
            p = rs.permutation(len(F))
            fit, cal, ev = p[:a.nfit], p[a.nfit:a.nfit + a.ncal], p[a.nfit + a.ncal:]
            mux = F[fit].mean(0, keepdims=True)
            Xf, Xc_, Xe = F[fit] - mux, F[cal] - mux, F[ev] - mux
            Yf, Yc_, Ye = Y[fit], Y[cal], Y[ev]

            hyp, pred_c, pred_e = [], [], []
            for ax in (0, 1):
                ym = Yf[:, ax].mean()
                lam, sf2 = fit_axis(Xf, Yf[:, ax] - ym)
                mc, vc = gp_axis(Xf, Yf[:, ax] - ym, Xc_, lam, sf2)
                me, ve = gp_axis(Xf, Yf[:, ax] - ym, Xe, lam, sf2)
                hyp.append((lam, sf2))
                pred_c.append((mc + ym, np.sqrt(vc)))
                pred_e.append((me + ym, np.sqrt(ve)))

            mu_e = np.stack([pred_e[0][0], pred_e[1][0]], 1)
            mae = float(np.mean(np.linalg.norm(mu_e - Ye, axis=1)) * DEG)

            for pc in PCS:
                for d in DELTAS:
                    if d == 0:
                        Ycal, iscor = Yc_, np.zeros(len(Yc_), bool)
                    else:
                        c = corr.apply_corruption(
                            Yc_, model="random_offset",
                            rng=np.random.RandomState(rs.randint(1 << 30)),
                            pc=pc, delta_deg=d, pool_labels=Y)
                        Ycal = c["corrupted_labels"]
                        iscor = np.asarray(c["is_corrupted"], bool)

                    rec = {"subject": s, "seed": seed, "pc": pc, "delta": float(d),
                           "n_fit": a.nfit, "n_cal": a.ncal, "n_eval": len(ev),
                           "n_corrupted": int(iscor.sum()), "mae_eval_deg": mae}
                    for ax, name in ((0, "pitch"), (1, "yaw")):
                        mc, sc = pred_c[ax]
                        me, se = pred_e[ax]
                        raw, cal_ = cpe_for_axis(Ycal[:, ax] - mc, sc,
                                                 Ye[:, ax] - me, se)
                        rec[f"cpe_raw_{name}"] = raw
                        rec[f"cpe_cal_{name}"] = cal_
                    if d > 0 and iscor.any() and not iscor.all():
                        r = np.linalg.norm(
                            Ycal - np.stack([pred_c[0][0], pred_c[1][0]], 1), axis=1)
                        rec["auc_detect"] = roc_auc(r * DEG, iscor)
                    rows.append(rec)
        print(f"  {s}: {len(rows)} cells", flush=True)

    with open(out_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(rows)} records -> {os.path.basename(out_path)}")
    report(rows)
    return 0


def _cpe(r):
    return (r["cpe_cal_pitch"] + r["cpe_cal_yaw"]) / 2


def report(rows):
    print(f"\n{'='*74}\nn_fit={rows[0]['n_fit']}  n_cal={rows[0]['n_cal']}  "
          f"n_eval={rows[0]['n_eval']}   (means over subjects and seeds)\n{'='*74}")
    base = {pc: float(np.mean([_cpe(r) for r in rows
                               if r["delta"] == 0 and r["pc"] == pc])) for pc in PCS}
    for pc in PCS:
        print(f"\n--- pc = {pc:g}   (clean-label CPE = {base[pc]:.4f}) ---")
        print(f"  {'delta':>6}{'CPE':>10}{'vs clean':>11}{'CPE raw':>10}{'AUC detect':>12}")
        for d in DELTAS:
            g = [r for r in rows if r["delta"] == d and r["pc"] == pc]
            if not g:
                continue
            cc = float(np.mean([_cpe(r) for r in g]))
            cr = float(np.mean([(r["cpe_raw_pitch"] + r["cpe_raw_yaw"]) / 2 for r in g]))
            au = [r["auc_detect"] for r in g if "auc_detect" in r]
            print(f"  {d:>6g}{cc:>10.4f}{cc-base[pc]:>+11.4f}{cr:>10.4f}"
                  + (f"{np.mean(au):>12.3f}" if au else f"{'--':>12}"))

    print(f"\n{'='*74}\nVERDICT\n{'='*74}")
    for pc in PCS:
        ds = [d for d in DELTAS if d > 0]
        au = {}
        for d in ds:
            g = [r["auc_detect"] for r in rows
                 if r["delta"] == d and r["pc"] == pc and "auc_detect" in r]
            if g:
                au[d] = float(np.mean(g))
        floor = None
        for i in range(1, len(ds)):
            lo, hi = ds[i - 1], ds[i]
            if lo in au and hi in au and au[hi] >= 0.7 > au[lo]:
                floor = lo + (0.7 - au[lo]) / (au[hi] - au[lo]) * (hi - lo)
                break
        if floor is None:
            print(f"  pc={pc:g}: floor not located on this grid -- inconclusive")
            continue
        below = [d for d in ds if d < floor]
        if not below:
            print(f"  pc={pc:g}: floor {floor:.2f} deg is below the smallest "
                  f"delta swept -- no sub-floor cells, inconclusive")
            continue
        sub = [r for r in rows if r["pc"] == pc and r["delta"] in below]
        cc = float(np.mean([_cpe(r) for r in sub]))
        # Bootstrap over subjects, not a magnitude threshold. An earlier draft
        # called anything under 0.02 "no material degradation", which is an
        # arbitrary cutoff and reported a real effect as absent.
        lo, hi, mean_d = _boot(rows, pc, below, 4000)
        print(f"  pc={pc:g}: detection floor at n_cal={rows[0]['n_cal']} is "
              f"{floor:.2f} deg (deltas below it: {below})")
        print(f"           CPE below floor {cc:.4f} vs clean {base[pc]:.4f}")
        print(f"           rise {mean_d:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]"
              f"  ({100*mean_d/base[pc]:+.0f}% relative)")
        print(f"           -> {'REAL: undetectable corruption still degrades calibration' if lo > 0 else 'not distinguishable from zero'}")


def _boot(rows, pc, below, n=4000):
    import random
    rng = random.Random(0)
    subs = sorted({r["subject"] for r in rows})
    by = {}
    for r in rows:
        if r["pc"] == pc:
            by.setdefault(r["subject"], []).append(r)
    d = []
    for _ in range(n):
        pick = [rng.choice(subs) for _ in subs]
        rr = [x for s in pick for x in by[s]]
        b = np.mean([_cpe(x) for x in rr if x["delta"] == 0])
        s_ = np.mean([_cpe(x) for x in rr if x["delta"] in below])
        d.append(s_ - b)
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)),
            float(np.mean(d)))


if __name__ == "__main__":
    sys.exit(main())
