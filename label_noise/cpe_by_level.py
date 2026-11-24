#!/usr/bin/env python3
# cpe_by_level.py -- where in the coverage curve does corruption do its damage?
"""THE PREDICTION

Contamination at rate p_c is a MIXTURE, not a rescaling. On the calibration set

    cal_cov(z) = P(|err_cal| <= z*sd)
               ~ (1 - p_c) * cov_clean(z)  +  p_c * cov_corrupt(z)

and for a displacement delta well above the model's own error, cov_corrupt(z) is
near zero at moderate z. So cal_cov SATURATES near 1 - p_c: no z on the grid
reaches a higher empirical coverage, and searchsorted clamps to the top of the
grid for every nominal level above it.

  Prediction: calibrated coverage error is concentrated at q > 1 - p_c and
  near-absent below. Breakpoints 0.9 / 0.8 / 0.6 for p_c = 0.1 / 0.2 / 0.4.

If it holds, the decoupling has an exact mechanism: coverage is a quantile
functional, the q-quantile of a contaminated sample is the clean q/(1-p_c)
quantile while q < 1 - p_c, and RMS -- a second-moment functional -- responds to
the tail that the quantiles ignore. That is why an equal-RMS rescale does MORE
damage than the corruption it matches.

Reuses unc_calib_probe's split, GP fit and rs consumption order. Aggregates in
memory; writes one compact summary file.

Usage:  python3 cpe_by_level.py [--qgrid 0.02] [--seeds 5]
"""
import argparse, json, os, sys
from collections import defaultdict
from statistics import NormalDist

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import unc_calib_probe as P     # noqa: E402
import corruption as corr       # noqa: E402

OUT = os.path.join(_HERE, "cpe_by_level_summary.json")
BANDS = {"below": [1., 2., 3., 4., 5.], "above": [8., 10., 14., 20.]}


def errs_by_level(err_cal, sd_cal, err_ev, sd_ev, qgrid):
    """Signed calibrated coverage error at each nominal level."""
    zs = [NormalDist().inv_cdf(0.5 + q / 2) for q in qgrid]
    cov = lambda e, s, z: float(np.mean(np.abs(e) <= z * s))   # noqa: E731
    cal_cov = P.pava([cov(err_cal, sd_cal, z) for z in zs])
    out = []
    for q in qgrid:
        j = min(int(np.searchsorted(cal_cov, q)), len(zs) - 1)
        out.append(cov(err_ev, sd_ev, zs[j]) - q)
    return np.asarray(out), float(cal_cov[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qgrid", type=float, default=0.02)
    ap.add_argument("--nfit", type=int, default=9)
    ap.add_argument("--ncal", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    Q = [round(float(q), 6) for q in np.arange(a.qgrid, 1.0, a.qgrid)]

    acc = defaultdict(list)     # (pc, band, arm) -> per-level |error| vectors
    sat = defaultdict(list)     # (pc, band) -> max attainable cal coverage

    for si, s in enumerate(P.SUBJ):
        z = np.load(os.path.join(P.FEAT_DIR, f"{s}.npz"))
        F, Y = z["F"].astype(np.float64), z["Y"].astype(np.float64)
        for seed in range(a.seeds):
            rs = np.random.RandomState(1000 * si + seed)
            p = rs.permutation(len(F))
            fit, cal, ev = p[:a.nfit], p[a.nfit:a.nfit + a.ncal], p[a.nfit + a.ncal:]
            mux = F[fit].mean(0, keepdims=True)
            Xf, Xc_, Xe = F[fit] - mux, F[cal] - mux, F[ev] - mux
            Yf, Yc_, Ye = Y[fit], Y[cal], Y[ev]
            pred_c, pred_e = [], []
            for ax in (0, 1):
                ym = Yf[:, ax].mean()
                lam, sf2 = P.fit_axis(Xf, Yf[:, ax] - ym)
                mc, vc = P.gp_axis(Xf, Yf[:, ax] - ym, Xc_, lam, sf2)
                me, ve = P.gp_axis(Xf, Yf[:, ax] - ym, Xe, lam, sf2)
                pred_c.append((mc + ym, np.sqrt(vc)))
                pred_e.append((me + ym, np.sqrt(ve)))

            for pc in P.PCS:
                for d in P.DELTAS:          # rs order must match the probe
                    if d == 0:
                        Ycal = Yc_
                    else:
                        Ycal = corr.apply_corruption(
                            Yc_, model="random_offset",
                            rng=np.random.RandomState(rs.randint(1 << 30)),
                            pc=pc, delta_deg=d, pool_labels=Y)["corrupted_labels"]
                    band = next((b for b, ds in BANDS.items() if d in ds), None)
                    if band is None:
                        continue
                    for ax in (0, 1):
                        mc, sc = pred_c[ax]
                        me, se = pred_e[ax]
                        e_ev, e_cln = Ye[:, ax] - me, Yc_[:, ax] - mc
                        e_msr = Ycal[:, ax] - mc
                        sca = float(np.sqrt(np.mean(e_msr ** 2))
                                    / np.sqrt(np.mean(e_cln ** 2)))
                        v_m, top = errs_by_level(e_msr, sc, e_ev, se, Q)
                        v_s, _ = errs_by_level(e_cln * sca, sc, e_ev, se, Q)
                        v_c, _ = errs_by_level(e_cln, sc, e_ev, se, Q)
                        acc[(pc, band, "measured")].append(np.abs(v_m))
                        acc[(pc, band, "scaled")].append(np.abs(v_s))
                        acc[(pc, band, "clean")].append(np.abs(v_c))
                        sat[(pc, band)].append(top)
        print(f"  {s} done", flush=True)

    show = [q for q in Q if abs(q * 100 - round(q * 100)) < 1e-6
            and round(q * 100) % 10 == 0] + [0.94, 0.98]
    show = sorted(set(show))
    print("\nMean |calibrated coverage error| by nominal level q.")
    print("Breakpoint prediction: damage concentrates at q > 1 - p_c.\n")
    for pc in P.PCS:
        bp = 1 - pc
        print(f"--- p_c = {pc:g}   predicted breakpoint q > {bp:.2f}   "
              f"(max attainable cal coverage, above-floor cells: "
              f"{np.mean(sat[(pc,'above')]):.3f}) ---")
        print(f"{'q':>6} " + "".join(f"{b+'/'+a:>17}" for b in ("below", "above")
                                     for a in ("clean", "measured")))
        for q in show:
            i = Q.index(round(q, 6))
            cells = []
            for b in ("below", "above"):
                for arm in ("clean", "measured"):
                    cells.append(np.mean([v[i] for v in acc[(pc, b, arm)]]))
            mark = "  <-- above breakpoint" if q > bp else ""
            print(f"{q:>6.2f} " + "".join(f"{c:>17.4f}" for c in cells) + mark)
        print()

    json.dump({"qgrid": Q,
               "mean_abs_err": {f"{pc}|{b}|{a}":
                                [float(x) for x in np.mean(acc[(pc, b, a)], 0)]
                                for pc in P.PCS for b in BANDS for a in
                                ("clean", "scaled", "measured")},
               "max_attainable_cal_cov": {f"{pc}|{b}": float(np.mean(sat[(pc, b)]))
                                          for pc in P.PCS for b in BANDS}},
              open(OUT, "w"), indent=2)
    print(f"wrote {os.path.basename(OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
