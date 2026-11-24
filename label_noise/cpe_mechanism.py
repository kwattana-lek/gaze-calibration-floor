#!/usr/bin/env python3
# cpe_mechanism.py -- is the CPE rise just residual inflation?
"""THE QUESTION

Corrupted calibration labels inflate |residual| on the calibration set. A
post-hoc calibrator fitted on inflated residuals widens its intervals, the
evaluation set then over-covers, and CPE rises. That chain is nearly
deterministic and would make Gap A a corollary of the damage-ratio result rather
than a phenomenon of its own.

But isotonic recalibration is INVARIANT to a global rescaling of the calibration
residuals: scaling every err_cal by s turns cal_cov(z) into cov(z/s), which only
relabels the grid, and in the limit of a fine grid and large n_cal the calibrator
absorbs it completely. So scale inflation alone predicts NO CPE rise.

What corruption actually produces is not a rescaling. It is a MIXTURE -- most
frames untouched, a minority displaced by delta -- which is heavy-tailed. A
calibrator can undo a scale change and cannot undo a shape change.

THREE ARMS, identical eval residuals and identical predictive sds:

  clean    : err_cal from the true labels                      (baseline)
  scaled   : clean err_cal multiplied by s, where s matches the
             corrupted set's RMS per axis                      (pure inflation)
  measured : err_cal from the actually-corrupted labels        (what the probe did)

  measured ~ scaled  -> the rise is scale inflation; Gap A is a corollary of
                        Experiment B and adds little.
  measured >  scaled  -> the rise is the SHAPE change that no post-hoc scale
                        calibration can absorb. That gap is the contribution.

The split, the GP fit and the corruption draws reproduce unc_calib_probe.py
exactly, including the order in which `rs` is consumed, so cells are comparable
to unc_calib_records.jsonl.

Runs on the cached UniGaze features. WRITES ONLY its own records file.

Usage:  python3 cpe_mechanism.py [--qgrid 0.02] [--seeds 5]
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import unc_calib_probe as P          # noqa: E402  (import only; main() never called)
import corruption as corr            # noqa: E402

OUT = os.path.join(_HERE, "cpe_mechanism_records.jsonl")
BELOW = [1.0, 2.0, 3.0, 4.0, 5.0]    # below the n_cal=50 detection floor (5.5-5.9 deg)
ARMS = ("clean", "scaled", "measured")


def cell_cpe(err_cal, sc, err_ev, se):
    return P.cpe_for_axis(err_cal, sc, err_ev, se)[1]      # calibrated branch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qgrid", type=float, default=0.02)
    ap.add_argument("--nfit", type=int, default=9)
    ap.add_argument("--ncal", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()

    P.QGRID = [round(float(q), 6) for q in np.arange(a.qgrid, 1.0, a.qgrid)]
    print(f"QGRID {len(P.QGRID)} levels; {a.seeds} seeds; "
          f"n_fit {a.nfit}, n_cal {a.ncal}\n", flush=True)

    rows = []
    for si, s in enumerate(P.SUBJ):
        f = os.path.join(P.FEAT_DIR, f"{s}.npz")
        if not os.path.exists(f):
            sys.exit(f"ABORT: no cached features for {s}")
        z = np.load(f)
        F, Y = z["F"].astype(np.float64), z["Y"].astype(np.float64)

        for seed in range(a.seeds):
            rs = np.random.RandomState(1000 * si + seed)
            p = rs.permutation(len(F))
            fit = p[:a.nfit]
            cal = p[a.nfit:a.nfit + a.ncal]
            ev = p[a.nfit + a.ncal:]
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

            # rs is consumed below in exactly the probe's order, so the
            # corruption draws match unc_calib_records.jsonl cell for cell.
            for pc in P.PCS:
                for d in P.DELTAS:
                    if d == 0:
                        Ycal = Yc_
                    else:
                        c = corr.apply_corruption(
                            Yc_, model="random_offset",
                            rng=np.random.RandomState(rs.randint(1 << 30)),
                            pc=pc, delta_deg=d, pool_labels=Y)
                        Ycal = c["corrupted_labels"]

                    rec = {"subject": s, "seed": seed, "pc": pc, "delta": float(d)}
                    for ax, nm in ((0, "pitch"), (1, "yaw")):
                        mc, sc = pred_c[ax]
                        me, se = pred_e[ax]
                        e_ev = Ye[:, ax] - me
                        e_cln = Yc_[:, ax] - mc
                        e_msr = Ycal[:, ax] - mc
                        r_cln = float(np.sqrt(np.mean(e_cln ** 2)))
                        r_msr = float(np.sqrt(np.mean(e_msr ** 2)))
                        scale = r_msr / r_cln if r_cln > 0 else 1.0
                        rec[f"scale_{nm}"] = scale
                        rec[f"clean_{nm}"] = cell_cpe(e_cln, sc, e_ev, se)
                        rec[f"scaled_{nm}"] = cell_cpe(e_cln * scale, sc, e_ev, se)
                        rec[f"measured_{nm}"] = cell_cpe(e_msr, sc, e_ev, se)
                    rows.append(rec)
        print(f"  {s}: {len(rows)} cells", flush=True)

    with open(OUT, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(rows)} records -> {os.path.basename(OUT)}")
    report(rows)
    return 0


def arm(r, k):
    return (r[f"{k}_pitch"] + r[f"{k}_yaw"]) / 2


def boot(rows, pc, key_a, key_b, n=4000):
    subj = sorted({r["subject"] for r in rows})
    by = {s: [r for r in rows if r["subject"] == s and r["pc"] == pc] for s in subj}
    rs, out = np.random.RandomState(0), []
    for _ in range(n):
        rr = [x for s in rs.choice(subj, len(subj), True) for x in by[s]]
        bel = [x for x in rr if x["delta"] in BELOW]
        out.append(np.mean([arm(x, key_a) for x in bel])
                   - np.mean([arm(x, key_b) for x in bel]))
    return np.percentile(out, [2.5, 97.5])


def report(rows):
    print("\nBelow-floor cells only (delta 1-5). CPE, calibrated branch.\n")
    print(f"{'pc':>5} {'scale':>7} {'clean':>9} {'scaled':>9} {'measured':>10} "
          f"{'meas-scaled':>13} {'95% CI':>22}")
    print("-" * 80)
    for pc in P.PCS:
        g = [r for r in rows if r["pc"] == pc and r["delta"] in BELOW]
        sc = np.mean([(r["scale_pitch"] + r["scale_yaw"]) / 2 for r in g])
        base = np.mean([arm(r, "clean") for r in rows
                        if r["pc"] == pc and r["delta"] == 0])
        vals = {k: np.mean([arm(r, k) for r in g]) for k in ARMS}
        lo, hi = boot(rows, pc, "measured", "scaled")
        print(f"{pc:>5} {sc:>7.3f} {base:>9.4f} {vals['scaled']:>9.4f} "
              f"{vals['measured']:>10.4f} {vals['measured']-vals['scaled']:>+13.4f} "
              f"[{lo:>+8.4f},{hi:>+8.4f}]")
    print("\n'scaled' is the pure-inflation prediction: what CPE would be if")
    print("corruption only rescaled the calibration residuals. The gap to")
    print("'measured' is what scale calibration cannot absorb.")


if __name__ == "__main__":
    sys.exit(main())
