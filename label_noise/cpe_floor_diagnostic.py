#!/usr/bin/env python3
# cpe_floor_diagnostic.py -- how much of CPE is the probe's own floor?
"""WHY THIS EXISTS

unc_calib_probe.py --selftest reports calibrated CPE 0.076-0.102 on WELL-SPECIFIED
Gaussian data with CLEAN labels and no corruption anywhere. The real-data
clean-label baseline is 0.084. If those are the same number then 0.084 is not a
property of the gaze model, it is the probe's floor, and the +0.003..+0.014
corruption effect cannot be quoted as a fraction of it.

Two candidate causes, separated here by varying one at a time.

  (a) QGRID discretization. cpe_for_axis can only SELECT a z from QGRID --
      searchsorted returns an index, never an interpolated level. Calibrated
      coverage is therefore confined to a 10-point set spaced 0.10 apart over
      most of the range, so a miscalibrated model is rounded to a neighbour and
      pays up to half a grid step.

  (b) Sampling noise in cal_cov. At n_cal = 50 a coverage estimate near q = 0.5
      carries SE sqrt(0.25/50) = 0.071, comparable to the grid spacing itself,
      so the selected index is close to a coin flip between neighbours.

  Prediction if (a): CPE_cal falls as QGRID is refined at fixed n_cal.
  Prediction if (b): CPE_cal falls as n_cal grows at fixed QGRID.
  Neither falls  -> the floor is real miscalibration at n_fit = 9 and the
                    baseline stands.

WRITES NOTHING. This module only reads functions from unc_calib_probe; it never
calls its main(), which would truncate unc_calib_records.jsonl.

Usage:  python3 cpe_floor_diagnostic.py
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import unc_calib_probe as P  # noqa: E402  (import-only; main() is never called)

NEV = 400
REPS = 24
DIM = 200


def synth(rng, nfit, ncal, nev, d=DIM):
    """The self-test's generative model: linear-Gaussian, correctly specified,
    feature scale deliberately mismatched from label scale."""
    n = nfit + ncal + nev
    w = rng.randn(d, 2) * 0.02
    X = rng.randn(n, d) * 0.05
    Y = X @ w + rng.randn(n, 2) * 0.03
    return X, Y


def run(qgrid, ncal, nfit=9, reps=REPS):
    """Mean (raw, calibrated) CPE over `reps` independent draws."""
    old = P.QGRID
    P.QGRID = list(qgrid)
    out = []
    try:
        for r in range(reps):
            rng = np.random.RandomState(r)
            X, Y = synth(rng, nfit, ncal, NEV)
            Xf, Xc, Xe = X[:nfit], X[nfit:nfit + ncal], X[nfit + ncal:]
            Yf, Yc, Ye = Y[:nfit], Y[nfit:nfit + ncal], Y[nfit + ncal:]
            mux = Xf.mean(0, keepdims=True)
            for ax in (0, 1):
                ym = Yf[:, ax].mean()
                lam, sf2 = P.fit_axis(Xf - mux, Yf[:, ax] - ym)
                mc, vc = P.gp_axis(Xf - mux, Yf[:, ax] - ym, Xc - mux, lam, sf2)
                me, ve = P.gp_axis(Xf - mux, Yf[:, ax] - ym, Xe - mux, lam, sf2)
                out.append(P.cpe_for_axis(Yc[:, ax] - (mc + ym), np.sqrt(vc),
                                          Ye[:, ax] - (me + ym), np.sqrt(ve)))
    finally:
        P.QGRID = old
    a = np.asarray(out, float)
    return float(a[:, 0].mean()), float(a[:, 1].mean())


GRIDS = [
    ("as shipped (10 pts)", P.QGRID),
    ("step 0.05 (19)", list(np.round(np.arange(0.05, 1.00, 0.05), 4))),
    ("step 0.02 (49)", list(np.round(np.arange(0.02, 1.00, 0.02), 4))),
    ("step 0.01 (99)", list(np.round(np.arange(0.01, 1.00, 0.01), 4))),
]
NCALS = [50, 200, 1000, 5000]


def main():
    print("Calibrated CPE on clean, correctly-specified synthetic data.")
    print("Everything below SHOULD be ~0. n_fit = 9 (the deployed protocol).\n")
    print(f"{'QGRID':>22} | " + " ".join(f"n_cal={n:<6}" for n in NCALS))
    print("-" * (24 + 13 * len(NCALS)))
    for name, g in GRIDS:
        cells = [run(g, n)[1] for n in NCALS]
        print(f"{name:>22} | " + " ".join(f"{c:<12.4f}" for c in cells))

    print("\nRaw (uncalibrated) CPE, same draws -- the model's own uncertainty:")
    print(f"{'QGRID':>22} | " + " ".join(f"n_cal={n:<6}" for n in NCALS))
    print("-" * (24 + 13 * len(NCALS)))
    for name, g in GRIDS:
        cells = [run(g, n)[0] for n in NCALS]
        print(f"{name:>22} | " + " ".join(f"{c:<12.4f}" for c in cells))

    print("\nn_fit sweep at the finest grid and largest n_cal "
          "(is the residue just 9-point fitting?):")
    fine = GRIDS[-1][1]
    for nfit in (9, 20, 50, 100):
        raw, cal = run(fine, 5000, nfit=nfit)
        print(f"  n_fit={nfit:<4} raw {raw:.4f}   calibrated {cal:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
