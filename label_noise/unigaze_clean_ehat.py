#!/usr/bin/env python3
# unigaze_clean_ehat.py — measure e_hat on UNCORRUPTED support labels
"""Why this exists.

Both arms estimate the personalized error scale e_hat as the mean leave-one-out
residual over the CLEAN frames of the sweep (exp_sigma_sweep.e_hat, docstring:
"mean LOO residual over UNCORRUPTED support frames"). But a clean frame in a
sweep cell is not a frame from a clean fit: its LOO residual is produced by a
ridge/backbone adaptation whose OTHER support labels are corrupted, so the
estimate is contaminated by the very perturbation the sweep injects, and the
contamination grows with delta.

That was harmless while the two arms swept comparable grids. It stopped being
harmless when unigaze_arm_extend.py pushed this arm's grid to delta=20 to chase
the 0.9 crossing: e_hat rose from 6.427 to 6.924 deg purely because the grid got
longer, which deflates c = delta_floor / e_hat by about 7% relative to a ResNet
arm whose grid never went past 10. Comparing the two c values after that is an
apples-to-oranges comparison manufactured by my own grid extension — the same
class of artifact as the three truncation artifacts already found in this sweep.

So measure the uncontaminated scale directly. With a frozen backbone this is one
9-fold leave-one-out ridge fit per subject on the TRUE labels: no corruption, no
seeds, nothing to average over. Costs under a second.

Writes unigaze_clean_ehat.json:
  {"lam":..., "per_subject": {"p00": 4.93, ...}, "e_hat_clean": 6.13,
   "definition": "mean over subjects of the subject's mean clean-fit LOO residual"}

unigaze_arm_ci.py reads this file when present and reports c under all three
e_hat definitions, so the reader can see how much of any agreement or
disagreement with the ResNet arm is an artifact of the estimator.

Usage:  python3 unigaze_clean_ehat.py [--lam 0.01]
"""
import argparse, json, os, sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from unigaze_arm import K, SUBJ, FEAT_DIR, residuals   # noqa: E402

OUT = os.path.join(_HERE, "unigaze_clean_ehat.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=0.01)
    a = ap.parse_args()

    per = {}
    for s in SUBJ:
        f = os.path.join(FEAT_DIR, f"{s}.npz")
        if not os.path.exists(f):
            sys.exit(f"ABORT: no cached features for {s} — run unigaze_arm.py first.")
        z = np.load(f)
        F, Y = z["F"][:K], z["Y"][:K]
        res, loo = residuals(F, Y, a.lam)          # TRUE labels — no corruption
        per[s] = float(np.mean(loo))
        if not np.isfinite(per[s]):
            sys.exit(f"ABORT: non-finite clean LOO residual for {s}")
        print(f"  {s}: clean LOO {per[s]:6.3f} deg   (resub {np.mean(res):.3f})")

    e = float(np.mean(list(per.values())))
    json.dump({"lam": a.lam, "per_subject": per, "e_hat_clean": e,
               "definition": "mean over subjects of the subject's mean "
                             "leave-one-out residual under a ridge fit on the "
                             "TRUE (uncorrupted) support labels"},
              open(OUT, "w"), indent=2)
    print(f"\ne_hat_clean = {e:.3f} deg over {len(per)} subjects")
    print(f"wrote {os.path.basename(OUT)}")
    print("next: python3 unigaze_arm_ci.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
