#!/usr/bin/env python3
# unigaze_arm_extend.py — APPEND-ONLY extension of the UniGaze arm sweep
"""The first UniGaze run leaves three things unmeasured, and all three are free
now that the features are cached (STAGE 1's 30 minutes is already paid; the
sweep itself is closed-form and took about one second for 495 cells).

  1. THE 0.7 CROSSING IS INTERPOLATED ACROSS A 1-DEG GAP.
     Pooled AUC is 0.612 at delta=5 and 0.716 at delta=6; delta_floor = 5.84 is
     a straight-line reading between them, not an observation. Adding 5.25, 5.5,
     5.75 makes it a measurement — the same correction exp_b_finegrid.py applied
     to Experiment B's damage-ratio crossing.

  2. THE 0.9 CROSSING IS OFF THE GRID ENTIRELY.
     The ResNet arm reports BOTH floors (0.7 at 4.2-4.8 deg, 0.9 at 8.0-8.7).
     This arm reports only the first: AUC at the top of the grid (delta=10) is
     0.814. Proportionality would put the 0.9 crossing near 12.5 deg. Extending
     to 12/14/16/20 either finds it there — which would be a second, independent
     confirmation on a different architecture — or does not, which matters more.

  3. PER-SUBJECT AUCs ARE TOO NOISY TO READ.
     At 3 seeds x 9 frames x pc=0.2 a subject-delta cell holds about 6 corrupted
     and 21 clean frames; the standard error on that AUC is roughly 0.13. The
     per-subject c values from the first run span 0.212 to 1.122, which is
     consistent with pure noise and cannot support or refute a per-subject
     constant. Twelve seeds cut that error by half.

Also adds pc in {0.1, 0.4} alongside 0.2, because it costs nothing and tests on
this arm the claim Section 8 makes for the ResNet one: that the floor does not
depend on the deployment's own corruption rate. The headline analysis stays at
pc=0.2 for comparability — unigaze_arm_ci.py filters by pc.

SAFETY
  unigaze_arm.py opens the records file with "w" and would wipe it. This script
  APPENDS, backs the file up first, and skips every (subject, delta, pc, seed)
  key already present, so it is idempotent and can be re-run.

Cost: a few minutes at most; no GPU work, no forward passes, features are read
from unigaze_features/*.npz.

Usage:  python3 unigaze_arm_extend.py [--lam 0.01]
"""
import argparse, json, os, shutil, sys, time
from datetime import datetime

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import corruption as corr                       # noqa: E402
from unigaze_arm import (K, SUBJ, FEAT_DIR, OUT, residuals,  # noqa: E402
                         rng_for, log)

# full target grid — existing combinations are skipped, not regenerated
DELTAS = [1, 1.5, 2, 2.5, 3, 4, 4.5, 5, 5.25, 5.5, 5.75, 6, 6.5, 7,
          7.25, 7.5, 7.75, 8, 8.5, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26]
PCS = [0.1, 0.2, 0.4]
SEEDS = list(range(12))


LAMBDAS = [1e-3, 1e-2, 1e-1, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--lam-sweep", action="store_true",
                    help="also generate lam in {1e-3, 1e-2, 1e-1, 1} so the "
                         "ridge penalty can be compared against the ResNet "
                         "frozen arm, whose 512-d ReLU features have a "
                         "different Gram scale — a fixed lam is NOT the same "
                         "amount of regularization in the two feature spaces")
    a = ap.parse_args()
    lams = LAMBDAS if a.lam_sweep else [a.lam]

    if not os.path.exists(OUT):
        sys.exit(f"ABORT: {OUT} not found — run unigaze_arm.py first.")
    missing = [s for s in SUBJ
               if not os.path.exists(os.path.join(FEAT_DIR, f"{s}.npz"))]
    if missing:
        sys.exit(f"ABORT: no cached features for {missing}. Run unigaze_arm.py "
                 f"first; this script does no forward passes.")

    old = [json.loads(l) for l in open(OUT)]
    have = {(r["subject"], float(r["delta"]), float(r["pc"]), int(r["seed"]),
             float(r.get("lam", 0.01))) for r in old}
    todo = [(s, d, pc, sd, lm) for s in SUBJ for d in DELTAS for pc in PCS
            for sd in SEEDS for lm in lams
            if (s, float(d), float(pc), sd, float(lm)) not in have]
    if not todo:
        print("nothing to add — the grid is already complete.")
        return 0

    bak = f"{OUT}.bak_{datetime.now():%Y%m%d_%H%M%S}"
    shutil.copy2(OUT, bak)
    log(f"{len(old)} existing records ({len(have)} cells); backed up -> "
        f"{os.path.basename(bak)}")
    log(f"adding {len(todo)} new cells "
        f"({len(DELTAS)} deltas x {len(PCS)} pc x {len(SEEDS)} seeds x "
        f"{len(lams)} lam x {len(SUBJ)} subjects, minus what exists)")

    by_subj = {}
    for s, d, pc, sd, lm in todo:
        by_subj.setdefault(s, []).append((d, pc, sd, lm))

    n, t0 = 0, time.time()
    with open(OUT, "a") as fh:                   # APPEND — never "w"
        for s in SUBJ:
            cells = by_subj.get(s)
            if not cells:
                continue
            z = np.load(os.path.join(FEAT_DIR, f"{s}.npz"))
            F, Y = z["F"], z["Y"]
            Fs, Ys = F[:K], Y[:K]
            for d, pc, sd, lm in cells:
                c = corr.apply_corruption(
                    Ys, model="random_offset",
                    rng=rng_for(s, "random_offset", d, pc, sd),
                    pc=pc, delta_deg=d, pool_labels=Y)
                _, loo = residuals(Fs, c["corrupted_labels"], lm)
                for i in range(K):
                    fh.write(json.dumps({
                        "arch": "unigaze_h14_cross_X",
                        "adaptation": "frozen+ridge",
                        "lam": lm, "subject": s, "seed": sd,
                        "model": "random_offset", "delta": float(d),
                        "pc": float(pc), "frame_idx": i,
                        "is_corrupted": bool(c["is_corrupted"][i]),
                        "actual_offset_deg": float(c["actual_offset_deg"][i]),
                        "loo_residual_deg": float(loo[i]),
                    }) + "\n")
                    n += 1
            fh.flush()
            log(f"  {s}: +{len(cells)} cells ({n} records, "
                f"{time.time()-t0:.0f}s elapsed)")

    log(f"\nappended {n} records -> {os.path.basename(OUT)}")
    log("next: python3 unigaze_arm_ci.py            (pc=0.2, the headline)")
    log("      python3 unigaze_arm_ci.py --pc all   (pc-invariance check)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
