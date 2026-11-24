#!/usr/bin/env python3
# exp_prior_tiebreak.py -- escaping the identifiability limit with a population prior
"""THE ARGUMENT

exp_identifiability.py shows that under coherent corruption the corrupted subset
is MORE internally self-consistent than the clean one (I < 1), so no residual
scorer can prefer the truth. But that impossibility is specific to scoring
residuals ON THE SUPPORT SET ALONE. It says the nine frames cannot settle the
question by themselves.

They do not have to. A coherently rotated cluster implies a ROTATED gaze
mapping, and a rotated mapping is anatomically implausible -- and fourteen other
subjects are available to say what plausible looks like.

  Residual tiebreak (provably fails at kappa=1):
      prefer the subset with the lower WITHIN-subset leave-one-out residual.
  Prior tiebreak (proposed):
      fit a leave-one-subject-out population map on the other 14 subjects'
      full 400 frames, and prefer the subset whose labels agree better with
      what that map predicts for its frames.

The prior tiebreak works when a subject's own deviation from the population map
is smaller than delta, so it should have its own floor -- near the cross-subject
error -- and that floor is reported alongside.

IMPLEMENTATION. The population map carries an unregularised bias column, which
makes the leave-one-subject-out fits an exact additive decomposition: accumulate
A_s = X^T X and B_s = X^T Y per subject once, then LOSO is (sum A) - A_s. No
centring, no approximation, no leakage of the held-out subject.

WRITES only its own summary.
"""
import argparse, json, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gaze_math as gm                # noqa: E402
import corruption_temporal as ct      # noqa: E402
import exp_coherence as E             # noqa: E402
from unigaze_arm import SUBJ, K, FEAT_DIR   # noqa: E402

OUT = os.path.join(_HERE, "prior_tiebreak_summary.json")
MIN_SUB = 3


def population_maps(lam, feat_dir=None, subjects=None):
    """Leave-one-subject-out ridge maps, exact via additive accumulation.

    feat_dir/subjects default to the dataset unigaze_arm selected (env-aware), so
    the MPIIFaceGaze behaviour is unchanged and a second corpus needs no edit."""
    subj = subjects or SUBJ
    root = feat_dir or FEAT_DIR
    d = None
    A_s, B_s, full = {}, {}, {}
    for s in subj:
        z = np.load(os.path.join(root, f"{s}.npz"))
        X, Y = z["F"].astype(np.float64), z["Y"].astype(np.float64)
        full[s] = (X, Y)
        Xa = np.hstack([X, np.ones((len(X), 1))])
        d = Xa.shape[1]
        A_s[s] = Xa.T @ Xa
        B_s[s] = Xa.T @ Y
    A_tot = sum(A_s.values()); B_tot = sum(B_s.values())
    reg = lam * np.eye(d); reg[-1, -1] = 0.0          # never regularise the bias
    W = {}
    for s in subj:
        W[s] = np.linalg.solve(A_tot - A_s[s] + reg, B_tot - B_s[s])
    return W, full


def predict(W_s, F):
    return np.hstack([F, np.ones((len(F), 1))]) @ W_s


def within(F, Y, lam):
    if len(F) < MIN_SUB:
        return None
    return float(np.mean(E.loo_resid(E.loo_functionals(F, lam), Y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop-lam", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--pc", type=float, default=0.4)
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()

    W, full = population_maps(a.pop_lam)
    xerr = float(np.mean([np.mean(gm.angular_error_deg(predict(W[s], full[s][0]),
                                                       full[s][1])) for s in SUBJ]))
    print(f"Leave-one-subject-out population map: cross-subject MAE {xerr:.2f} deg")
    print("That is the prior's own noise floor -- it cannot resolve a rotation "
          "much smaller than it.\n")
    print(f"Tiebreak accuracy (fraction of cells where the TRUE subset is preferred)")
    print(f"p_c={a.pc:g}, clustered locality, {a.seeds} seeds x {len(SUBJ)} subjects. "
          f"0.5 = chance.\n")

    out = {"cross_subject_mae": xerr}
    for kap in (0.0, 1.0):
        print(f"--- kappa = {kap:g} "
              f"{'(independent: residuals should work)' if kap==0 else '(coherent: residuals provably fail)'} ---")
        print(f"{'delta':>7} {'residual':>11} {'prior':>9} {'n':>7}")
        for d in (4, 6, 8, 10, 14, 20):
            res_ok, pri_ok, n = 0, 0, 0
            for s in SUBJ:
                F9, Y9 = full[s][0][:K], full[s][1][:K]
                pred = predict(W[s], F9)
                for seed in range(a.seeds):
                    c = ct.apply_coherent(Y9, s, d, a.pc, seed, kap,
                                          locality="clustered")
                    m = c["is_corrupted"]; Yc = c["corrupted_labels"]
                    wc = within(F9[~m], Y9[~m], a.lam)
                    wx = within(F9[m], Yc[m], a.lam)
                    if wc is None or wx is None:
                        continue
                    n += 1
                    res_ok += int(wc < wx)                      # residual tiebreak
                    eC = float(np.mean(gm.angular_error_deg(Y9[~m], pred[~m])))
                    eX = float(np.mean(gm.angular_error_deg(Yc[m], pred[m])))
                    pri_ok += int(eC < eX)                      # prior tiebreak
            if n:
                print(f"{d:>7} {res_ok/n:>11.3f} {pri_ok/n:>9.3f} {n:>7}")
                out[f"{kap}|{d}"] = {"residual": res_ok/n, "prior": pri_ok/n, "n": n}
        print()
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"wrote {os.path.basename(OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
