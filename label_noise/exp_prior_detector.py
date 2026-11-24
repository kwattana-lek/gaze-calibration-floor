#!/usr/bin/env python3
# exp_prior_detector.py -- per-frame detection, no oracle partition.
"""WHAT THIS FIXES

exp_prior_tiebreak.py asked: GIVEN the true partition, can a population prior say
which subset holds the correct labels? That establishes the information exists,
but it is not a method -- in deployment the partition is exactly what you do not
know.

Here every frame is scored on its own and thresholded, the way the deployed
detector already works (resubstitution residual at tau = 3.42 deg). Three
scorers, identical cells:

  resub  : |label - prediction of a ridge fit on all nine support labels|
           -- what the paper ships
  loo    : |label - leave-one-out prediction|  -- the expensive alternative
  prior  : |label - prediction of a leave-one-subject-out POPULATION map|
           -- proposed

THE PREDICTION. resub and loo read a frame against the mapping the other support
frames imply, so coherent corruption corrupts the reference itself and their
floors rise with kappa. The prior scores each frame against a mapping estimated
entirely off-subject, which no amount of coherence within these nine frames can
move -- so its floor should be FLAT in kappa and set instead by the
cross-subject error.

ON FAIRNESS. The prior uses information the residual scorers do not: fourteen
other subjects. That asymmetry is the thesis, not a confound -- the support set
provably cannot settle the question alone. The information is also free in
practice, since a personalization pipeline already starts from a cross-subject
model.

WRITES only its own summary.
"""
import argparse, json, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gaze_math as gm                                  # noqa: E402
import corruption_temporal as ct                        # noqa: E402
import exp_coherence as E                               # noqa: E402
import exp_prior_tiebreak as PT                         # noqa: E402
from unigaze_arm import SUBJ, K, residuals, roc_auc, FEAT_DIR   # noqa: E402

try:
    from prepare_rtgene_labels import REPEAT_SESSIONS
except Exception:
    REPEAT_SESSIONS = {}

OUT = os.path.join(_HERE, "prior_detector_summary.json")


def resub_functional(F, lam):
    """Fixed linear map from all n labels to the in-sample prediction.
    Same centring correction as exp_coherence.loo_functionals."""
    n = len(F)
    mu = F.mean(0, keepdims=True)
    Xc = F - mu
    G = Xc @ Xc.T
    A = G + lam * np.eye(n)
    B = np.linalg.solve(A, Xc @ Xc.T)          # rows: b_i for each x_i
    return B + (1.0 - B.sum(1, keepdims=True)) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--pop-lam", type=float, default=1.0)
    ap.add_argument("--pc", type=float, default=0.2)
    ap.add_argument("--locality", default="clustered")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--feat", default=None,
                    help="feature cache dir; defaults to the dataset unigaze_arm selected")
    ap.add_argument("--subj", default=None,
                    help="comma-separated subject ids; defaults to unigaze_arm.SUBJ")
    ap.add_argument("--label", default="MPIIFaceGaze")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--allow-repeat-sessions", action="store_true",
                    help=argparse.SUPPRESS)
    a = ap.parse_args()

    subj = ([x for x in a.subj.split(",") if x] if a.subj else list(SUBJ))
    feat_dir = os.path.join(_HERE, a.feat) if a.feat else FEAT_DIR

    # The population map is LEAVE-ONE-SUBJECT-OUT. RT-GENE session ids are not
    # person ids -- s014 repeats s002 and s015 repeats s005 (Zenodo doi
    # 10.5281/zenodo.2529036) -- so holding out s014 while s002 is still in the
    # map leaks the held-out PERSON into their own "cross-subject" reference.
    # That inflates the prior scorer specifically, which is the proposed method,
    # so it is refused rather than warned about.
    dup = [x for x in subj if x in REPEAT_SESSIONS and REPEAT_SESSIONS[x] in subj]
    if dup and not a.allow_repeat_sessions:
        sys.exit("ABORT: " + ", ".join(f"{x} repeats {REPEAT_SESSIONS[x]}" for x in dup)
                 + ". A leave-one-subject-out map built over these leaks the held-out "
                   "person into their own reference and inflates the prior scorer. "
                   "Pass the unique-participant list (see participants.json).")

    print(f"dataset: {a.label} -- {len(subj)} subjects from "
          f"{os.path.basename(feat_dir)}\n")

    W, full = PT.population_maps(a.pop_lam, feat_dir=feat_dir, subjects=subj)
    feats = {s: (full[s][0][:K], full[s][1][:K]) for s in subj}
    loo_f = {s: E.loo_functionals(feats[s][0], a.lam) for s in subj}
    res_f = {s: resub_functional(feats[s][0], a.lam) for s in subj}
    pop_pred = {s: PT.predict(W[s], feats[s][0]) for s in subj}

    # exactness guard, both hat matrices, against the arm's own implementation
    s0 = subj[0]; F0, Y0 = feats[s0]
    ref_r, ref_l = residuals(F0, Y0, a.lam)
    d_r = float(np.max(np.abs(ref_r - gm.angular_error_deg(res_f[s0] @ Y0, Y0))))
    d_l = float(np.max(np.abs(ref_l - E.loo_resid(loo_f[s0], Y0))))
    print(f"exactness vs unigaze_arm.residuals:  resub {d_r:.2e}   loo {d_l:.2e}")
    if max(d_r, d_l) > 1e-9:
        sys.exit("ABORT: hat matrices do not reproduce the arm.")
    xerr = float(np.mean([np.mean(gm.angular_error_deg(PT.predict(W[s], full[s][0]),
                                                       full[s][1])) for s in subj]))
    print(f"population map cross-subject MAE: {xerr:.2f} deg\n")

    GRID = E.DELTAS_BASE + E.DELTAS_EXT
    SCORERS = ("resub", "loo", "prior")
    out = {"cross_subject_mae": xerr, "pc": a.pc, "locality": a.locality,
           "label": a.label, "subjects": subj, "n": len(subj),
           "feat_dir": os.path.basename(feat_dir)}

    print(f"Per-frame detection floor (delta at ROC-AUC = 0.7), p_c={a.pc:g}, "
          f"{a.locality}\n")
    # "spread" is max-min and is therefore always positive, which hides whether a
    # scorer's floor RISES or FALLS with coherence. The signed end-to-end drift
    # k=1 minus k=0 is the quantity the claim is about, so both are printed.
    print(f"{'scorer':>8}" + "".join(f"{'k='+str(k):>9}" for k in E.KAPPAS)
          + f"{'  spread':>10}{'   drift':>10}")
    print("-" * (8 + 9 * len(E.KAPPAS) + 20))
    for sc in SCORERS:
        floors = {}
        for kap in E.KAPPAS:
            aucs = []
            for d in GRID:
                v, p = [], []
                for s in subj:
                    F, Y = feats[s]
                    for seed in range(a.seeds):
                        c = ct.apply_coherent(Y, s, d, a.pc, seed, kap,
                                              locality=a.locality)
                        Yc = c["corrupted_labels"]
                        if sc == "resub":
                            r = gm.angular_error_deg(res_f[s] @ Yc, Yc)
                        elif sc == "loo":
                            r = E.loo_resid(loo_f[s], Yc)
                        else:
                            r = gm.angular_error_deg(Yc, pop_pred[s])
                        v.extend(np.atleast_1d(r).tolist())
                        p.extend(c["is_corrupted"].tolist())
                aucs.append(roc_auc(v, p))
            floors[kap] = E.crossing(GRID, aucs, 0.7)[0]
        vals = [floors[k] for k in E.KAPPAS]
        ok = [x for x in vals if x]
        spread = f"{max(ok)-min(ok):>+10.2f}" if len(ok) == len(vals) else f"{'--':>10}"
        f0, f1 = floors[E.KAPPAS[0]], floors[E.KAPPAS[-1]]
        drift = f"{f1-f0:>+10.2f}" if (f0 and f1) else f"{'--':>10}"
        print(f"{sc:>8}" + "".join(f"{v:>9.2f}" if v else f"{'none':>9}"
                                   for v in vals) + spread + drift)
        out[sc] = {str(k): floors[k] for k in E.KAPPAS}
        out[sc + "_drift"] = (f1 - f0) if (f0 and f1) else None
        out[sc + "_spread"] = (max(ok) - min(ok)) if len(ok) == len(vals) else None

    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwrote {os.path.basename(a.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
