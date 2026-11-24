#!/usr/bin/env python3
# exp_transport.py -- does c(kappa) survive a change of capture condition?
"""THE QUESTION

Everything in the manuscript is measured on one dataset. The claim that the
detectability floor is model-relative but process-dependent has been established
across architectures and adaptation depths; whether it holds across CAPTURE
CONDITIONS is the decisive open experiment, and this runs it.

Two outcomes, both reportable, and the section is written to take either:

  c(kappa) lands in the same band  -> the constant and its kappa-dependence are
      properties of the detection problem, not of one corpus. The strongest
      version of the paper.
  it does not                      -> c is capture-condition-specific, which is a
      sharper limit on the published constant than anything in Section 6 and
      reframes the contribution around scope.

The two corpora are NOT preprocessed identically, and the section must say so.
MPIIFaceGaze_Normalized went through GazeHub's dpc.norm() virtual-camera
normalisation; RT-GENE ships its own face crops and GazeHub's RT-GENE script
applies no warp, so none was invented here. Both are 224x224 and both are resized
to 112 by the same loader, and prepare_rtgene.py verifies that the columns the
loader reads carry the same convention (GazeTo3d is checked against the reference
build's own 2DGaze/3DGaze columns to 1e-5 before any label is emitted).

That difference shifts ehat and delta_floor individually -- which is why the
transport claim is made about their RATIO c, invariant to a common rescaling of
the angular scale, and not about either quantity alone. Stated as a limitation
rather than gated away: a pipeline that forced RT-GENE through a normalisation it
does not have would be the larger confound.

RT-GENE SESSIONS ARE NOT PARTICIPANTS. The corpus has 17 sessions but 15 people:
two were recorded a second time (Zenodo doi 10.5281/zenodo.2529036). Residuals
are POOLED across subjects into one ROC per cell, so a person present twice is
weighted twice. Pass --subj-b the unique-participant list for the primary result
(prepare_rtgene_labels.py writes both lists to participants.json) and re-run with
all sessions and a second --out as a robustness check. Nothing here trains, so
there is no leakage; the concern is pooling weight alone.

Reads two cached feature sets; needs no torch and no retraining.

Usage:
  python3 exp_transport.py --feat-a unigaze_features        --subj-a p00,...
                           --feat-b rtgene_unigaze_features --subj-b s000,...
"""
import argparse, json, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import corruption_temporal as ct        # noqa: E402
import exp_coherence as E               # noqa: E402
from unigaze_arm import roc_auc, K      # noqa: E402

OUT = os.path.join(_HERE, "transport_summary.json")
LAM, PC, SEEDS = 1e-2, 0.2, 10
GRID = E.DELTAS_BASE + E.DELTAS_EXT


def load(feat_dir, subjects):
    d = os.path.join(_HERE, feat_dir)
    if not os.path.isdir(d):
        sys.exit(f"ABORT: no feature cache at {d}")
    out = {}
    for s in subjects:
        f = os.path.join(d, f"{s}.npz")
        if not os.path.exists(f):
            sys.exit(f"ABORT: {f} missing -- extract features for {s} first.")
        z = np.load(f)
        out[s] = (z["F"][:K].astype(np.float64), z["Y"][:K].astype(np.float64))
    return out


def arm(feats, label):
    subj = sorted(feats)
    funcs = {s: E.loo_functionals(feats[s][0], LAM) for s in subj}
    e_clean = float(np.mean([np.mean(E.loo_resid(funcs[s], feats[s][1]))
                             for s in subj]))
    cs, floors = {}, {}
    for kap in E.KAPPAS:
        aucs = []
        for d in GRID:
            v, p = [], []
            for s in subj:
                F, Y = feats[s]
                for sd in range(SEEDS):
                    c = ct.apply_coherent(Y, s, d, PC, sd, kap)
                    v.extend(np.atleast_1d(
                        E.loo_resid(funcs[s], c["corrupted_labels"])).tolist())
                    p.extend(c["is_corrupted"].tolist())
            aucs.append(roc_auc(v, p))
        fl = E.crossing(GRID, aucs, 0.7)[0]
        floors[kap] = fl
        cs[kap] = fl / e_clean if fl else None
    print(f"\n{label}: {len(subj)} subjects, clean-fit e_hat {e_clean:.2f} deg")
    print(f"  {'kappa':>7} {'floor':>8} {'c':>8}")
    for kap in E.KAPPAS:
        f_, c_ = floors[kap], cs[kap]
        print(f"  {kap:>7} {(f'{f_:.2f}' if f_ else '--'):>8} "
              f"{(f'{c_:.3f}' if c_ else '--'):>8}")
    return {"n": len(subj), "e_clean": e_clean,
            "floor": {str(k): floors[k] for k in E.KAPPAS},
            "c": {str(k): cs[k] for k in E.KAPPAS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-a", default="unigaze_features")
    ap.add_argument("--subj-a", default=",".join(f"p{i:02d}" for i in range(15)))
    ap.add_argument("--feat-b", default="rtgene_unigaze_features")
    ap.add_argument("--subj-b", required=True,
                    help="comma-separated RT-GENE session ids, e.g. s000,s001,...")
    ap.add_argument("--label-a", default="MPIIFaceGaze")
    ap.add_argument("--label-b", default="RT-GENE")
    ap.add_argument("--out", default=OUT,
                    help="summary json; give a second path for the robustness run "
                         "so the primary result is not overwritten")
    a = ap.parse_args()

    A = arm(load(a.feat_a, [s for s in a.subj_a.split(",") if s]), a.label_a)
    B = arm(load(a.feat_b, [s for s in a.subj_b.split(",") if s]), a.label_b)

    print(f"\n{'':>7} {'c ' + a.label_a:>18} {'c ' + a.label_b:>18} {'ratio':>8}")
    print("  " + "-" * 54)
    ok = []
    for kap in E.KAPPAS:
        ca, cb = A["c"][str(kap)], B["c"][str(kap)]
        r = cb / ca if (ca and cb) else None
        if r: ok.append(r)
        print(f"{kap:>7} {(f'{ca:.3f}' if ca else '--'):>18} "
              f"{(f'{cb:.3f}' if cb else '--'):>18} "
              f"{(f'{r:.2f}x' if r else '--'):>8}")
    if ok:
        print(f"\n  ratio across kappa: {min(ok):.2f}x .. {max(ok):.2f}x  "
              f"(mean {np.mean(ok):.2f}x)")
        print("\n  Reading: a ratio near 1 at every kappa means both the constant and")
        print("  its kappa-dependence transport. A ratio that is near 1 at kappa=0 but")
        print("  not at kappa=1 would mean the constant transports and its process")
        print("  dependence does not -- report that separately, it is a different claim.")
    json.dump({a.label_a: A, a.label_b: B,
               "subjects_a": [s for s in a.subj_a.split(",") if s],
               "subjects_b": [s for s in a.subj_b.split(",") if s]},
              open(a.out, "w"), indent=2)
    print(f"\nwrote {os.path.basename(a.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
