#!/usr/bin/env python3
# exp_finetuned_detector.py -- the low-kappa ordering on a strongly personalized arm.
"""THE QUESTION

On the frozen arms the population prior beat both residual scorers at every
kappa, including kappa = 0. We attributed the kappa = 0 half to the arm rather
than the method: a population map with ~4.3 deg error is simply a better
predictor than a nine-point ridge on a frozen backbone, whose e_hat exceeds
6 deg. Under FULL FINE-TUNING e_hat is about 3.4 deg, so the personalized model
should win at low kappa and the two scorers should be complementary rather than
ranked. This measures that.

WHY IT NEEDS NO RETRAINING

The prior scorer never uses the personalized model. It compares a frame's label
against a leave-one-subject-out prediction, so its floor depends only on the
label, the LOPO prediction and delta -- all available from
lopo_support_predictions.json. The residual floors at kappa = 0 are already in
sigma_sweep_records.jsonl, whose corruption model (random_offset, independent
azimuths) IS kappa = 0.

WHAT IT DOES NOT ESTABLISH

The residual floors at kappa > 0 on this arm. Those need the sweep re-run with
coherent corruption, which requires fine-tuning per cell (~6.2 h per level on
MPS). The coherence effect itself is established on three other arms and is
mechanistic, so the expectation is that these rise too -- but this script does
not measure it, and the manuscript should not claim it.

Run dump_lopo_predictions.py on the Mac first. This script needs no torch.
"""
import json, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gaze_math as gm                  # noqa: E402
import corruption_temporal as ct        # noqa: E402
import exp_coherence as E               # noqa: E402
from unigaze_arm import roc_auc         # noqa: E402

PRED = os.path.join(_HERE, "lopo_support_predictions.json")
RECS = os.path.join(_HERE, "sigma_sweep_records.jsonl")
SEEDS, PC = 10, 0.2


def residual_floors():
    """kappa=0 floors for the fine-tuned arm, from the existing sweep."""
    rows = [json.loads(l) for l in open(RECS)]
    out = {}
    for lvl in sorted({r["sigma_level"] for r in rows}):
        g = [r for r in rows if r["sigma_level"] == lvl]
        ds = sorted({r["delta"] for r in g if r["delta"] > 0})
        for key, fld in (("resub", "source_residual_deg"), ("loo", "loo_residual_deg")):
            aucs = []
            for d in ds:
                c = [r for r in g if r["delta"] == d and r[fld] is not None]
                aucs.append(roc_auc([r[fld] for r in c],
                                    [r["is_corrupted"] for r in c]))
            out[(lvl, key)] = E.crossing(ds, aucs, 0.7)[0]
    return out


def main():
    if not os.path.exists(PRED):
        sys.exit(f"ABORT: {os.path.basename(PRED)} not found.\n"
                 "Run 'Run LOPO Prediction Dump.command' on the Mac first -- it "
                 "needs torch and the dataset, which this VM does not have.")
    P = json.load(open(PRED))
    meta = P.pop("_meta")
    subs = sorted(P)
    print(f"LOPO cross-subject MAE on support frames: "
          f"{meta['cross_subject_mae_deg']:.2f} deg\n")

    # --- prior scorer: needs only labels and LOPO predictions ---
    print("Population-prior floor on the fine-tuned arm (no retraining required)")
    print(f"{'kappa':>7} {'floor@0.7':>11}")
    grid = E.DELTAS_BASE + E.DELTAS_EXT
    prior = {}
    for kap in E.KAPPAS:
        aucs = []
        for d in grid:
            v, p = [], []
            for s in subs:
                Y = np.asarray(P[s]["true_pitchyaw"], float)
                pred = np.asarray(P[s]["pred_pitchyaw"], float)
                for seed in range(SEEDS):
                    c = ct.apply_coherent(Y, s, d, PC, seed, kap, locality="clustered")
                    v.extend(np.atleast_1d(
                        gm.angular_error_deg(c["corrupted_labels"], pred)).tolist())
                    p.extend(c["is_corrupted"].tolist())
            aucs.append(roc_auc(v, p))
        prior[kap] = E.crossing(grid, aucs, 0.7)[0]
        print(f"{kap:>7} {prior[kap]:>11.2f}" if prior[kap]
              else f"{kap:>7} {'none':>11}")

    # --- residual scorers at kappa = 0, from the existing sweep ---
    print(f"\nResidual floors at kappa=0, from sigma_sweep_records.jsonl")
    rf = residual_floors()
    print(f"{'level':>7} {'resub':>9} {'loo':>9} {'prior':>9}")
    for lvl in sorted({k[0] for k in rf}):
        r, l = rf.get((lvl, "resub")), rf.get((lvl, "loo"))
        print(f"{lvl:>7} " + " ".join(f"{x:>9.2f}" if x else f"{'none':>9}"
                                      for x in (r, l, prior[0.0])))

    print("\nReading: where the personalized model is strong, the residual "
          "scorers should win at kappa=0.\nThe prior's value is that its floor "
          "does not move with kappa -- not that it dominates everywhere.")
    json.dump({"prior_floor": {str(k): prior[k] for k in E.KAPPAS},
               "residual_floors_kappa0": {f"{a}|{b}": v for (a, b), v in rf.items()},
               "cross_subject_mae": meta["cross_subject_mae_deg"]},
              open(os.path.join(_HERE, "finetuned_detector_summary.json"), "w"),
              indent=2)
    print("\nwrote finetuned_detector_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
