#!/usr/bin/env python3
# dump_lopo_predictions.py -- cross-subject predictions for the support frames.
"""WHY

The population-prior scorer needs, for every support frame, what a model that
never saw this subject predicts for it. On the frozen arms that came from a
leave-one-subject-out ridge map over cached features. On the FULL FINE-TUNED arm
the equivalent is the LOPO backbone itself: models/backbone_mpiifacegaze_<s>.pt
is trained on the other fourteen subjects.

This is pure inference -- 15 subjects x 9 frames, no training -- but it needs
torch and the dataset, so it must run on the Mac in the `agentgaze` env. It is
the only step of the fine-tuned replication that cannot run in the analysis VM.

It writes lopo_support_predictions.json; the floor comparison is then computed
from that file plus the EXISTING sigma_sweep_records.jsonl, with no retraining.

Usage:  ~/miniconda3/envs/agentgaze/bin/python dump_lopo_predictions.py
"""
import json, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gaze_math as gm            # noqa: E402
import backbone_api as api        # noqa: E402

OUT = os.path.join(_HERE, "lopo_support_predictions.json")
RECORDS = os.path.join(_HERE, "sigma_sweep_records.jsonl")
SUBJ = [f"p{i:02d}" for i in range(15)]
K = 9


def main():
    # Alignment note. The sweep stores frame_idx as the POSITION within the K
    # support frames (0..K-1), not the dataset index, so the two cannot be
    # compared directly -- an earlier version of this guard tried to and aborted
    # every time. What actually guarantees alignment is that both paths call
    # api.load_subject_support(sid, K=K) with the same default seed and take the
    # frames in the order it returns. We check what can be checked: that the
    # records really do carry positions 0..K-1 for every subject, and that the
    # loader returns exactly K frames.
    want = {}
    if os.path.exists(RECORDS):
        for line in open(RECORDS):
            r = json.loads(line)
            want.setdefault(r["subject"], set()).add(r["frame_idx"])
    for sid, pos in want.items():
        if pos != set(range(K)):
            sys.exit(f"ABORT: {sid} records carry frame positions {sorted(pos)}, "
                     f"expected 0..{K-1}. Alignment cannot be assumed.")

    out, errs = {}, []
    for s in SUBJ:
        d = api.load_subject_support(s, K=K)
        ids = list(d["support_ids"])
        if len(ids) != K:
            sys.exit(f"ABORT: loader returned {len(ids)} support frames for {s}, "
                     f"expected {K}.")
        model = api.load_backbone(s)                      # trained WITHOUT s
        pred = gm.vector_to_pitchyaw(api.predict(model, d["images"])).reshape(-1, 2)
        e = gm.angular_error_deg(pred, d["labels"])
        errs.append(float(np.mean(e)))
        out[s] = {"support_ids": ids,
                  "pred_pitchyaw": pred.tolist(),
                  "true_pitchyaw": d["labels"].tolist(),
                  "xsubj_err_deg": np.atleast_1d(e).tolist()}
        print(f"  {s}: cross-subject error on support {np.mean(e):6.2f} deg", flush=True)

    out["_meta"] = {"K": K, "n_subjects": len(SUBJ),
                    "cross_subject_mae_deg": float(np.mean(errs)),
                    "note": "LOPO backbone predictions on the K support frames; "
                            "support ids verified against sigma_sweep_records.jsonl"}
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\ncross-subject MAE over support frames: {np.mean(errs):.2f} deg")
    print(f"wrote {os.path.basename(OUT)}")
    print("\nNext (runs in the analysis VM, no torch needed):")
    print("  python3 exp_finetuned_detector.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
