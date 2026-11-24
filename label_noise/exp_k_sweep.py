#!/usr/bin/env python3
# exp_k_sweep.py -- is the floor a few-shot phenomenon or a nine-shot one? (reviewer, minor)
"""Repeats the fine-tuned ResNet-18 L0 sweep at K = 5, 9 and 16 support frames with ONE
driver, so the three are comparable by construction. K = 9 is the control: it must
reproduce the L0 row of Table 5 (floor 3.84, e_hat 3.43) to within seed noise, which
verifies that the adaptation settings used here (backbone_api defaults: 20 steps,
lr 1e-5, BN frozen -- the Section-12 recipe) match those behind the recorded sweep.
Cell timing on this Mac: 5.9 s at K = 9 (20 steps); 150 steps gives the same residual
to two decimals at 6x the cost, so 20 is used.

Everything else is exp_sigma_sweep's: primitives, SHA-256 seeding (delta formatted as
run_phase1 formats it), the L0 checkpoints in models_sigma/L0, deltas 1..10, pc 0.2,
seeds 0-2. Corruption draws min_clean=1, so at K = 5 and pc = 0.2 exactly one frame is
corrupted per cell and at K = 16 four are.

APPEND-ONLY to sigma_sweep_records_K.jsonl; records carry "K" and sigma_level "K<K>".
Cells already present are skipped, so the run resumes after an interruption.

Usage:  python3 exp_k_sweep.py --K 5 9 16
"""
import argparse, json, os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); _AG = os.path.dirname(_HERE)
for p in (_HERE, os.path.join(_AG, "scripts"), _AG):
    if p not in sys.path: sys.path.insert(0, p)
import exp_sigma_sweep as S                     # noqa: E402
import backbone_api as api                      # noqa: E402

OUT = os.path.join(_HERE, "sigma_sweep_records_K.jsonl")
MODELS = os.path.join(_AG, "models_sigma", "L0")
DELTAS = [1, 1.5, 2, 2.5, 3, 4, 4.5, 5, 6, 8, 10]
SEEDS = [0, 1, 2]; PC = 0.2

def done_keys():
    keys = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l); keys.add((r["K"], r["subject"], r["seed"], r["delta"]))
    return keys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, nargs="+", default=[5, 9, 16])
    ap.add_argument("--subjects", nargs="*", default=[f"p{i:02d}" for i in range(15)])
    a = ap.parse_args()
    have = done_keys(); n = 0
    with open(OUT, "a") as fh:
        for K in a.K:
            S.K = K                                            # the loader reads the module constant
            for sid in a.subjects:
                s = api.load_subject_support(sid, K=K, seed=SEEDS[0])
                subj = {"support_labels": s["labels"], "images": s["images"], "pool_labels": s["pool_labels"]}
                assert len(subj["images"]) == K
                bb = api.load_backbone(sid, ckpt_path=os.path.join(MODELS, f"backbone_mpiifacegaze_{sid}.pt"))
                for seed in SEEDS:
                    todo = [d for d in DELTAS if (K, sid, seed, float(d)) not in have]
                    for d in todo:
                        for r in S.score_cell(sid, subj, bb, seed, "random_offset", float(d), PC,
                                              api.DEFAULT_STEPS, api.DEFAULT_LR):
                            r["K"] = K; r["sigma_level"] = f"K{K}"
                            fh.write(json.dumps(r) + "\n"); n += 1
                    fh.flush()
                    print(f"  K={K} {sid} seed={seed}: {len(todo)} cells", flush=True)
    print(f"appended {n} records -> {OUT}")

if __name__ == "__main__":
    sys.exit(main())
