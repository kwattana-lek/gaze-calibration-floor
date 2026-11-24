#!/usr/bin/env python
"""Prereq (c): reproduce detector_lopo's 0.9512 AND persist per-fold weights.

detector_lopo.py has no torch.save. This wrapper imports its primitives
(build_epoch_data / train / score) and re-runs the exact LOPO loop, saving each
fold's ResNet-18 state_dict. No edit to detector_lopo.py or any experiment logic.

Weights -> models/detectors/detector_lopo_s{seed}_{held}.pt
Results -> experiments/detector_lopo/results_regen_20260809.json
"""
import os, sys, json, argparse, time

AG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(AG, "scripts"))
sys.path.insert(0, AG)

import numpy as np
import torch
import detector_lopo as dl                       # reuses its functions unchanged
from cqs_detection import roc_auc, pr_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)        # detector_lopo defaults ->
    ap.add_argument("--seeds", type=int, default=3)      # 45 folds -> 0.9512
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--save-seeds", type=int, default=3, help="save weights for seeds < this")
    a = ap.parse_args()

    wdir = os.path.join(AG, "models", "detectors"); os.makedirs(wdir, exist_ok=True)
    outp = os.path.join(AG, "experiments", "detector_lopo", "results_regen_20260809.json")
    per_fold, t0 = [], time.time()

    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        print(f"=== seed {seed}: building data (MediaPipe) ===", flush=True)
        imgs, y, modes, subs, cqs = dl.build_epoch_data(a.n, rng)
        for held in dl.SUBJECTS:
            te = np.where(subs == held)[0]; tr = np.where(subs != held)[0]
            net = dl.train(imgs, y, tr, a.epochs, a.bs, rng)
            if seed < a.save_seeds:
                torch.save(net.state_dict(), os.path.join(wdir, f"detector_lopo_s{seed}_{held}.pt"))
            ds = dl.score(net, imgs, te)
            rec = {"seed": seed, "subject": held,
                   "det_roc": round(roc_auc(ds, y[te]), 4),
                   "det_pr": round(pr_auc(ds, y[te]), 4)}
            per_fold.append(rec)
            print(f"  [seed{seed} held={held}] det ROC {rec['det_roc']:.3f}", flush=True)
            json.dump(per_fold, open(outp, "w"), indent=2)   # incremental

    rocs = np.array([r["det_roc"] for r in per_fold], float)
    m, sd = rocs.mean(), rocs.std(ddof=1)
    ci = 1.96 * sd / np.sqrt(len(rocs))
    print(f"\n{len(rocs)} folds | mean det ROC-AUC = {m:.4f} +/- {ci:.4f} (95%) | expected 0.9512 "
          f"| |d|={abs(m-0.9512):.4f} -> {'reproduces' if abs(m-0.9512)<=0.01 else 'DIFFERS'}")
    print(f"weights saved for seeds < {a.save_seeds} -> {wdir}")
    print(f"results -> {outp}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
