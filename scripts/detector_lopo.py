"""
Solidify the detector's headline number: full leave-one-person-out x seeds.

Trains the learned clean/corrupt detector with each subject held out in turn,
across several seeds, and aggregates ROC-AUC / PR-AUC (mean +/- 95% CI) plus
per-mode AUC and the hand-crafted CQS baseline on the same held-out sets.

Design notes
------------
* Images + sub-scores for all subjects are computed ONCE per seed (MediaPipe is
  the cost); each fold reuses them, holding out one subject for test.
* The CQS baseline uses clean-only-calibrated sigma_max, same as everywhere.
* results written incrementally to experiments/detector_lopo/results.json so the
  (long) run is safe to interrupt.

Run from repo root:
    PYTHONPATH=. python scripts/detector_lopo.py --fast
    PYTHONPATH=. python scripts/detector_lopo.py --n 200 --seeds 3 --epochs 8
"""
import os, sys, json, argparse, time
import numpy as np
import cv2
import torch
import torch.nn as nn

sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
from src.config import CQS_W1, CQS_W2, CQS_W3, DEVICE
from cqs_detection import (face_paths, load_img, landmarks_xy, eye_bbox,
                           head_pose, corrupt, roc_auc, pr_auc)
from src.cqs import q_landmark, q_sharpness, q_head_pose
from train_detector import build_resnet, to_batch

SUBJECTS = [f'p{i:02d}' for i in range(15)]
MODES = ['motion_blur', 'blink', 'occlusion', 'illumination', 'head_yaw', 'head_pitch']
OUT = 'experiments/detector_lopo'


def cqs_of(im, pts, sigma_max):
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    qlm = q_landmark(im); qsh = q_sharpness(g, eye_bbox(pts, im.shape), sigma_max=sigma_max)
    p, y, r = head_pose(im, pts); qhp = q_head_pose(p, y, r)
    return qlm**CQS_W1 * qsh**CQS_W2 * qhp**CQS_W3


def build_epoch_data(n, rng):
    """One clean + one corrupt per sampled frame; returns arrays incl subject/mode/CQS."""
    clean = {}
    for s in SUBJECTS:
        paths = face_paths(s); idx = rng.choice(len(paths), size=min(n, len(paths)), replace=False)
        clean[s] = []
        for j in idx:
            im = load_img(paths[j])
            if im is not None: clean[s].append((im, landmarks_xy(im)))
    lap = []
    for s in SUBJECTS:
        for im, pts in clean[s]:
            g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY); x1, y1, x2, y2 = eye_bbox(pts, im.shape)
            if g[y1:y2, x1:x2].size: lap.append(cv2.Laplacian(g[y1:y2, x1:x2], cv2.CV_64F).var())
    sigma_max = float(np.percentile(lap, 25))

    imgs, y, modes, subs, cqs = [], [], [], [], []
    for s in SUBJECTS:
        for im, pts in clean[s]:
            imgs.append(cv2.resize(im, (112, 112))); y.append(0); modes.append('clean'); subs.append(s)
            cqs.append(cqs_of(im, pts, sigma_max))
            ci, m = corrupt(im, pts, rng)
            imgs.append(cv2.resize(ci, (112, 112))); y.append(1); modes.append(m); subs.append(s)
            cqs.append(cqs_of(ci, landmarks_xy(ci), sigma_max))
    return (np.stack(imgs), np.array(y, float), np.array(modes),
            np.array(subs), np.array(cqs, float))


def train(imgs, y, tr_idx, epochs, bs, rng):
    net = build_resnet().to(DEVICE); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(); yt = torch.tensor(y, dtype=torch.float32)
    for _ in range(epochs):
        net.train(); order = rng.permutation(tr_idx)
        for i in range(0, len(order), bs):
            b = order[i:i+bs]; opt.zero_grad()
            bce(net(to_batch(imgs[b], DEVICE)), yt[b].to(DEVICE)).backward(); opt.step()
    net.eval(); return net


def score(net, imgs, idx, bs=256):
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            out.append(torch.sigmoid(net(to_batch(imgs[idx[i:i+bs]], DEVICE))).cpu().numpy())
    return np.concatenate(out)


def ci95(v):
    v = np.asarray(v, float); return 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--subjects', nargs='+', default=SUBJECTS)
    ap.add_argument('--fast', action='store_true')
    a = ap.parse_args()
    if a.fast: a.n, a.seeds, a.epochs, a.subjects = 60, 1, 4, ['p00', 'p01', 'p02']
    os.makedirs(OUT, exist_ok=True)

    per_fold = []   # dict per (seed, held-out subject)
    t_start = time.time()
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        print(f"\n=== seed {seed}: building data (MediaPipe) ===")
        imgs, y, modes, subs, cqs = build_epoch_data(a.n, rng)
        for held in a.subjects:
            te = np.where(subs == held)[0]; tr = np.where(subs != held)[0]
            net = train(imgs, y, tr, a.epochs, a.bs, rng)
            ds = score(net, imgs, te)
            rec = {'seed': seed, 'subject': held,
                   'det_roc': round(roc_auc(ds, y[te]), 4),
                   'det_pr': round(pr_auc(ds, y[te]), 4),
                   'cqs_roc': round(roc_auc(1 - cqs[te], y[te]), 4)}
            cln = te[y[te] == 0]
            for m in MODES:
                mm = te[(y[te] == 1) & (modes[te] == m)]
                sel = np.concatenate([cln, mm])
                rec[f'det_{m}'] = round(roc_auc(score(net, imgs, sel), y[sel]), 4)
            per_fold.append(rec)
            print(f"  [seed{seed} held={held}] det ROC {rec['det_roc']:.3f}  "
                  f"PR {rec['det_pr']:.3f}   cqs {rec['cqs_roc']:.3f}")
            with open(os.path.join(OUT, 'results.json'), 'w') as f:
                json.dump(per_fold, f, indent=2)

    # aggregate
    def col(k): return np.array([r[k] for r in per_fold], float)
    print("\n================ DETECTOR LOPO SUMMARY ================")
    print(f"folds: {len(per_fold)}  ({len(a.subjects)} subjects x {a.seeds} seeds)  "
          f"total {time.time()-t_start:.0f}s")
    for k, lab in [('det_roc', 'Detector ROC-AUC'), ('det_pr', 'Detector PR-AUC'),
                   ('cqs_roc', 'CQS ROC-AUC (baseline)')]:
        v = col(k); print(f"  {lab:24s} {v.mean():.3f} ± {ci95(v):.3f}")
    print("\n  Per-mode detector ROC-AUC:")
    for m in MODES:
        v = col(f'det_{m}'); print(f"    {m:14s} {v.mean():.3f} ± {ci95(v):.3f}")
    print("======================================================")
    print("Headline: Detector ROC-AUC (mean±95%CI) vs CQS baseline. Saved to "
          + os.path.join(OUT, 'results.json'))


if __name__ == '__main__':
    main()
