"""
Learned calibration-quality detector — the reframed contribution.

Trains a ResNet-18 to classify a calibration frame as clean vs corrupted, and
evaluates it honestly against the hand-crafted CQS that fails:

  A. SUBJECT SPLIT   : train on 12 subjects, test on 3 held-out subjects.
                       Reports overall + per-mode AUC, and the CQS baseline on
                       the same test set (the contrast that carries the paper).
  B. LEAVE-ONE-CORRUPTION-OUT (--loco): train with 5 corruption types, test
                       detection on the 6th UNSEEN type. The rigor check that the
                       detector generalizes to novel corruption, not memorizes it.

Corruption is the independent full-face-correct simulator from cqs_detection.py;
sigma_max for the CQS baseline is calibrated on clean frames only.

Run from repo root:
    PYTHONPATH=. python scripts/train_detector.py --fast
    PYTHONPATH=. python scripts/train_detector.py --n 250 --epochs 8
    PYTHONPATH=. python scripts/train_detector.py --n 250 --epochs 8 --loco
"""
import os, sys, argparse, copy, time
import numpy as np
import cv2
import torch
import torch.nn as nn

sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
from src.config import CQS_W1, CQS_W2, CQS_W3, DEVICE
from cqs_detection import (face_paths, load_img, landmarks_xy, eye_bbox,
                           head_pose, corrupt, roc_auc, pr_auc)
from src.cqs import q_landmark, q_sharpness, q_head_pose

SUBJECTS = [f'p{i:02d}' for i in range(15)]
TEST = ['p00', 'p01', 'p02']
MODES = ['motion_blur', 'blink', 'occlusion', 'illumination', 'head_yaw', 'head_pitch']
IMG = 112
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_batch(u8, dev):
    x = u8.astype(np.float32) / 255.
    x = (x - MEAN) / STD
    x = np.ascontiguousarray(np.transpose(x, (0, 3, 1, 2)))
    return torch.from_numpy(x).to(dev)


def build_resnet():
    import torchvision
    try: m = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    except Exception: m = torchvision.models.resnet18(weights=None)
    m.fc = nn.Sequential(nn.Linear(m.fc.in_features, 1), nn.Flatten(0))
    return m


def subscores(image_rgb, sigma_max, pts):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    qlm = q_landmark(image_rgb)
    qsh = q_sharpness(gray, eye_bbox(pts, image_rgb.shape), sigma_max=sigma_max)
    p, y, r = head_pose(image_rgb, pts); qhp = q_head_pose(p, y, r)
    return qlm, qsh, qhp


def build_dataset(n, rng):
    """One clean + one corrupted (random mode) per sampled frame."""
    clean = {s: [] for s in SUBJECTS}
    for s in SUBJECTS:
        paths = face_paths(s); idx = rng.choice(len(paths), size=min(n, len(paths)), replace=False)
        for j in idx:
            im = load_img(paths[j])
            if im is not None: clean[s].append((im, landmarks_xy(im)))
        print(f"  {s}: {len(clean[s])}")
    lap = []
    for s in SUBJECTS:
        for im, pts in clean[s]:
            g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY); x1, y1, x2, y2 = eye_bbox(pts, im.shape)
            if g[y1:y2, x1:x2].size: lap.append(cv2.Laplacian(g[y1:y2, x1:x2], cv2.CV_64F).var())
    sigma_max = float(np.percentile(lap, 25))
    print(f"\nsigma_max = {sigma_max:.1f}\n")

    imgs, y, modes, subs, Q = [], [], [], [], []
    for s in SUBJECTS:
        for im, pts in clean[s]:
            imgs.append(cv2.resize(im, (IMG, IMG))); y.append(0); modes.append('clean'); subs.append(s)
            Q.append(subscores(im, sigma_max, pts))
            ci, m = corrupt(im, pts, rng)
            imgs.append(cv2.resize(ci, (IMG, IMG))); y.append(1); modes.append(m); subs.append(s)
            Q.append(subscores(ci, sigma_max, landmarks_xy(ci)))
    return (np.stack(imgs), np.array(y, float), np.array(modes), np.array(subs),
            np.array(Q, float), sigma_max)


def train_detector(imgs, y, tr_idx, epochs, bs, rng):
    net = build_resnet().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(); yt = torch.tensor(y, dtype=torch.float32)
    for ep in range(epochs):
        net.train(); order = rng.permutation(tr_idx)
        for i in range(0, len(order), bs):
            b = order[i:i+bs]
            opt.zero_grad()
            out = net(to_batch(imgs[b], DEVICE))
            bce(out, yt[b].to(DEVICE)).backward(); opt.step()
    return net


def scores(net, imgs, idx, bs=256):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i:i+bs]
            out.append(torch.sigmoid(net(to_batch(imgs[b], DEVICE))).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=250)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--loco', action='store_true')
    ap.add_argument('--fast', action='store_true')
    a = ap.parse_args()
    if a.fast: a.n, a.epochs = 60, 4
    rng = np.random.default_rng(0)

    print("Building dataset (MediaPipe — a few minutes)...")
    imgs, y, modes, subs, Q, sigma_max = build_dataset(a.n, rng)
    qlm, qsh, qhp = Q.T
    cqs = qlm**CQS_W1 * qsh**CQS_W2 * qhp**CQS_W3
    print(f"Total samples: {len(y)}  ({int((y==0).sum())} clean / {int((y==1).sum())} corrupt)")

    # ── A. subject split ──
    tr = np.where(~np.isin(subs, TEST))[0]; te = np.where(np.isin(subs, TEST))[0]
    print(f"\nTraining detector (subject split: {len(tr)} train / {len(te)} test)...")
    t0 = time.time(); net = train_detector(imgs, y, tr, a.epochs, a.bs, rng)
    ds = scores(net, imgs, te); print(f"  trained in {time.time()-t0:.0f}s")

    print("\n================ SUBJECT-SPLIT DETECTION ================")
    print(f"  Learned ResNet-18 : ROC-AUC {roc_auc(ds, y[te]):.3f}   PR-AUC {pr_auc(ds, y[te]):.3f}")
    print(f"  Hand-crafted CQS  : ROC-AUC {roc_auc(1-cqs[te], y[te]):.3f}   (baseline that fails)")
    print("\n  Per-mode AUC (learned detector, clean vs that mode):")
    cln = te[y[te] == 0]
    for m in MODES:
        mm = te[(y[te] == 1) & (modes[te] == m)]
        sel = np.concatenate([cln, mm])
        sc = scores(net, imgs, sel); yl = y[sel]
        print(f"    {m:14s} n={len(mm):4d}  detector {roc_auc(sc, yl):.3f}   "
              f"cqs {roc_auc(1-cqs[sel], yl):.3f}")

    # ── B. leave-one-corruption-out ──
    if a.loco:
        print("\n================ LEAVE-ONE-CORRUPTION-OUT (generalization) ================")
        print("  train on 5 corruption types, test detection on the 6th UNSEEN type:")
        for h in MODES:
            tr_l = np.where((modes != h) & (modes != 'clean') | (modes == 'clean'))[0]
            tr_l = tr_l[np.isin(subs[tr_l], [s for s in SUBJECTS if s not in TEST])]
            tr_l = tr_l[(modes[tr_l] != h)]
            neth = train_detector(imgs, y, tr_l, max(4, a.epochs//2), a.bs, rng)
            cln = te[y[te] == 0]; mm = te[(y[te] == 1) & (modes[te] == h)]
            sel = np.concatenate([cln, mm])
            sc = scores(neth, imgs, sel)
            print(f"    held-out={h:14s} n={len(mm):4d}  unseen-mode AUC {roc_auc(sc, y[sel]):.3f}")
    print("=========================================================================")
    print("Learned detector AUC is the paper's core result; CQS columns are the failing baseline.")


if __name__ == '__main__':
    main()
