"""
Linchpin experiment: train a proper MPIIFaceGaze gaze backbone (leave-one-person-out)
and test whether per-subject calibration actually reduces error.

This replaces the frozen-FAZE encoder that capped at ~8 deg with no adaptation
benefit. We train a full CNN on N-1 subjects, evaluate cross-subject on the
held-out subject, then fine-tune on K=9 calibration frames of that subject and
re-evaluate. The gap between cross-subject and calibrated MAE is the number that
decides which paper is viable.

Usage (defaults are tuned for a MacBook-Air-sized run):
    PYTHONPATH=. python scripts/train_backbone.py --test_subject p00 --epochs 25
    PYTHONPATH=. python scripts/train_backbone.py --fast          # quick smoke test

Outputs a trained backbone to models/backbone_mpiifacegaze_<test_subject>.pt
"""
import os, sys, argparse, copy, time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '.')
from src.config import DEVICE

# Dataset selection. Defaults are exactly the previous hardcoded values, so every
# existing record and checkpoint reproduces unchanged; a second dataset is
# selected by environment variable rather than by editing this file:
#     GAZE_DATA_ROOT=/path/to/RTGENE_Normalized GAZE_SUBJECTS=s000,s001,... 
_DEFAULT_ROOT = os.path.join("data", "MPIIFaceGaze_Normalized")
ROOT = os.environ.get("GAZE_DATA_ROOT", _DEFAULT_ROOT)
ALL_SUBJECTS = ([s for s in os.environ["GAZE_SUBJECTS"].split(",") if s]
                if os.environ.get("GAZE_SUBJECTS")
                else [f'p{i:02d}' for i in range(15)])
SIGN = -1                       # confirmed gaze-sign for this GazeHub format
IMG  = 112                      # input resolution
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32)


# ── data ──────────────────────────────────────────────────────────
def load_rows(subj):
    lines = open(os.path.join(ROOT, "Label", subj + ".label")).read().strip().split("\n")
    h = lines[0].split(); gi, fi = h.index("3DGaze"), h.index("Face")
    out = []
    for ln in lines[1:]:
        p = ln.split()
        g = np.array([float(x) for x in p[gi].split(",")], dtype=np.float32)
        g = SIGN * g
        g = g / (np.linalg.norm(g) + 1e-8)
        out.append((p[fi], g))
    return out


def load_images(subj, cap, rng):
    rows = load_rows(subj)
    if cap and len(rows) > cap:
        idx = rng.choice(len(rows), size=cap, replace=False)
        rows = [rows[i] for i in idx]
    imgs = np.empty((len(rows), IMG, IMG, 3), np.uint8)
    labs = np.empty((len(rows), 3), np.float32)
    for i, (f, g) in enumerate(rows):
        im = cv2.imread(os.path.join(ROOT, "Image", f))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        imgs[i] = cv2.resize(im, (IMG, IMG), interpolation=cv2.INTER_AREA)
        labs[i] = g
    return imgs, labs


def to_batch(imgs_u8, device):
    x = imgs_u8.astype(np.float32) / 255.
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.ascontiguousarray(np.transpose(x, (0, 3, 1, 2)))
    return torch.from_numpy(x).to(device)


# ── model ─────────────────────────────────────────────────────────
def build_model():
    try:
        import torchvision
        try:
            m = torchvision.models.resnet18(weights="IMAGENET1K_V1")
            print("[model] resnet18 (ImageNet-pretrained)")
        except Exception:
            m = torchvision.models.resnet18(weights=None)
            print("[model] resnet18 (from scratch — no pretrained weights)")
        m.fc = nn.Linear(m.fc.in_features, 3)
        return m
    except Exception as e:
        print(f"[model] torchvision unavailable ({e}); using compact CNN")
        return CompactCNN()


class CompactCNN(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(i, o): return nn.Sequential(
            nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.f = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128),
                               blk(128, 256), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(256, 3)
    def forward(self, x): return self.fc(self.f(x).flatten(1))


def angular_loss(pred, tgt):
    p = F.normalize(pred, dim=1); t = F.normalize(tgt, dim=1)
    return torch.acos((p * t).sum(1).clamp(-1 + 1e-6, 1 - 1e-6)).mean()


def mae_deg(model, imgs_u8, labs, device, bs=256):
    model.eval(); errs = []
    with torch.no_grad():
        for i in range(0, len(imgs_u8), bs):
            x = to_batch(imgs_u8[i:i+bs], device)
            p = F.normalize(model(x), dim=1).cpu().numpy()
            l = labs[i:i+bs]
            l = l / (np.linalg.norm(l, axis=1, keepdims=True) + 1e-8)
            d = np.clip((p * l).sum(1), -1, 1)
            errs.append(np.degrees(np.arccos(d)))
    return float(np.concatenate(errs).mean())


# ── main ──────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_subject', default='p00')
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--train_cap', type=int, default=1500, help='frames/subject')
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--k', type=int, default=9, help='calibration frames')
    ap.add_argument('--fast', action='store_true')
    args = ap.parse_args()
    if args.fast:
        args.epochs, args.train_cap = 5, 400

    rng = np.random.default_rng(0)
    train_subj = [s for s in ALL_SUBJECTS if s != args.test_subject]
    print(f"Device={DEVICE}  test={args.test_subject}  train={len(train_subj)} subj\n")

    print("Loading training images...")
    Xtr, Ytr = [], []
    for s in train_subj:
        xi, yi = load_images(s, args.train_cap, rng)
        Xtr.append(xi); Ytr.append(yi)
        print(f"  {s}: {len(xi)}")
    Xtr = np.concatenate(Xtr); Ytr = np.concatenate(Ytr)
    print(f"Total train: {len(Xtr)} frames")

    print("Loading held-out subject...")
    Xte, Yte = load_images(args.test_subject, None, rng)
    print(f"  {args.test_subject}: {len(Xte)} frames\n")

    model = build_model().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    n = len(Xtr)
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); order = rng.permutation(n); losses = []
        for i in range(0, n, args.bs):
            idx = order[i:i+args.bs]
            x = to_batch(Xtr[idx], DEVICE)
            y = torch.from_numpy(Ytr[idx]).to(DEVICE)
            opt.zero_grad(); loss = angular_loss(model(x), y); loss.backward(); opt.step()
            losses.append(loss.item())
        sched.step()
        te = mae_deg(model, Xte, Yte, DEVICE) if (ep % 5 == 0 or ep == args.epochs-1) else None
        msg = f"epoch {ep:2d}  train {np.degrees(np.mean(losses)):5.2f}  ({time.time()-t0:.0f}s)"
        if te is not None: msg += f"   held-out MAE {te:.2f} deg"
        print(msg)

    cross = mae_deg(model, Xte, Yte, DEVICE)
    os.makedirs('models', exist_ok=True)
    ckpt = f"models/backbone_mpiifacegaze_{args.test_subject}.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"\nSaved backbone -> {ckpt}")

    # ── per-subject calibration ──
    # split held-out subject into calibration (K) and eval (rest), disjoint
    perm = rng.permutation(len(Xte))
    cal_idx, ev_idx = perm[:args.k], perm[args.k:]
    Xc, Yc = Xte[cal_idx], Yte[cal_idx]
    Xe, Ye = Xte[ev_idx], Yte[ev_idx]
    base_eval = mae_deg(model, Xe, Ye, DEVICE)

    def calibrate(mode, lr, steps):
        m = copy.deepcopy(model)
        m.eval()                      # CRITICAL: keep BatchNorm on running stats for K=9
        if mode == 'fc':
            for p in m.parameters(): p.requires_grad = False
            for p in m.fc.parameters(): p.requires_grad = True
        else:
            for p in m.parameters(): p.requires_grad = True
        params = [p for p in m.parameters() if p.requires_grad]
        o = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
        xc = to_batch(Xc, DEVICE); yc = torch.from_numpy(Yc).to(DEVICE)
        for _ in range(steps):
            o.zero_grad(); angular_loss(m(xc), yc).backward(); o.step()
        return mae_deg(m, Xe, Ye, DEVICE)

    print("\n================ LINCHPIN RESULT ================")
    print(f"Cross-subject MAE (all held-out frames): {cross:.2f} deg")
    print(f"Base MAE on eval split (no calibration): {base_eval:.2f} deg")
    print(f"\nCalibration sweep (BatchNorm frozen, K={args.k}):")
    best = (base_eval, 'none')
    for mode, lr, steps in [('fc', 5e-4, 20), ('fc', 5e-4, 40), ('fc', 1e-3, 30),
                            ('full', 1e-5, 20), ('full', 3e-5, 20)]:
        c = calibrate(mode, lr, steps)
        tag = f"{mode} lr={lr:.0e} steps={steps}"
        print(f"  {tag:24s} -> {c:5.2f} deg   (gain {base_eval-c:+.2f})")
        if c < best[0]: best = (c, tag)
    print(f"\nBest calibrated: {best[0]:.2f} deg via [{best[1]}]  "
          f"(gain {base_eval-best[0]:+.2f})")
    print("=================================================")
    print("If best calibration gain is ~0.5-1.5 deg -> personalization works, full paper viable.")
    print("If gains stay <=0 even with BN frozen -> base model too strong to personalize;")
    print("   pivot to the CQS-detection + robustness paper.")


if __name__ == '__main__':
    main()
