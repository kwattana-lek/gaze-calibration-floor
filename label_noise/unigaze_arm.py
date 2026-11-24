#!/usr/bin/env python
# unigaze_arm.py — second-architecture arm of the detectability-floor sweep
"""Tests whether the model-relative floor constant c = delta_floor / e_hat
survives a change of ARCHITECTURE, not merely a change of accuracy.

WHY THIS SHAPE
--------------
The sigma sweep (L0-L3) varied the accuracy of one ResNet-18 and found the floor
proportional to e_hat across a 2.6x range. The remaining axis is architecture.
The only UniGaze checkpoint that has not seen MPIIFaceGaze is
unigaze_h14_cross_X (632M params); full fine-tuning it costs ~4700 h per level
and is not viable.

But a FROZEN backbone needs no backward pass at all. The support images never
change across the sweep — only their labels do — so each subject's 9 feature
vectors are computed ONCE and every subsequent adaptation is a closed-form ridge
fit on cached features. That turns an infeasible arm into a short one.

PROTOCOL DIFFERENCE — STATE THIS IN THE PAPER
The ResNet arm fine-tunes all parameters; this arm fits a linear head on a
frozen backbone. So it varies architecture AND adaptation depth together. It
tests whether the floor is model-relative under a different model and a
different (shallower) adaptation, which is a broader claim in one direction and
a weaker one in another. Do not present it as a like-for-like replication.

WHAT IT COMPUTES
  features   : hook the gaze head's INPUT, so whatever the backbone produces is
               captured without assuming a particular API
  adaptation : kernel ridge in dual form, w = X^T (X X^T + lambda I)^-1 y,
               exact for n=9 << d and instant
  residuals  : leave-one-out only. With 9 points in a 1280-d feature space the
               ridge interpolates the support exactly, so the RESUBSTITUTION
               residual is ~0 for every frame and carries no signal. That is a
               real consequence of the frozen-backbone protocol: the scorer this
               paper ships for the ResNet arm cannot be used here, and only LOO
               is meaningful. Verified on synthetic features before running.
  e_hat      : mean clean-frame LOO residual — same definition as the ResNet arm
  floor      : ROC-AUC of the LOO residual vs is_corrupted, per delta

Corruption uses label_noise/corruption.py and the same
SHA-256(subject|model|delta|pc|seed) seeding as the ResNet sweep, so cells
correspond one-to-one even though the records are kept in a SEPARATE file and
must never be pooled.

SAFETY
  Stage 0 runs the entire pipeline on one subject and aborts the batch on any
  failure, so a bad assumption costs seconds rather than the whole run.
  Features are cached to disk per subject; re-running skips completed subjects.
  Writes only to unigaze_arm_records.jsonl and unigaze_features/.

Usage:  python unigaze_arm.py               # self-test, then full run
        python unigaze_arm.py --selftest    # stage 0 only
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import corruption as corr                     # noqa: E402
import gaze_math as gm                        # noqa: E402

MODEL = "unigaze_h14_cross_X"
LEAKY = {"unigaze_b16_joint", "unigaze_l16_joint", "unigaze_h14_joint"}
K = 9
DELTAS = [1, 1.5, 2, 2.5, 3, 4, 4.5, 5, 6, 8, 10]
PCS = [0.2]
SEEDS = [0, 1, 2]
# Subject list and feature cache follow the selected dataset. Defaults are the
# previous hardcoded values, so every existing cache and record reproduces; a
# second dataset sets GAZE_SUBJECTS and GAZE_FEATURE_DIR so that extracting its
# features CANNOT overwrite the MPIIFaceGaze cache the manuscript rests on.
SUBJ = ([s for s in os.environ["GAZE_SUBJECTS"].split(",") if s]
        if os.environ.get("GAZE_SUBJECTS")
        else [f"p{i:02d}" for i in range(15)])
LAMBDAS = [1e-3, 1e-2, 1e-1, 1.0]             # ridge sensitivity
FACE = os.environ.get(
    "GAZE_DATA_ROOT",
    os.path.expanduser("~/Desktop/Documents Since 29-01-2569/Datasets/"
                       "MPIIFaceGaze_Normalized"))
FEAT_DIR = os.path.join(_HERE, os.environ.get("GAZE_FEATURE_DIR",
                                              "unigaze_features"))
# STAGE 2 opens OUT with "w". Pointing this script at a second dataset without
# also redirecting OUT would TRUNCATE the MPIIFaceGaze sweep records the
# manuscript rests on -- a cache guard on FEAT_DIR alone does not prevent that.
# So the dangerous combination is made impossible rather than merely documented:
# selecting a dataset requires naming a records file for it.
OUT = os.path.join(_HERE, os.environ.get("GAZE_RECORDS",
                                         "unigaze_arm_records.jsonl"))
def _guard_dataset_redirect():
    """Called from main() only -- other modules import roc_auc/K from here and
    must not be aborted by a GAZE_SUBJECTS left in the caller's environment."""
    if not os.environ.get("GAZE_SUBJECTS"):
        return
    if not os.environ.get("GAZE_RECORDS"):
        sys.exit("ABORT: GAZE_SUBJECTS is set (a non-default dataset) but "
                 "GAZE_RECORDS is not. STAGE 2 truncates the records file; running "
                 "like this would destroy unigaze_arm_records.jsonl. Set it, e.g.\n"
                 "  GAZE_RECORDS=rtgene_unigaze_arm_records.jsonl")
    if not os.environ.get("GAZE_FEATURE_DIR"):
        sys.exit("ABORT: GAZE_SUBJECTS is set but GAZE_FEATURE_DIR is not; the "
                 "second dataset's features would land in the MPIIFaceGaze cache.")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def dkey(d):
    f = float(d)
    return str(int(f)) if f.is_integer() else repr(f)


def rng_for(sid, model, delta, pc, seed):
    k = f"{sid}|{model}|{dkey(delta)}|{pc}|{seed}".encode()
    return np.random.RandomState(
        int.from_bytes(hashlib.sha256(k).digest()[:8], "big") % (2 ** 32))


# --------------------------------------------------------------- ridge head
def ridge_fit_predict(Xtr, Ytr, Xte, lam):
    """Kernel ridge in dual form. n << d, so this is exact and instant."""
    Xtr = np.asarray(Xtr, np.float64)
    Ytr = np.asarray(Ytr, np.float64)
    mu = Xtr.mean(0, keepdims=True)
    ym = Ytr.mean(0, keepdims=True)
    Xc, Yc = Xtr - mu, Ytr - ym
    G = Xc @ Xc.T
    alpha = np.linalg.solve(G + lam * np.eye(len(G)), Yc)
    return (np.asarray(Xte, np.float64) - mu) @ Xc.T @ alpha + ym


def residuals(F, labels, lam):
    """(resubstitution, leave-one-out) angular residuals in degrees."""
    n = len(F)
    res = ridge_fit_predict(F, labels, F, lam)
    loo = np.empty_like(res)
    for i in range(n):
        m = np.arange(n) != i
        loo[i] = ridge_fit_predict(F[m], labels[m], F[i:i + 1], lam)[0]
    # gaze_math.angular_error_deg takes (pitch, yaw) pairs and broadcasts.
    return (gm.angular_error_deg(res, labels),
            gm.angular_error_deg(loo, labels))


# ------------------------------------------------------------- feature pass
def build_extractor(model, torch):
    """Hook the gaze head's INPUT — robust to however the head is named."""
    head, best = None, -1
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and mod.out_features in (2, 3):
            depth = name.count(".")
            if depth > best:
                head, best = mod, depth
    if head is None:
        raise RuntimeError("no Linear(.,2|3) head found — inspect the model")
    box = {}
    head.register_forward_hook(lambda m, inp, out: box.__setitem__("f", inp[0].detach()))

    def extract(x):
        with torch.no_grad():
            model(x)
        f = box["f"]
        return f.reshape(f.shape[0], -1).float().cpu().numpy()
    return extract


def subject_features(sid, extract, torch, cap):
    from PIL import Image
    lab = os.path.join(FACE, "Label", f"{sid}.label")
    rows = []
    with open(lab) as fh:
        hdr = next(fh).split()
        gi = hdr.index("2DGaze")
        for line in fh:
            p = line.split()
            if len(p) > gi:
                rows.append((p[0], [float(v) for v in p[gi].split(",")]))
    idx = np.random.RandomState(42).permutation(len(rows))[:cap]
    paths = [rows[i][0] for i in idx]
    labels = np.array([rows[i][1] for i in idx])
    feats, B = [], 8
    for i in range(0, len(paths), B):
        arr = np.stack([np.asarray(Image.open(os.path.join(FACE, "Image", q))
                                   .convert("RGB")) for q in paths[i:i + B]])
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div(255)
        feats.append(extract(x.to(next(iter(_dev)))))
    return np.concatenate(feats), labels


_dev = []


def roc_auc(s, p):
    s = np.asarray(s, float); p = np.asarray(p, bool)
    ok = ~np.isnan(s); s, p = s[ok], p[ok]
    a, b = int(p.sum()), int((~p).sum())
    if a == 0 or b == 0:
        return float("nan")
    u, inv, c = np.unique(s, return_inverse=True, return_counts=True)
    cs = np.cumsum(c); av = (cs - c + cs + 1) / 2.0
    return float((av[inv][p].sum() - a * (a + 1) / 2) / (a * b))


# ------------------------------------------------------------------ driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cap", type=int, default=400,
                    help="frames per subject to featurise (support drawn from these)")
    ap.add_argument("--lam", type=float, default=1e-2)
    a = ap.parse_args()

    _guard_dataset_redirect()   # before torch: a misdirected run must cost nothing
    log("=== UniGaze arm ===")
    log(f"model {MODEL} (the only checkpoint not trained on MPIIFaceGaze)")
    assert MODEL not in LEAKY, "refusing a MPIIFaceGaze-contaminated checkpoint"

    import torch
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    _dev.append(dev)
    import unigaze
    log(f"torch {torch.__version__}  device {dev}")
    t = time.time()
    model = unigaze.load(MODEL, device=str(dev))
    model.eval()
    log(f"loaded in {time.time()-t:.0f}s, "
        f"{sum(p.numel() for p in model.parameters())/1e6:.0f}M params")
    extract = build_extractor(model, torch)
    os.makedirs(FEAT_DIR, exist_ok=True)

    # ---------------- STAGE 0 — prove the pipeline on one subject -----------
    probe = SUBJ[0]
    log(f"STAGE 0: self-test on {probe}")
    F, Y = subject_features(probe, extract, torch, cap=32)
    log(f"  features {F.shape}  labels {Y.shape}")
    if F.ndim != 2 or F.shape[0] != len(Y):
        sys.exit("SELF-TEST FAILED: feature/label shape mismatch")
    sup = np.arange(K)
    r_res, r_loo = residuals(F[sup], Y[sup], a.lam)
    log(f"  resub residual  mean {r_res.mean():.2f} deg")
    log(f"  LOO   residual  mean {r_loo.mean():.2f} deg")
    if not np.isfinite(r_loo).all():
        sys.exit("SELF-TEST FAILED: non-finite residuals")
    if r_loo.mean() > 45:
        sys.exit(f"SELF-TEST FAILED: LOO residual {r_loo.mean():.1f} deg is not "
                 f"a gaze signal — the hooked tensor is probably not the head input")
    c = corr.apply_corruption(Y[sup], model="random_offset",
                              rng=rng_for(probe, "random_offset", 6, 0.2, 0),
                              pc=0.2, delta_deg=6, pool_labels=Y)
    log(f"  corruption fires: {int(np.sum(c['is_corrupted']))}/9 frames")
    log("STAGE 0 PASSED\n")
    if a.selftest:
        return 0

    # ---------------- STAGE 1 — cache features ------------------------------
    log(f"STAGE 1: features for {len(SUBJ)} subjects (cap {a.cap})")
    for s in SUBJ:
        f = os.path.join(FEAT_DIR, f"{s}.npz")
        if os.path.exists(f):
            log(f"  {s}: cached")
            continue
        t = time.time()
        F, Y = subject_features(s, extract, torch, a.cap)
        np.savez_compressed(f, F=F.astype(np.float32), Y=Y)
        log(f"  {s}: {F.shape} in {time.time()-t:.0f}s")

    # ---------------- STAGE 2 — sweep ---------------------------------------
    log(f"\nSTAGE 2: sweep {len(SUBJ)}x{len(DELTAS)}x{len(SEEDS)} cells "
        f"(closed-form; fast)")
    n = 0
    with open(OUT, "w") as fh:
        for s in SUBJ:
            z = np.load(os.path.join(FEAT_DIR, f"{s}.npz"))
            F, Y = z["F"], z["Y"]
            sup = np.arange(K)
            Fs, Ys = F[sup], Y[sup]
            for lam in ([a.lam] if True else LAMBDAS):
                for seed in SEEDS:
                    for d in DELTAS:
                        for pc in PCS:
                            c = corr.apply_corruption(
                                Ys, model="random_offset",
                                rng=rng_for(s, "random_offset", d, pc, seed),
                                pc=pc, delta_deg=d, pool_labels=Y)
                            _, loo = residuals(Fs, c["corrupted_labels"], lam)
                            for i in range(K):
                                fh.write(json.dumps({
                                    "arch": "unigaze_h14_cross_X",
                                    "adaptation": "frozen+ridge",
                                    "lam": lam, "subject": s, "seed": seed,
                                    "model": "random_offset", "delta": float(d),
                                    "pc": float(pc), "frame_idx": i,
                                    "is_corrupted": bool(c["is_corrupted"][i]),
                                    "actual_offset_deg": float(c["actual_offset_deg"][i]),
                                    "loo_residual_deg": float(loo[i]),
                                }) + "\n")
                                n += 1
            log(f"  {s} done ({n} records)")

    # ---------------- STAGE 3 — floor ---------------------------------------
    recs = [json.loads(l) for l in open(OUT)]
    clean = [r["loo_residual_deg"] for r in recs if not r["is_corrupted"]]
    eh = float(np.mean(clean))
    log(f"\nSTAGE 3: e_hat = {eh:.3f} deg (mean clean-frame LOO residual)")
    log(f"  {'delta':>6}{'AUC':>9}")
    ds, aucs = [], []
    for d in DELTAS:
        cell = [(r["loo_residual_deg"], r["is_corrupted"])
                for r in recs if r["delta"] == d]
        au = roc_auc([x for x, _ in cell], [c for _, c in cell])
        ds.append(d); aucs.append(au)
        log(f"  {d:>6g}{au:>9.3f}")
    cross = None
    for (d0, a0), (d1, a1) in zip(zip(ds, aucs), list(zip(ds, aucs))[1:]):
        if a0 < 0.7 <= a1:
            cross = d0 + (0.7 - a0) * (d1 - d0) / (a1 - a0); break
    log("")
    if cross:
        log(f"  delta_floor@0.7 = {cross:.2f} deg     c = {cross/eh:.3f}")
        log(f"  ResNet-18 arm: c = 1.02-1.12 across L0-L2")
        log("  -> compare. c in that band supports invariance ACROSS ARCHITECTURE.")
    else:
        log("  no 0.7 crossing on the grid — floor is off-grid; report the curve.")
    log(f"\nrecords -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
