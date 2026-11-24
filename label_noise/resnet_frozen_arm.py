#!/usr/bin/env python3
# resnet_frozen_arm.py — the cell that breaks the architecture/adaptation confound
"""THE PROBLEM THIS SOLVES

The UniGaze arm found a different curve SHAPE: R = floor@0.9 / floor@0.7 is
2.69 [2.30, 3.26] against the ResNet arm's 2.08 [1.82, 2.37], dR = +0.66,
p = 0.035. But that arm changed two things at once — architecture (ResNet-18 ->
ViT-H/14) AND adaptation depth (full fine-tune -> frozen backbone + ridge head).
So the finding currently cannot be attributed to either, and every statement of
it has to carry that caveat.

The missing cell is the fourth corner of a 2x2:

                          full fine-tune        frozen + ridge head
    ResNet-18             R = 2.08  (have)      R = ?          <- THIS SCRIPT
    UniGaze ViT-H/14      infeasible, ~4700h    R = 2.69 (have)

Read the answer off the new cell:
  R ~ 2.08  -> adaptation depth does NOT move the shape; the difference is
               ARCHITECTURE, and the confound is broken in favour of the
               interesting reading.
  R ~ 2.69  -> the shape difference is ADAPTATION DEPTH; UniGaze contributes
               nothing, and the paper must say the effect is about how deeply
               you adapt, not what you adapt.
  in between -> both contribute; report the decomposition, claim neither alone.

Every outcome is publishable and every outcome is informative. That is what
makes this worth running and the alternatives (a third architecture, more seeds)
not: they add precision to a quantity whose meaning is still ambiguous.

WHY IT IS CHEAP
The 15 per-subject LOPO checkpoints already exist at three accuracy levels in
models_sigma/{L0,L1,L2}. A frozen backbone needs no backward pass: the 9 support
images never change across the sweep, only their labels, so features are
extracted ONCE (45 forward passes of 9 frames through an 11M-param net at 112px)
and every adaptation afterwards is a closed-form ridge fit. Minutes, not hours.

PROTOCOL — IDENTICAL TO THE UNIGAZE ARM BY CONSTRUCTION
It imports residuals() and rng_for() from unigaze_arm rather than reimplementing
them, so the ridge solver, the LOO scheme and the SHA-256
(subject|model|delta|pc|seed) corruption draws are the same code on the same
grid. The corruption realizations are therefore IDENTICAL cell-for-cell to the
UniGaze arm's, which makes the two frozen arms a paired comparison and isolates
the feature extractor as the only difference between them.

L3 is excluded: its 0.7 crossing is censored on the delta<=10 grid the ResNet
sweep used, so it has no R to compare (logs/ARM_SHAPE_COMPARE.md).

SAFETY
Append-only and idempotent: existing (level, subject, delta, pc, seed, lam)
cells are skipped, the records file is backed up before writing, and nothing
under models/ or models_sigma/ is opened for writing at any point.

Usage:
  python3 resnet_frozen_arm.py --selftest     # stage 0 only, ~20 s
  python3 resnet_frozen_arm.py                # full arm
  python3 resnet_frozen_arm.py --lam-sweep    # robustness: lam in {1e-3..1}
"""
import argparse, json, os, shutil, sys, time
from datetime import datetime

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_AG = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import corruption as corr                                   # noqa: E402
import backbone_api as api                                  # noqa: E402
from unigaze_arm import K, residuals, rng_for, roc_auc, log  # noqa: E402

LEVELS = ["L0", "L1", "L2"]          # L3's 0.7 crossing is censored — no R
SUBJ = [f"p{i:02d}" for i in range(15)]
DELTAS = [1, 1.5, 2, 2.5, 3, 4, 4.5, 5, 5.25, 5.5, 5.75, 6, 6.5, 7,
          7.25, 7.5, 7.75, 8, 8.5, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26]
PCS = [0.1, 0.2, 0.4]
SEEDS = list(range(12))
LAMBDAS = [1e-3, 1e-2, 1e-1, 1.0]

CKPT_ROOT = os.path.join(_AG, "models_sigma")
FEAT_DIR = os.path.join(_HERE, os.environ.get("GAZE_RESNET_FEATURE_DIR",
                                              "resnet_frozen_features"))
OUT = os.path.join(_HERE, "resnet_frozen_records.jsonl")
CLEAN = os.path.join(_HERE, "resnet_frozen_clean_ehat.json")


# ------------------------------------------------------------------ features
def extract(level, sid, torch):
    """9 support frames -> (9, 512) penultimate features + (9, 2) pitch/yaw.

    The features are captured by hooking the INPUT of the gaze head during a
    normal api.predict() call, so the image preprocessing is whatever the
    validated pipeline already does — no transform is reimplemented here, which
    is where a silent scale bug would otherwise hide.
    """
    ck = os.path.join(CKPT_ROOT, level, f"backbone_mpiifacegaze_{sid}.pt")
    if not os.path.exists(ck):
        raise FileNotFoundError(f"missing checkpoint: {ck}")
    model = api.load_backbone(sid, ckpt_path=ck)

    head = None
    for _, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and m.out_features in (2, 3):
            head = m
    if head is None:
        raise RuntimeError("no gaze head (Linear with out_features in {2,3}) found")

    box = {}
    h = head.register_forward_hook(
        lambda mod, inp, out: box.__setitem__("f", inp[0].detach().cpu().numpy()))
    try:
        d = api.load_subject_support(sid, K=K)
        api.predict(model, d["images"])            # fires the hook
    finally:
        h.remove()
    F = np.asarray(box["f"], np.float64)
    if F.ndim != 2 or F.shape[0] != K:
        raise RuntimeError(f"unexpected feature shape {F.shape}; expected ({K}, D)")
    return F, np.asarray(d["labels"], np.float64), np.asarray(d["pool_labels"])


def cached(level, sid, torch):
    f = os.path.join(FEAT_DIR, level, f"{sid}.npz")
    if os.path.exists(f):
        z = np.load(f)
        return z["F"], z["Y"], z["P"]
    F, Y, P = extract(level, sid, torch)
    os.makedirs(os.path.dirname(f), exist_ok=True)
    np.savez_compressed(f, F=F.astype(np.float32), Y=Y, P=P)
    return F, Y, P


# ------------------------------------------------------------------ crossings
def crossing(ds, aa, target):
    for i in range(1, len(ds)):
        if aa[i] >= target > aa[i - 1]:
            if aa[-1] < target:
                return float("nan"), "unsustained"
            f = (target - aa[i - 1]) / (aa[i] - aa[i - 1])
            return ds[i - 1] + f * (ds[i] - ds[i - 1]), "ok"
    return float("nan"), ("censored_hi" if aa[-1] < target else "censored_lo")


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--lam-sweep", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    lams = LAMBDAS if a.lam_sweep else [a.lam]

    import torch  # noqa: F401  (needed by extract())

    # ---------------- STAGE 0 ----------------
    log("STAGE 0: self-test on L0/p00")
    F, Y, P = cached("L0", "p00", torch)
    log(f"  features {F.shape}  labels {Y.shape}  pool {P.shape}")
    res, loo = residuals(F, Y, a.lam)
    log(f"  resub residual mean {res.mean():.3f} deg   "
        f"LOO residual mean {loo.mean():.3f} deg")
    if not np.isfinite(loo).all():
        sys.exit("SELF-TEST FAILED: non-finite LOO residuals")
    if not (0.5 < loo.mean() < 45):
        sys.exit(f"SELF-TEST FAILED: LOO residual {loo.mean():.2f} deg is not a "
                 f"gaze signal — the hooked tensor is probably not the head input")
    c = corr.apply_corruption(Y, model="random_offset",
                              rng=rng_for("p00", "random_offset", 6, 0.2, 0),
                              pc=0.2, delta_deg=6, pool_labels=P)
    log(f"  corruption fires {int(np.sum(c['is_corrupted']))}/{K} frames")
    log("STAGE 0 PASSED\n")
    if a.selftest:
        return 0

    # ---------------- STAGE 1 — features + clean e_hat ----------------
    log(f"STAGE 1: features for {len(LEVELS)} levels x {len(SUBJ)} subjects")
    feats, clean = {}, {}
    t0 = time.time()
    for lv in LEVELS:
        clean[lv] = {}
        for s in SUBJ:
            F, Y, P = cached(lv, s, torch)
            feats[(lv, s)] = (F, Y, P)
            clean[lv][s] = float(np.mean(residuals(F, Y, a.lam)[1]))
        log(f"  {lv}: e_hat_clean = {np.mean(list(clean[lv].values())):.3f} deg "
            f"({time.time()-t0:.0f}s)")
    json.dump({"lam": a.lam, "per_level_per_subject": clean,
               "definition": "mean over subjects of the subject's mean LOO "
                             "residual under a ridge fit on TRUE labels"},
              open(CLEAN, "w"), indent=2)

    # ---------------- STAGE 2 — sweep (append-only) ----------------
    have = set()
    if os.path.exists(OUT) and os.path.getsize(OUT):
        for l in open(OUT):
            r = json.loads(l)
            have.add((r["level"], r["subject"], float(r["delta"]),
                      float(r["pc"]), int(r["seed"]), float(r["lam"])))
        bak = f"{OUT}.bak_{datetime.now():%Y%m%d_%H%M%S}"
        shutil.copy2(OUT, bak)
        log(f"\n{len(have)} existing cells; backed up -> {os.path.basename(bak)}")
    todo = [(lv, s, d, pc, sd, lm)
            for lv in LEVELS for s in SUBJ for d in DELTAS for pc in PCS
            for sd in SEEDS for lm in lams
            if (lv, s, float(d), float(pc), sd, float(lm)) not in have]
    log(f"STAGE 2: {len(todo)} new cells (closed-form)")
    if not todo:
        log("  nothing to add.")
    n, t0 = 0, time.time()
    with open(OUT, "a") as fh:                      # APPEND — never "w"
        for lv in LEVELS:
            for s in SUBJ:
                cells = [t for t in todo if t[0] == lv and t[1] == s]
                if not cells:
                    continue
                F, Y, P = feats[(lv, s)]
                for _, _, d, pc, sd, lm in cells:
                    c = corr.apply_corruption(
                        Y, model="random_offset",
                        rng=rng_for(s, "random_offset", d, pc, sd),
                        pc=pc, delta_deg=d, pool_labels=P)
                    _, loo = residuals(F, c["corrupted_labels"], lm)
                    for i in range(K):
                        fh.write(json.dumps({
                            "arch": "resnet18", "adaptation": "frozen+ridge",
                            "level": lv, "lam": lm, "subject": s, "seed": sd,
                            "model": "random_offset", "delta": float(d),
                            "pc": float(pc), "frame_idx": i,
                            "is_corrupted": bool(c["is_corrupted"][i]),
                            "actual_offset_deg": float(c["actual_offset_deg"][i]),
                            "loo_residual_deg": float(loo[i]),
                        }) + "\n")
                        n += 1
            fh.flush()
            log(f"  {lv} done ({n} records, {time.time()-t0:.0f}s)")

    # ---------------- STAGE 3 — R per level ----------------
    recs = [json.loads(l) for l in open(OUT)]
    recs = [r for r in recs if abs(r["lam"] - a.lam) < 1e-12
            and abs(r["pc"] - 0.2) < 1e-9]
    log("\nSTAGE 3: shape ratio R = floor@0.9 / floor@0.7   (lam=%g, pc=0.2)" % a.lam)
    log(f"  {'level':>6}{'e_hat':>9}{'d@0.7':>9}{'d@0.9':>9}{'R':>8}")
    for lv in LEVELS:
        rs = [r for r in recs if r["level"] == lv]
        ds = sorted({r["delta"] for r in rs})
        aa = []
        for d in ds:
            cell = [r for r in rs if r["delta"] == d]
            aa.append(roc_auc([x["loo_residual_deg"] for x in cell],
                              [x["is_corrupted"] for x in cell]))
        f7, s7 = crossing(ds, aa, 0.7)
        f9, s9 = crossing(ds, aa, 0.9)
        eh = np.mean(list(clean[lv].values())) if lv in clean else float("nan")
        R = f9 / f7 if (s7 == "ok" and s9 == "ok") else float("nan")
        log(f"  {lv:>6}{eh:>9.3f}{f7:>9.2f}{f9:>9.2f}{R:>8.3f}"
            + ("" if s7 == "ok" and s9 == "ok" else f"   ({s7}/{s9})"))
    log("\n  ResNet-18 FULL fine-tune : R = 1.98, 1.97, 2.14 (L0, L1, L2)")
    log("  UniGaze  frozen + ridge  : R = 2.69 [2.30, 3.26]")
    log("  -> R near 2.0 means the shape difference is ARCHITECTURE.")
    log("     R near 2.7 means it is ADAPTATION DEPTH.")
    log("     Do not eyeball it: run arm_shape_compare.py for the CIs.")
    log(f"\nrecords -> {os.path.basename(OUT)}")
    log("next: python3 arm_shape_compare.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
