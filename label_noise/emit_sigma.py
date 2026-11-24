# emit_sigma.py  (Prompt 1 corrected — σ diagnostic emission)
"""Emit clean (p_c=0) per-frame residuals on the DETECTION support
(calibration_split(n, K=9, seed=42)), under both backbone instances {s0, s1}.

  P1 (primary):  support LOO residual — adapt on the other 8 clean support
                 frames, score the held-out one. 9 per subject per backbone.
  P2 (secondary): held-out eval residual — adapt on all 9 clean support,
                 score every non-support frame.

Writes P1/P2 CSVs to <out> and prints the split-42 clean 15-fold aggregate MAE
(s0, P2-based) for the cross-process verification gate. Deterministic given a
fixed backbone (calibrate std=0.000), so two processes must match exactly.
"""
import argparse, csv, os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); _AG = os.path.dirname(_HERE)
for p in (_HERE, os.path.join(_AG, "scripts"), _AG):
    if p not in sys.path: sys.path.insert(0, p)
import torch
import gaze_math as gm, backbone_api as api, loo_residual as lr
import train_backbone as tb

SUBJ = [f"p{i:02d}" for i in range(15)]
BB = {"s0": os.path.join(_AG, "models"), "s1": os.path.join(_AG, "models", "seed1")}


def load_bb(path, device):
    m = tb.build_model(); m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.to(device).eval()


def eval_resid(model, images, labels, chunk=512):
    """Angular residual (deg) of `model` on `images` vs (pitch,yaw) `labels`, chunked."""
    lab_vecs = gm.pitchyaw_to_vector(np.asarray(labels).reshape(-1, 2))
    out = []
    for i in range(0, len(images), chunk):
        pv = api.predict(model, images[i:i + chunk])
        out.append(gm.angular_error_vec_deg(pv, lab_vecs[i:i + chunk]))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = api.get_device()
    p1, p2, agg = [], [], []
    for s in SUBJ:
        imgs, vecs = tb.load_images(s, None, np.random.default_rng(0))     # all frames, file order
        n = len(imgs)
        support_ids, _ = api.calibration_split(n, 9, seed=42)
        support_ids = np.asarray(support_ids)
        sup_imgs = imgs[support_ids]
        sup_lab = gm.vector_to_pitchyaw(vecs[support_ids]).reshape(-1, 2)   # clean labels
        em = np.ones(n, bool); em[support_ids] = False
        ev_imgs, ev_lab = imgs[em], gm.vector_to_pitchyaw(vecs[em]).reshape(-1, 2)
        for bname, bdir in BB.items():
            bb = load_bb(os.path.join(bdir, f"backbone_mpiifacegaze_{s}.pt"), device)
            r1 = lr.loo_residuals(bb, sup_imgs, sup_lab)                    # 9 clean support-LOO
            for i, v in enumerate(r1):
                p1.append((s, bname, int(support_ids[i]), float(v)))
            adapted = api.adapt(bb, sup_imgs, sup_lab)                      # personalize on 9 clean
            r2 = eval_resid(adapted, ev_imgs, ev_lab)                      # ~2991 eval residuals
            for i, v in enumerate(r2):
                p2.append((s, bname, i, float(v)))
            if bname == "s0":
                agg.append(float(np.mean(r2)))
        print(f"  {s} done  (eval frames={len(ev_imgs)})", flush=True)
    with open(os.path.join(a.out, "P1_support_loo.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["subject", "backbone", "frame_id", "residual_deg"]); w.writerows(p1)
    with open(os.path.join(a.out, "P2_eval.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["subject", "backbone", "eval_idx", "residual_deg"]); w.writerows(p2)
    print(f"SPLIT42_CLEAN_AGG_MAE_s0={np.mean(agg):.6f}")
    print(f"wrote P1 ({len(p1)}) + P2 ({len(p2)}) -> {a.out}")


if __name__ == "__main__":
    main()
