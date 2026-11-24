# exp_a_run.py
"""Experiment A sweep: six per-frame scorers over 15 subjects, deltas
{3,4,5,6,8,10} + structured, pc {0.1,0.2,0.4}. One JSONL record per support
frame with every scorer's score + is_corrupted + actual_offset, so the analysis
evaluates each exactly as the Phase-1 LOO residual was.

Backbones: seed-0 set in models/ (co-teaching peer A) and seed-1 set in
models/seed1/ (peer B). Corruption uses the fixed SHA-256 seeding. No experiment
code edited; reuses backbone_api / corruption / loo_residual / exp_a_scorers.
"""
import argparse, hashlib, json, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_AG = os.path.dirname(_HERE)
for p in (_HERE, os.path.join(_AG, "scripts"), _AG):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import gaze_math as gm
import backbone_api as api
import corruption as corr
import exp_a_scorers as sc
import train_backbone as tb

DELTAS = [3, 4, 5, 6, 8, 10]
PCS = [0.1, 0.2, 0.4]
K = 9


def load_bb(path, device):
    m = tb.build_model(); m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.to(device).eval()


def features(model, images):
    """512-d penultimate (avgpool) features for the support frames (SUGE)."""
    device = next(model.parameters()).device
    grabbed = {}
    def hook(m, i, o):                     # must return None, else it replaces the output
        grabbed["f"] = o.detach().flatten(1).cpu().numpy()
    h = model.avgpool.register_forward_hook(hook)
    with torch.no_grad():
        model(tb.to_batch(np.asarray(images), device))
    h.remove()
    return grabbed["f"]


def _rng(subject, model, delta, pc, seed):
    key = f"{subject}|{model}|{delta}|{pc}|{seed}".encode()
    return np.random.RandomState(int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2 ** 32))


def cells(deltas, pcs):
    out = [("random_offset", d, pc) for d in deltas for pc in pcs]
    out += [("structured_swap", None, pc) for pc in pcs]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=[f"p{i:02d}" for i in range(15)])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--deltas", type=int, nargs="+", default=DELTAS)
    ap.add_argument("--pcs", type=float, nargs="+", default=PCS)
    ap.add_argument("--steps", type=int, default=api.DEFAULT_STEPS)
    ap.add_argument("--lr", type=float, default=api.DEFAULT_LR)
    ap.add_argument("--models", default=os.path.join(_AG, "models"))
    ap.add_argument("--seed1", default=os.path.join(_AG, "models", "seed1"))
    ap.add_argument("--out", default=os.path.join(_HERE, "exp_a_records.jsonl"))
    a = ap.parse_args()
    device = api.get_device()
    print(f"[exp A] device={device} subjects={len(a.subjects)} seeds={a.seeds} "
          f"deltas={a.deltas} pcs={a.pcs}", flush=True)

    n_written = 0
    with open(a.out, "w") as f:
        for s in a.subjects:
            ba = load_bb(os.path.join(a.models, f"backbone_mpiifacegaze_{s}.pt"), device)
            bb_path = os.path.join(a.seed1, f"backbone_mpiifacegaze_{s}.pt")
            bb = load_bb(bb_path, device) if os.path.exists(bb_path) else None
            sup = api.load_subject_support(s, K=K, seed=42)      # fixed 9-frame support
            images, labels, pool = sup["images"], sup["labels"], sup["pool_labels"]
            feats = features(ba, images)
            for seed in a.seeds:
                for model, delta, pc in cells(a.deltas, a.pcs):
                    c = corr.apply_corruption(labels, model=model, rng=_rng(s, model, delta, pc, seed),
                                              pc=pc, delta_deg=delta, pool_labels=pool)
                    obs, mask, off = c["corrupted_labels"], c["is_corrupted"], c["actual_offset_deg"]
                    sco = sc.score_all(ba, images, obs, model_b=bb, feats=feats,
                                       steps=a.steps, lr_=a.lr)
                    for i in range(K):
                        rec = {"subject": s, "seed": int(seed), "model": model,
                               "delta": (None if delta is None else int(delta)),
                               "pc": float(pc), "frame_idx": i,
                               "is_corrupted": bool(mask[i]), "actual_offset_deg": float(off[i]),
                               "dividemix_failed": bool(sco["_dividemix_failed"]),
                               "dividemix_reason": sco["_dividemix_reason"]}
                        for k in sc.SCORERS:
                            v = sco[k][i]
                            rec[f"score_{k}"] = (None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v))
                        f.write(json.dumps(rec) + "\n"); n_written += 1
                    f.flush()
            print(f"  [{s}] done ({n_written} records)", flush=True)
    print(f"[exp A] wrote {n_written} records -> {a.out}")


if __name__ == "__main__":
    main()
