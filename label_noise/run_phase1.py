# run_phase1.py
"""
Phase-1 sweep driver (HANDOFF §4.3-4.5).

Produces one JSONL record PER SUPPORT FRAME (§5 schema):
    subject, seed, delta, pc, frame_idx, is_corrupted,
    actual_offset_deg, loo_residual_deg, source_residual_deg
plus: model ("random_offset" | "structured_swap") and
      loo_residual_purified_deg (§4.4, null if purification disabled).

Sweep (§4.1/§4.2/§4.4):
    * corruption models: random_offset (delta in {5,10,15,20,30}) and
      structured_swap (delta = null; severity is the realised donor offset).
    * contamination pc in {0.1, 0.2, 0.4}.
    * K = 9 support frames; subjects x seeds folds (default 15 x 3 = 45, §4.1).

Deliverable (§4.5): DETECTION AUC, not downstream MAE. After the sweep this
prints ROC-AUC per (model, delta, pc) using the LOO residual (and the source
residual) as the detector of is_corrupted, and writes a summary JSON. Full
bootstrap-CI + the structured-offset histogram + plots live in analyze_phase1.py
(§5), which consumes this JSONL.

Modes:
    --synthetic (default): SyntheticBackbone, no checkpoint / no images. Fast;
        validates the whole pipeline and the AUC logic.
    --real: load_backbone + load_subject_support against ColumbiaGaze. Slow
        (per-fold adaptation); intended for the actual feasibility numbers.
"""
import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gaze_math as gm
import backbone_api as api
import corruption as corr
import loo_residual as lr

DELTAS = [5, 10, 15, 20, 30]
PCS = [0.1, 0.2, 0.4]
K = 9


# ---------------------------------------------------------------------------
# Synthetic subject pools: plausible gaze manifold per subject.
# ---------------------------------------------------------------------------
def make_synthetic_subjects(n_subjects=15, pool_size=105, K=K, seed=0):
    """
    dict sid -> {support_labels (K,2) pitch,yaw truth, images (K,3) truth vectors,
                 pool_labels (pool_size,2)}. Support split uses the real
                 calibration_split so it matches the DAP K-shot machinery.
    """
    rng = np.random.RandomState(seed)
    subjects = {}
    for s in range(n_subjects):
        pool = np.stack([gm.random_unit_pitchyaw(rng) for _ in range(pool_size)], 0)
        sup_ids, _ = api.calibration_split(pool_size, K, seed=42)
        support = pool[sup_ids]
        subjects[f"S{s:02d}"] = {
            "support_labels": support,
            "images": gm.pitchyaw_to_vector(support),   # synthetic image = truth
            "pool_labels": pool,
        }
    return subjects


def load_real_subjects(cgd_root, subject_ids=None, K=K, seed=42):
    ids = subject_ids or sorted(
        d for d in os.listdir(cgd_root) if os.path.isdir(os.path.join(cgd_root, d))
    )
    subjects = {}
    for sid in ids:
        s = api.load_subject_support(sid, K=K, seed=seed, cgd_root=cgd_root)
        subjects[sid] = {
            "support_labels": s["labels"],
            "images": s["images"],
            "pool_labels": s["pool_labels"],
        }
    return subjects


# ---------------------------------------------------------------------------
# One (subject, seed, model, delta, pc) cell -> K frame records.
# ---------------------------------------------------------------------------
def score_cell(sid, subj, backbone, seed, model, delta, pc, steps, lr_, purify):
    clean = subj["support_labels"]
    # Stable, process-independent seed (Python's hash() is salted per process).
    # `seed` is part of the digest so the per-seed loop still varies the draw.
    _key = f"{sid}|{model}|{delta}|{pc}|{seed}".encode()
    rng = np.random.RandomState(
        int.from_bytes(hashlib.sha256(_key).digest()[:8], "big") % (2 ** 32)
    )
    c = corr.apply_corruption(
        clean, model=model, rng=rng, pc=pc,
        delta_deg=delta, pool_labels=subj["pool_labels"],
    )
    obs = c["corrupted_labels"]

    src = lr.source_residuals(backbone, subj["images"], obs)
    loo = lr.loo_residuals(backbone, subj["images"], obs, steps=steps, lr=lr_)
    pur = (lr.loo_residuals_purified(backbone, subj["images"], obs,
                                     steps=steps, lr=lr_)[0]
           if purify else np.full(K, np.nan))

    records = []
    for i in range(len(clean)):
        records.append({
            "subject": sid, "seed": int(seed), "model": model,
            "delta": (None if model == "structured_swap" else int(delta)),
            "pc": float(pc), "frame_idx": int(i),
            "is_corrupted": bool(c["is_corrupted"][i]),
            "actual_offset_deg": float(c["actual_offset_deg"][i]),
            "loo_residual_deg": (None if np.isnan(loo[i]) else float(loo[i])),
            "source_residual_deg": (None if np.isnan(src[i]) else float(src[i])),
            "loo_residual_purified_deg": (None if np.isnan(pur[i]) else float(pur[i])),
        })
    return records


def run_sweep(subjects, backbone_factory, seeds, deltas, pcs,
              steps, lr_, purify, out_path):
    records = []
    with open(out_path, "w") as f:
        for seed in seeds:
            for sid, subj in subjects.items():
                backbone = backbone_factory(seed, sid)
                cells = [("random_offset", d, pc) for d in deltas for pc in pcs]
                cells += [("structured_swap", None, pc) for pc in pcs]
                for model, delta, pc in cells:
                    recs = score_cell(sid, subj, backbone, seed, model, delta, pc,
                                      steps, lr_, purify)
                    for r in recs:
                        f.write(json.dumps(r) + "\n")
                    records.extend(recs)
    return records


# ---------------------------------------------------------------------------
# AUC summary (the deliverable, §4.5). Full CI/plots -> analyze_phase1.py.
# ---------------------------------------------------------------------------
def summarize_auc(records):
    groups = defaultdict(list)
    for r in records:
        key = (r["model"], r["delta"], r["pc"])
        groups[key].append(r)

    summary = []
    for (model, delta, pc), recs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][1] or 0)):
        y = np.array([r["is_corrupted"] for r in recs], dtype=bool)
        loo = np.array([np.nan if r["loo_residual_deg"] is None else r["loo_residual_deg"] for r in recs])
        src = np.array([np.nan if r["source_residual_deg"] is None else r["source_residual_deg"] for r in recs])
        offs = np.array([r["actual_offset_deg"] for r in recs if r["is_corrupted"]])
        summary.append({
            "model": model, "delta": delta, "pc": pc,
            "n_frames": len(recs), "n_corrupted": int(y.sum()),
            "auc_loo": lr.roc_auc(loo, y),
            "auc_source": lr.roc_auc(src, y),
            "median_offset_deg": (float(np.median(offs)) if offs.size else None),
        })
    return summary


def print_summary(summary):
    print(f"\n{'model':16s} {'delta':>5s} {'pc':>4s} {'n':>5s} {'n+':>4s} "
          f"{'AUC_loo':>8s} {'AUC_src':>8s} {'med_off':>8s}")
    print("-" * 68)
    for s in summary:
        d = "  -" if s["delta"] is None else f"{s['delta']:>3d}"
        mo = "   -" if s["median_offset_deg"] is None else f"{s['median_offset_deg']:6.2f}"
        al = "  nan" if np.isnan(s["auc_loo"]) else f"{s['auc_loo']:6.3f}"
        asrc = "  nan" if np.isnan(s["auc_source"]) else f"{s['auc_source']:6.3f}"
        print(f"{s['model']:16s} {d:>5s} {s['pc']:>4.1f} {s['n_frames']:>5d} "
              f"{s['n_corrupted']:>4d} {al:>8s} {asrc:>8s} {mo:>8s}")


def main():
    ap = argparse.ArgumentParser(description="Phase-1 label-noise detection sweep")
    ap.add_argument("--real", action="store_true", help="use real backbone + ColumbiaGaze (slow)")
    ap.add_argument("--out", default=os.path.join(_HERE, "phase1_records.jsonl"))
    ap.add_argument("--summary", default=os.path.join(_HERE, "phase1_auc_summary.json"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-subjects", type=int, default=15)
    ap.add_argument("--deltas", type=int, nargs="+", default=DELTAS)
    ap.add_argument("--pcs", type=float, nargs="+", default=PCS)
    ap.add_argument("--steps", type=int, default=None, help="adapt steps (default: 150 real / 1 synthetic)")
    ap.add_argument("--lr", type=float, default=api.DEFAULT_LR)
    ap.add_argument("--no-purify", action="store_true")
    ap.add_argument("--cgd-root", default=os.path.join(os.path.dirname(_HERE), "ColumbiaGaze"))
    ap.add_argument("--subjects", nargs="*", default=None, help="real mode: subset of subject ids")
    args = ap.parse_args()

    purify = not args.no_purify

    if args.real:
        steps = args.steps if args.steps is not None else api.DEFAULT_STEPS
        subjects = load_real_subjects(args.cgd_root, args.subjects)
        def backbone_factory(seed, sid):
            return api.load_backbone(subject_id=sid, seed=seed)
        print(f"[real] {len(subjects)} subjects, steps={steps}")
    else:
        steps = args.steps if args.steps is not None else 1   # synthetic adapt is closed-form
        subjects = make_synthetic_subjects(n_subjects=args.n_subjects, K=K, seed=0)
        _synth = api.make_synthetic_backbone()
        def backbone_factory(seed, sid):
            return _synth
        print(f"[synthetic] {len(subjects)} subjects")

    records = run_sweep(subjects, backbone_factory, args.seeds, args.deltas, args.pcs,
                        steps, args.lr, purify, args.out)
    print(f"wrote {len(records)} frame records -> {args.out}")

    summary = summarize_auc(records)
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    print_summary(summary)
    print(f"\nwrote AUC summary -> {args.summary}")
    print("Next: analyze_phase1.py for bootstrap CIs, the offset histogram, and the kill-criterion check (§4.5).")


if __name__ == "__main__":
    main()
