#!/usr/bin/env python
"""Verify Phase-2 prerequisites (a)(b)(c) -> logs/PHASE2_PREREQ_REPORT.md.
(a) floor 0.9 crossing from combined delta 3-10; (b) 15-fold backbone mean vs
4.51/3.47 with per-fold drift; (c) detector mean vs 0.9512 + weight count."""
import os, json, glob
import numpy as np

AG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LN, LOGS = os.path.join(AG, "label_noise"), os.path.join(AG, "logs")
L = []; w = lambda *x: L.append(" ".join(str(t) for t in x))


def roc_auc(scores, pos):
    scores = np.asarray(scores, float); pos = np.asarray(pos, bool)
    ok = ~np.isnan(scores); scores, pos = scores[ok], pos[ok]
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0: return float("nan")
    u, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); avg = (csum - cnt + csum + 1) / 2.0
    return float((avg[inv][pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def crossing(ds, aucs, level):
    d = np.array(ds, float); a = np.array(aucs, float); o = np.argsort(d); d, a = d[o], a[o]
    if np.all(a < level): return f">{d.max():.0f}"
    if a[0] >= level: return f"<={d.min():.0f}"
    for i in range(1, len(d)):
        if a[i - 1] < level <= a[i]:
            t = (level - a[i - 1]) / (a[i] - a[i - 1]); return round(float(d[i - 1] + t * (d[i] - d[i - 1])), 2)
    return "n/a"


def load(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


w("# Phase-2 prerequisites — verification report\n")

# (a) floor 0.9 crossing over combined delta 3-10
w("## (a) Floor extension δ∈{8,10} — locate the ROC-AUC 0.9 crossing\n")
recs = load(os.path.join(LN, "phase1_records_floor_v2.jsonl")) + load(os.path.join(LN, "phase1_records_floor_d8_10.jsonl"))
if recs:
    pcs = sorted({r["pc"] for r in recs})
    deltas = sorted({r["delta"] for r in recs if r["model"] == "random_offset" and r["delta"] is not None})
    w(f"Combined deltas = {deltas} (v2 δ3–6 + new δ8,10). Cells: 450 frames each.\n")
    w("| pc | δ@0.7 | δ@0.9 | AUC_loo per δ |"); w("|---|---|---|---|")
    for pc in pcs:
        aucs = []
        for d in deltas:
            y = [r["is_corrupted"] for r in recs if r["model"] == "random_offset" and r["delta"] == d and abs(r["pc"] - pc) < 1e-9]
            s = [r["loo_residual_deg"] for r in recs if r["model"] == "random_offset" and r["delta"] == d and abs(r["pc"] - pc) < 1e-9]
            aucs.append(roc_auc([np.nan if v is None else v for v in s], y))
        per = " ".join(f"{d}:{a:.3f}" for d, a in zip(deltas, aucs))
        w(f"| {pc} | {crossing(deltas, aucs, 0.7)} | {crossing(deltas, aucs, 0.9)} | {per} |")
else:
    w("_Floor δ8,10 records missing._")
w("")

# (b) 15-fold backbone regeneration
w("## (b) Backbone regeneration — 15 folds persisted, mean reproduces 4.51 / 3.47\n")
new_p = os.path.join(AG, "experiments", "lopo_backbone", "results_regen_20260809.json")
rec_p = os.path.join(AG, "experiments", "lopo_backbone", "results.json")
if os.path.exists(new_p):
    new = {r["subject"]: r for r in json.load(open(new_p))}
    rec = {r["subject"]: r for r in json.load(open(rec_p))} if os.path.exists(rec_p) else {}
    cross = np.array([new[s]["cross_subject_mae"] for s in new])
    full = np.array([new[s]["calib_full_mae"] for s in new])
    w(f"Folds regenerated: {len(new)}.  "
      f"15-fold mean cross_subject_mae = **{cross.mean():.3f}** (recorded 4.511), "
      f"calib_full_mae = **{full.mean():.3f}** (recorded 3.472).")
    w(f"-> {'MEAN REPRODUCES' if abs(cross.mean()-4.511)<0.1 and abs(full.mean()-3.472)<0.1 else 'MEAN DIFFERS'} "
      f"(tol 0.1° on the 15-fold mean, per handoff).\n")
    if rec:
        dc = np.array([abs(rec[s]["cross_subject_mae"] - new[s]["cross_subject_mae"]) for s in new if s in rec])
        w(f"Per-fold cross drift: median {np.median(dc):.3f}°, max {dc.max():.3f}° "
          f"(handoff expects ≤~0.8° from MPS training nondeterminism).\n")
    # persistence check
    saved = sorted(glob.glob(os.path.join(AG, "models", "backbone_mpiifacegaze_p*.pt")))
    w(f"Persisted backbones in models/: {len(saved)}/15 "
      f"({', '.join(os.path.basename(p).split('_')[-1][:-3] for p in saved)}).")
else:
    w("_Step (b) output missing — see logs/prereq_b_backbones.log._")
w("")

# (c) detector persistence + reproduction
w("## (c) Detector weights persisted + 0.9512 reproduced\n")
det_p = os.path.join(AG, "experiments", "detector_lopo", "results_regen_20260809.json")
if os.path.exists(det_p):
    d = json.load(open(det_p)); rocs = np.array([r["det_roc"] for r in d], float)
    m = rocs.mean(); ci = 1.96 * rocs.std(ddof=1) / np.sqrt(len(rocs))
    nweights = len(glob.glob(os.path.join(AG, "models", "detectors", "detector_lopo_*.pt")))
    w(f"Folds: {len(rocs)}. Mean det ROC-AUC = **{m:.4f}** ± {ci:.4f} (95%), expected 0.9512 "
      f"→ {'reproduces' if abs(m-0.9512)<=0.01 else 'DIFFERS'}.")
    w(f"Persisted detector weights: {nweights} files in models/detectors/.")
else:
    w("_Step (c) output missing — see logs/prereq_c_detector.log._")
w("")

# wall times
ts = os.path.join(LOGS, "phase2_prereq_timings.tsv")
if os.path.exists(ts):
    w("## Wall time\n"); w("| step | duration |"); w("|---|---|")
    for line in open(ts):
        p = line.strip().split("\t")
        if len(p) == 3:
            s = int(float(p[2]) - float(p[1])); w(f"| {p[0]} | {s//3600}h{(s%3600)//60:02d}m{s%60:02d}s |")

open(os.path.join(LOGS, "PHASE2_PREREQ_REPORT.md"), "w").write("\n".join(L) + "\n")
print("wrote", os.path.join(LOGS, "PHASE2_PREREQ_REPORT.md"))
