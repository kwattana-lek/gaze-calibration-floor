#!/usr/bin/env python
"""Regenerate report sections 2-4 for a given floor-sweep jsonl.
Usage: analyze_floor.py <records.jsonl> [out.md]
Sections: (2) AUC table by delta/pc with 95% bootstrap CIs + corrupted counts,
(3) delta where AUC_loo crosses 0.7 and 0.9 per pc, (4) structured-swap offset
percentiles + fraction below the crossing."""
import json, sys
import numpy as np

RNG = np.random.RandomState(0)
B = 2000


def roc_auc(scores, pos):
    scores = np.asarray(scores, float); pos = np.asarray(pos, bool)
    ok = ~np.isnan(scores); scores, pos = scores[ok], pos[ok]
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return float('nan')
    u, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); avg = (csum - cnt + csum + 1) / 2.0
    return float((avg[inv][pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def boot_ci(scores, pos):
    scores = np.asarray(scores, float); pos = np.asarray(pos, bool); n = len(scores)
    a = []
    for _ in range(B):
        idx = RNG.randint(0, n, n)
        v = roc_auc(scores[idx], pos[idx])
        if not np.isnan(v):
            a.append(v)
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if a else (float('nan'), float('nan'))


def cell(recs, model, delta, pc):
    y, loo, src = [], [], []
    for r in recs:
        if r['model'] == model and r['delta'] == delta and abs(r['pc'] - pc) < 1e-9:
            y.append(r['is_corrupted'])
            loo.append(np.nan if r['loo_residual_deg'] is None else r['loo_residual_deg'])
            src.append(np.nan if r['source_residual_deg'] is None else r['source_residual_deg'])
    return np.array(y, bool), np.array(loo, float), np.array(src, float)


def crossing(deltas, aucs, level):
    d = np.array(deltas, float); a = np.array(aucs, float)
    o = np.argsort(d); d, a = d[o], a[o]
    if np.all(a < level):
        return f">{d.max():.0f}"
    if a[0] >= level:
        return f"<={d.min():.0f}"
    for i in range(1, len(d)):
        if a[i - 1] < level <= a[i]:
            t = (level - a[i - 1]) / (a[i] - a[i - 1])
            return round(float(d[i - 1] + t * (d[i] - d[i - 1])), 2)
    return "n/a"


recs = [json.loads(l) for l in open(sys.argv[1])]
out = sys.argv[2] if len(sys.argv) > 2 else None
L = []; w = lambda *x: L.append(" ".join(str(t) for t in x))

deltas = sorted({r['delta'] for r in recs if r['model'] == 'random_offset' and r['delta'] is not None})
pcs = sorted({r['pc'] for r in recs})
w(f"# Floor sweep (fixed seeding) — sections 2-4")
w(f"\nSource: `{sys.argv[1]}` — {len(recs)} records. deltas={deltas} pcs={pcs}. Bootstrap B={B}.\n")

w("## 2. AUC by delta and pc (LOO & source), 95% bootstrap CI, corrupted counts\n")
w("| model | δ | pc | n | corrupted | AUC_loo [95% CI] | AUC_src [95% CI] |")
w("|---|---|---|---|---|---|---|")
loo_by_pc = {pc: {} for pc in pcs}
for pc in pcs:
    for d in deltas:
        y, loo, src = cell(recs, 'random_offset', d, pc)
        al, cl = roc_auc(loo, y), boot_ci(loo, y); asr, cs = roc_auc(src, y), boot_ci(src, y)
        loo_by_pc[pc][d] = al
        w(f"| random_offset | {d} | {pc} | {len(y)} | {int(y.sum())} | "
          f"{al:.3f} [{cl[0]:.3f}, {cl[1]:.3f}] | {asr:.3f} [{cs[0]:.3f}, {cs[1]:.3f}] |")
    y, loo, src = cell(recs, 'structured_swap', None, pc)
    al, cl = roc_auc(loo, y), boot_ci(loo, y); asr, cs = roc_auc(src, y), boot_ci(src, y)
    w(f"| structured_swap | – | {pc} | {len(y)} | {int(y.sum())} | "
      f"{al:.3f} [{cl[0]:.3f}, {cl[1]:.3f}] | {asr:.3f} [{cs[0]:.3f}, {cs[1]:.3f}] |")

w("\n## 3. δ where AUC_loo crosses 0.7 and 0.9, per pc\n")
w("| pc | δ@AUC=0.7 | δ@AUC=0.9 |"); w("|---|---|---|")
cross07 = {}
for pc in pcs:
    dl = sorted(loo_by_pc[pc]); la = [loo_by_pc[pc][d] for d in dl]
    c07 = crossing(dl, la, 0.7); cross07[pc] = c07
    w(f"| {pc} | {c07} | {crossing(dl, la, 0.9)} |")
w(f"\n_Floor covers δ∈{deltas}; `>N` lies beyond the swept range._\n")

w("## 4. Structured-swap realised offsets — percentiles & fraction below crossing\n")
allo = np.array([r['actual_offset_deg'] for r in recs if r['model'] == 'structured_swap' and r['is_corrupted']], float)
pcts = [5, 10, 25, 50, 75, 90, 95]
w("| percentile | " + " | ".join(f"p{p}" for p in pcts) + " |")
w("|---|" + "---|" * len(pcts))
w("| deg | " + " | ".join(f"{np.percentile(allo, p):.2f}" for p in pcts) + " |")
w("\n| pc | crossing δ@0.7 | structured offsets < crossing |"); w("|---|---|---|")
for pc in pcs:
    op = np.array([r['actual_offset_deg'] for r in recs if r['model'] == 'structured_swap'
                   and r['is_corrupted'] and abs(r['pc'] - pc) < 1e-9], float)
    c = cross07[pc]
    if isinstance(c, (int, float)):
        w(f"| {pc} | {c} | {(op < c).mean()*100:.1f}%  ({int((op < c).sum())}/{op.size}) |")
    else:
        w(f"| {pc} | {c} | n/a (outside swept range) |")

text = "\n".join(L) + "\n"
if out:
    open(out, 'w').write(text); print("wrote", out)
else:
    print(text)
