#!/usr/bin/env python
"""Proof that the seeding fix makes corruption process-independent.
Compares two run_phase1 outputs (produced in separate processes with different
PYTHONHASHSEED): corrupted frame indices and cell AUC must be identical."""
import json, sys, hashlib
from collections import defaultdict
import numpy as np


def load(p):
    return [json.loads(l) for l in open(p)]


def roc_auc(scores, pos):
    scores = np.array([np.nan if s is None else s for s in scores], float)
    pos = np.asarray(pos, bool)
    ok = ~np.isnan(scores); scores, pos = scores[ok], pos[ok]
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return float('nan')
    u, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); avg = (csum - cnt + csum + 1) / 2.0
    return float((avg[inv][pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def cells(recs):
    d = defaultdict(lambda: {'idx': [], 'y': [], 'loo': [], 'src': []})
    for r in recs:
        k = (r['subject'], r['model'], r['delta'], r['pc'])
        d[k]['idx'].append((r['frame_idx'], r['is_corrupted']))
        d[k]['y'].append(r['is_corrupted'])
        d[k]['loo'].append(r['loo_residual_deg'])
        d[k]['src'].append(r['source_residual_deg'])
    return d


c1, c2 = cells(load(sys.argv[1])), cells(load(sys.argv[2]))
print(f"run1={sys.argv[1]}  run2={sys.argv[2]}\n")
allok = True
for k in sorted(c1, key=str):
    corr1 = sorted(i for i, c in c1[k]['idx'] if c)
    corr2 = sorted(i for i, c in c2[k]['idx'] if c)
    a1l, a2l = roc_auc(c1[k]['loo'], c1[k]['y']), roc_auc(c2[k]['loo'], c2[k]['y'])
    a1s, a2s = roc_auc(c1[k]['src'], c1[k]['y']), roc_auc(c2[k]['src'], c2[k]['y'])
    same = (corr1 == corr2
            and (np.isnan(a1l) and np.isnan(a2l) or abs(a1l - a2l) < 1e-12)
            and (np.isnan(a1s) and np.isnan(a2s) or abs(a1s - a2s) < 1e-12))
    allok &= same
    print(f"cell {k}")
    print(f"  corrupted frame_idx : run1={corr1}  run2={corr2}")
    print(f"  AUC_loo  : {a1l:.6f} / {a2l:.6f}   AUC_src : {a1s:.6f} / {a2s:.6f}")
    print(f"  -> {'MATCH' if same else 'DIFFER'}\n")

print("per-seed variation (same sid/model/delta/pc, different seed -> different draw):")
for s in (0, 1, 2):
    key = f"p00|random_offset|5|0.2|{s}".encode()
    print(f"  seed {s} -> rng_seed {int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % (2**32)}")

print("\nVERDICT:", "PASS — corruption identical across processes" if allok else "FAIL")
sys.exit(0 if allok else 1)
