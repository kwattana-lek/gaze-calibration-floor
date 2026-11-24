#!/usr/bin/env python
"""STEP-1 MPS sanity gate (new tooling; does not modify experiment code).

Compares the delta=5 / pc=0.2 / random_offset cell computed on CPU vs MPS with
IDENTICAL corruption (both runs launched with PYTHONHASHSEED=0, so
run_phase1.score_cell's hash-seeded RNG draws the same corrupted frames/offsets
on both). With identical inputs, the only difference is the compute backend, so a
divergence > 0.05 in ROC-AUC indicates the MPS backend is corrupting results.

Usage: overnight_gate.py <cpu.jsonl> <mps.jsonl>
Exit 0 = PASS (continue), 2 = ABORT.
"""
import json, sys
import numpy as np

REF_LOO, REF_SRC = 0.776, 0.697   # user's full-cell MPS reference (5 subj x 3 seed)
THRESH = 0.05


def roc_auc(scores, pos):
    scores = np.asarray(scores, float); pos = np.asarray(pos, bool)
    ok = ~np.isnan(scores); scores, pos = scores[ok], pos[ok]
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return float('nan')
    u, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); avg = (csum - cnt + csum + 1) / 2.0
    ranks = avg[inv]
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def cell_auc(recs, model='random_offset', delta=5, pc=0.2):
    loo, src, y = [], [], []
    for r in recs:
        if r['model'] == model and r['delta'] == delta and abs(r['pc'] - pc) < 1e-9:
            y.append(r['is_corrupted'])
            loo.append(np.nan if r['loo_residual_deg'] is None else r['loo_residual_deg'])
            src.append(np.nan if r['source_residual_deg'] is None else r['source_residual_deg'])
    return roc_auc(loo, y), roc_auc(src, y), int(np.sum(y)), len(y)


def main():
    cpu = [json.loads(l) for l in open(sys.argv[1])]
    mps = [json.loads(l) for l in open(sys.argv[2])]
    cl, cs, cn, cN = cell_auc(cpu)
    ml, ms, mn, mN = cell_auc(mps)
    dloo, dsrc = abs(cl - ml), abs(cs - ms)
    out = []
    out.append("================ STEP 1 — MPS SANITY GATE ================")
    out.append("cell: random_offset delta=5 pc=0.2  (subjects p00,p01, seed 0)")
    out.append(f"  frames: CPU n={cN} corrupted={cn} | MPS n={mN} corrupted={mn}")
    out.append(f"  CPU:  auc_loo={cl:.4f}  auc_source={cs:.4f}")
    out.append(f"  MPS:  auc_loo={ml:.4f}  auc_source={ms:.4f}")
    out.append(f"  |CPU-MPS|: loo={dloo:.4f}  source={dsrc:.4f}   (abort threshold {THRESH})")
    out.append(f"  context — full-cell MPS reference (5subj x 3seed): loo={REF_LOO} source={REF_SRC}")
    out.append(f"            CPU vs reference: loo={abs(cl-REF_LOO):.4f} source={abs(cs-REF_SRC):.4f}")
    out.append("            (reference is hash-salted from a different process; not expected to match exactly")
    out.append("             at this small n — the CPU-vs-MPS identical-corruption test above is the real gate)")
    ok = (dloo <= THRESH) and (dsrc <= THRESH) and not (np.isnan(dloo) or np.isnan(dsrc))
    out.append(f"  VERDICT: {'PASS — MPS agrees with CPU; continuing' if ok else 'ABORT — MPS diverges from CPU by >0.05'}")
    out.append("==========================================================")
    print("\n".join(out))
    sys.exit(0 if ok else 2)


if __name__ == '__main__':
    main()
