"""Step 3: compute frozen policy thresholds from gate_transfer_scores.jsonl, exactly per PREREG_step3_gate_transfer.md.
Writes results/step3_thresholds.json (input to `gate_transfer_run.py mae`) and results/step3_ehat.json.
Conventions fixed here (not specified in the prereg, chosen before any metric was looked at):
 - flag rule: score >= tau (as in gate_transfer_run.py);
 - C (oracle): per p_c, tau minimising realised cost per session = FP + w*FN (c_fp=1, w=4), pooled over subjects and seeds;
 - D (Youden): pooled over subjects, seeds and p_c (as the paper's 3.42 gate). For the MAE pass the in-sample tau is used;
   detection metrics for D use LOPO thresholds (step3_analysis.py);
 - LOO_C / LOO_D: the same two rules applied to the LOO residual (Step 4 comparison)."""
import json, os, numpy as np
REV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = [json.loads(l) for l in open(os.path.join(REV, "results", "gate_transfer_scores.jsonl"))]
LEVELS, PCS, W = ["L0", "L1", "L2", "L3"], [0.1, 0.2, 0.4], 4.0

def cands(x):
    u = np.unique(x); return np.concatenate([[u[0] - 1e-6], (u[1:] + u[:-1]) / 2, [u[-1] + 1e-6]])

def oracle(sc, y, n_sess, w=W):
    best = None
    for t in cands(sc):
        f = sc >= t; c = ((f & ~y).sum() + w * ((~f) & y).sum()) / n_sess
        if best is None or c < best[1]: best = (float(t), float(c))
    return best

def youden(sc, y):
    best = None
    for t in cands(sc):
        f = sc >= t; j = f[y].mean() - f[~y].mean()
        if best is None or j > best[1]: best = (float(t), float(j))
    return best

ehat, thr = {}, {}
for L in LEVELS:
    clean = [r for r in R if r["level"] == L and r["kind"] == "clean"]
    cells = [r for r in R if r["level"] == L and r["kind"] == "cell"]
    e_clean = float(np.mean([np.mean(r["loo"]) for r in clean]))
    e_clean_resub = float(np.mean([np.mean(r["resub"]) for r in clean]))
    e_held = float(np.mean([r["heldout_mae"] for r in clean]))
    e_dep = {}
    for pc in PCS:
        subj = sorted({r["subject"] for r in cells})
        e_dep[str(pc)] = float(np.mean([np.mean([np.mean(r["loo"]) for r in cells if r["subject"] == s and r["pc"] == pc]) for s in subj]))
    ehat[L] = {"e_clean": e_clean, "e_clean_resub_mean": e_clean_resub, "e_heldout": e_held, "e_dep": e_dep}
    t = {"A": 3.42, "B": 0.96 * e_clean, "Bdep": {k: 0.96 * v for k, v in e_dep.items()}}
    thr[L] = t
eL0 = ehat["L0"]["e_clean"]
for L in LEVELS:
    thr[L]["Bprime"] = 3.42 / eL0 * ehat[L]["e_clean"]
    cells = [r for r in R if r["level"] == L and r["kind"] == "cell"]
    for pre, key in (("", "resub"), ("LOO_", "loo")):
        C = {}
        for pc in PCS:
            cc = [r for r in cells if r["pc"] == pc]
            sc = np.concatenate([r[key] for r in cc]); y = np.concatenate([r["is_corrupted"] for r in cc]).astype(bool)
            C[str(pc)] = oracle(sc, y, len(cc))[0]
        sc = np.concatenate([r[key] for r in cells]); y = np.concatenate([r["is_corrupted"] for r in cells]).astype(bool)
        thr[L][pre + "C"] = C; thr[L][pre + "D"] = youden(sc, y)[0]
json.dump(ehat, open(os.path.join(REV, "results", "step3_ehat.json"), "w"), indent=1)
json.dump(thr, open(os.path.join(REV, "results", "step3_thresholds.json"), "w"), indent=1)
for L in LEVELS:
    e = ehat[L]; t = thr[L]
    print(L, "e_clean=%.2f e_held=%.2f e_dep=%s" % (e["e_clean"], e["e_heldout"], {k: round(v, 2) for k, v in e["e_dep"].items()}))
    print("   ", {k: (round(v, 2) if not isinstance(v, dict) else {a: round(b, 2) for a, b in v.items()}) for k, v in t.items()})
