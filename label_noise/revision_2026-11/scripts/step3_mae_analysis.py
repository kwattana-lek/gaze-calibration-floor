"""Downstream held-out MAE after each policy (results/gate_transfer_mae.jsonl). Per-subject means over seeds,
bootstrap over subjects (B=2000). Writes results/03_gate_transfer_results.csv (the Step 3 deliverable: detection +
MAE merged per level x policy x pc) and results/03_mae_paired.json."""
import json, csv, os, numpy as np
REV = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P = lambda f: os.path.join(REV, "results", f)
M = [json.loads(l) for l in open(P("gate_transfer_mae.jsonl"))]
DET = {(r["level"], r["policy"], r["pc"]): r for r in csv.DictReader(open(P("03_gate_transfer_detection.csv")))}
LEVELS, PCS = ["L0", "L1", "L2", "L3"], [0.1, 0.2, 0.4]; rng = np.random.default_rng(7)
POL = ["naive", "oracle_true", "A", "B", "Bprime", "Bdep", "C", "D", "LOO_C", "LOO_D"]
def boot(v):
    v = np.asarray(v, float); bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
    return round(float(v.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)
def per_subj(L, pol, pcs):
    d = {}
    for r in M:
        if r["level"] == L and r["pc"] in pcs: d.setdefault(r["subject"], []).append(r[pol])
    return {s: np.mean(v) for s, v in d.items()}
rows, paired = [], {}
for L in LEVELS:
    for pcs, lab in [([pc], str(pc)) for pc in PCS] + [(PCS, "pooled")]:
        ps = {p: per_subj(L, p, pcs) for p in POL}; S = sorted(ps["naive"])
        for p in POL:
            m = boot([ps[p][s] for s in S]); dn = boot([ps[p][s] - ps["naive"][s] for s in S])
            row = {"level": L, "policy": p, "pc": lab, "n_subjects": len(S), "mae": m[0], "mae_lo": m[1], "mae_hi": m[2],
                   "mae_minus_naive": dn[0], "mmn_lo": dn[1], "mmn_hi": dn[2]}
            det = DET.get((L, p, lab))
            if det: row.update({k: det[k] for k in ["tau", "recall", "fpr", "precision", "f1", "flags", "cost", "cost_lo", "cost_hi", "cost_w1", "cost_w2", "cost_w8"]})
            rows.append(row)
        for a, b in [("B", "A"), ("A", "D"), ("B", "D"), ("LOO_C", "C"), ("LOO_D", "D"), ("A", "oracle_true")]:
            paired.setdefault(L, {}).setdefault(lab, {})[f"{a}-{b}"] = boot([ps[a][s] - ps[b][s] for s in S])
keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[-1].keys()).index(k) if k in rows[-1] else 99)
with open(os.path.join(REV, "03_gate_transfer_results.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
json.dump(paired, open(P("03_mae_paired.json"), "w"), indent=1)
for r in rows:
    if r["pc"] == "pooled": print(r["level"], r["policy"], "MAE %.3f [%.3f,%.3f] Δnaive %+.3f [%+.3f,%+.3f] n=%d" % (r["mae"], r["mae_lo"], r["mae_hi"], r["mae_minus_naive"], r["mmn_lo"], r["mmn_hi"], r["n_subjects"]))
for L in LEVELS: print(L, paired[L]["pooled"])
