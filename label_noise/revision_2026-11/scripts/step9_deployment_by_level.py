"""Review item 9: fixed (A, tau = 3.42) vs scaled (B, tau = 0.96 e_hat) per backbone level and contamination rate,
and recall split by displacement band (near the floor vs far above it).

Inputs (read-only, produced by the frozen Step 3 pipeline):
  03_gate_transfer_results.csv     per level x policy x p_c: recall, FPR, precision, recaptures (flags), cost, MAE
  results/03_auc.csv               resubstitution ROC-AUC per level x p_c
  results/gate_transfer_scores.jsonl  per-frame resub scores, corruption flags and realised offsets
  results/step3_thresholds.json    frozen thresholds
Outputs:
  results/09_deployment_by_level_pc.csv   one row per level x p_c x policy (A, B)
  results/09_recall_by_band.csv           recall of A and B by displacement band, in units of the level's e_hat,
                                          pooled over p_c; 95% subject bootstrap (B = 2000, seed 0)
"""
import csv, json, os
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REV = os.path.dirname(HERE); R = os.path.join(REV, "results")
res = list(csv.DictReader(open(os.path.join(REV, "03_gate_transfer_results.csv"))))
auc = {(r["level"], r["pc"]): r for r in csv.DictReader(open(os.path.join(R, "03_auc.csv"))) if r["scorer"] == "resub"}
thr = json.load(open(os.path.join(R, "step3_thresholds.json")))
key = {(r["level"], r["policy"], r["pc"]): r for r in res}

rows = []
for lv in ["L0", "L1", "L2", "L3"]:
    for pc in ["0.1", "0.2", "0.4"]:
        naive = float(key[(lv, "naive", pc)]["mae"]); orc = float(key[(lv, "oracle_true", pc)]["mae_minus_naive"])
        for pol in ["A", "B"]:
            r = key[(lv, pol, pc)]
            rows.append(dict(level=lv, pc=pc, policy=pol, tau=f"{float(r['tau']):.2f}", auc_resub=auc[(lv, pc)]["auc"],
                             recall=r["recall"], fpr=r["fpr"], precision=r["precision"], recaptures=r["flags"],
                             cost_w4=r["cost"], cost_lo=r["cost_lo"], cost_hi=r["cost_hi"],
                             dmae=r["mae_minus_naive"], dmae_lo=r["mmn_lo"], dmae_hi=r["mmn_hi"],
                             dmae_true_C=f"{orc:.4f}", mae_naive=f"{naive:.4f}"))
with open(os.path.join(R, "09_deployment_by_level_pc.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# recall by displacement band
ehat = {lv: float(next(r for r in csv.DictReader(open(os.path.join(R, "03_gate_transfer_detection.csv"))) if r["level"] == lv)["e_clean"]) for lv in thr}
BANDS = [(0, 1.5, "<1.5e"), (1.5, 3.0, "1.5-3e"), (3.0, 1e9, ">=3e")]
per = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # (lv, pol, band) -> subject -> [hits, n]
cnt = defaultdict(int)
for line in open(os.path.join(R, "gate_transfer_scores.jsonl")):
    r = json.loads(line)
    if r.get("kind") != "cell":
        continue
    lv = r["level"]; e = ehat[lv]
    for corr, off, sc in zip(r["is_corrupted"], r["offset"], r["resub"]):
        if not corr:
            continue
        b = next(lab for lo, hi, lab in BANDS if lo <= off / e < hi)
        cnt[(lv, b)] += 1
        for pol in ("A", "B"):
            h = per[(lv, pol, b)][r["subject"]]; h[0] += sc >= thr[lv][pol]; h[1] += 1
out = []
g = np.random.RandomState(0)
for lv in ["L0", "L1", "L2", "L3"]:
    tot = sum(cnt[(lv, b)] for _, _, b in BANDS)
    for _, _, b in BANDS:
        rec = dict(level=lv, band=b, n_frames=cnt[(lv, b)], share=f"{cnt[(lv, b)] / tot:.3f}")
        for pol in ("A", "B"):
            d = per[(lv, pol, b)]; subs = sorted(d)
            hits = np.array([d[s][0] for s in subs], float); n = np.array([d[s][1] for s in subs], float)
            boot = []
            for _ in range(2000):
                i = g.randint(0, len(subs), len(subs)); boot.append(hits[i].sum() / n[i].sum())
            rec[f"recall_{pol}"] = f"{hits.sum() / n.sum():.3f}"
            rec[f"recall_{pol}_ci"] = f"[{np.percentile(boot, 2.5):.3f}, {np.percentile(boot, 97.5):.3f}]"
        out.append(rec)
with open(os.path.join(R, "09_recall_by_band.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
for r in rows: print(r)
for r in out: print(r)

# paired per-subject cost difference B - A (c_fn = 4) per level x p_c and pooled; 95% subject bootstrap (B = 2000, seed 1)
cost = defaultdict(lambda: defaultdict(list))  # (lv, pc) -> subject -> [cost_B - cost_A per session]
for line in open(os.path.join(R, "gate_transfer_scores.jsonl")):
    r = json.loads(line)
    if r.get("kind") != "cell":
        continue
    lv = r["level"]; c = {}
    for pol in ("A", "B"):
        fl = [s >= thr[lv][pol] for s in r["resub"]]
        fp = sum(f and not k for f, k in zip(fl, r["is_corrupted"])); fn = sum((not f) and k for f, k in zip(fl, r["is_corrupted"]))
        c[pol] = fp + 4 * fn
    for pk in (str(r["pc"]), "pooled"):
        cost[(lv, pk)][r["subject"]].append(c["B"] - c["A"])
g = np.random.RandomState(1); out2 = []
for lv in ["L0", "L1", "L2", "L3"]:
    for pk in ["0.1", "0.2", "0.4", "pooled"]:
        d = cost[(lv, pk)]; subs = sorted(d); m = np.array([np.mean(d[s]) for s in subs])
        boot = [m[g.randint(0, len(m), len(m))].mean() for _ in range(2000)]
        out2.append(dict(level=lv, pc=pk, B_minus_A=f"{m.mean():.3f}", lo=f"{np.percentile(boot, 2.5):.3f}", hi=f"{np.percentile(boot, 97.5):.3f}"))
with open(os.path.join(R, "09_B_minus_A_cost_by_pc.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out2[0])); w.writeheader(); w.writerows(out2)
for r in out2: print(r)
