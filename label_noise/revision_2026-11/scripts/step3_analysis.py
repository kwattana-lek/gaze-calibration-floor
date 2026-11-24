"""Step 3 detection analysis. Inputs: results/gate_transfer_scores.jsonl, results/step3_thresholds.json.
Outputs: results/03_gate_transfer_detection.csv (per level x policy x pc-or-pooled, subject-bootstrap CIs),
results/03_decision.json, results/03_auc.csv. D uses LOPO Youden thresholds (leave one subject out)."""
import json, os, csv, numpy as np
from sklearn.metrics import roc_auc_score
REV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = [json.loads(l) for l in open(os.path.join(REV, "results", "gate_transfer_scores.jsonl"))]
THR = json.load(open(os.path.join(REV, "results", "step3_thresholds.json")))
EH = json.load(open(os.path.join(REV, "results", "step3_ehat.json")))
LEVELS, PCS, B, TOL = ["L0", "L1", "L2", "L3"], [0.1, 0.2, 0.4], 2000, 0.02
rng = np.random.default_rng(20261123)
POL = ["A", "B", "Bprime", "Bdep", "C", "D", "LOO_C", "LOO_D"]

def cands(x):
    u = np.unique(x); return np.concatenate([[u[0] - 1e-6], (u[1:] + u[:-1]) / 2, [u[-1] + 1e-6]])
def youden(sc, y):
    best = None
    for t in cands(sc):
        f = sc >= t; j = f[y].mean() - f[~y].mean()
        if best is None or j > best[1]: best = (t, j)
    return best[0]

rows_out, dec, auc_rows = [], {}, []
for L in LEVELS:
    cells = [r for r in R if r["level"] == L and r["kind"] == "cell"]
    subj = sorted({r["subject"] for r in cells})
    # LOPO Youden for D and LOO_D
    lopo = {}
    for key, pre in (("resub", ""), ("loo", "LOO_")):
        for s in subj:
            o = [r for r in cells if r["subject"] != s]
            lopo[(pre, s)] = youden(np.concatenate([r[key] for r in o]), np.concatenate([r["is_corrupted"] for r in o]).astype(bool))
    # per subject x pc x policy counts
    M = {}  # (pol, pc, s) -> dict of counts
    for r in cells:
        y = np.array(r["is_corrupted"], bool)
        for pol in POL:
            key = "loo" if pol.startswith("LOO_") else "resub"; sc = np.array(r[key])
            t = THR[L][pol]
            if pol in ("D", "LOO_D"): t = lopo[("LOO_" if pol == "LOO_D" else "", r["subject"])]
            elif isinstance(t, dict): t = t[str(r["pc"])]
            f = sc >= t
            d = M.setdefault((pol, r["pc"], r["subject"]), dict(tp=0, fp=0, fn=0, tn=0, n=0))
            d["tp"] += int((f & y).sum()); d["fp"] += int((f & ~y).sum()); d["fn"] += int((~f & y).sum()); d["tn"] += int((~f & ~y).sum()); d["n"] += 1
    def metr(ds, w=4.0):
        tp, fp, fn, tn, n = (sum(d[k] for d in ds) for k in ("tp", "fp", "fn", "tn", "n"))
        rec = tp / (tp + fn) if tp + fn else np.nan; fpr = fp / (fp + tn)
        prec = tp / (tp + fp) if tp + fp else np.nan
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
        return dict(recall=rec, fpr=fpr, precision=prec, f1=f1, flags=(tp + fp) / n, cost=(fp + w * fn) / n,
                    cost_w1=(fp + fn) / n, cost_w2=(fp + 2 * fn) / n, cost_w8=(fp + 8 * fn) / n)
    def subj_metric(pol, pcs, s):  # one subject, pooled over pcs
        return metr([M[(pol, pc, s)] for pc in pcs])
    for pcs, lab in [([pc], str(pc)) for pc in PCS] + [(PCS, "pooled")]:
        per = {pol: [subj_metric(pol, pcs, s) for s in subj] for pol in POL}
        for pol in POL:
            rowd = {"level": L, "e_clean": round(EH[L]["e_clean"], 3), "policy": pol, "pc": lab}
            t = THR[L][pol]
            rowd["tau"] = (round(float(np.mean([lopo[("LOO_" if pol == "LOO_D" else "", s)] for s in subj])), 3) if pol in ("D", "LOO_D")
                           else round(t[lab], 3) if isinstance(t, dict) and lab != "pooled" else "per-pc" if isinstance(t, dict) else round(t, 3))
            for m in ["recall", "fpr", "precision", "f1", "flags", "cost", "cost_w1", "cost_w2", "cost_w8"]:
                v = np.array([p[m] for p in per[pol]], float)
                bs = [np.nanmean(v[rng.integers(0, len(v), len(v))]) for _ in range(B)]
                rowd[m] = round(float(np.nanmean(v)), 4); rowd[m + "_lo"] = round(float(np.nanpercentile(bs, 2.5)), 4); rowd[m + "_hi"] = round(float(np.nanpercentile(bs, 97.5)), 4)
            rows_out.append(rowd)
        # paired differences vs D
        for pol in ["A", "B", "Bprime", "Bdep"]:
            for m in ["recall", "fpr", "cost"]:
                v = np.array([a[m] - b[m] for a, b in zip(per[pol], per["D"])], float)
                bs = [np.nanmean(v[rng.integers(0, len(v), len(v))]) for _ in range(B)]
                dec.setdefault(L, {}).setdefault(lab, {})[f"{pol}-D_{m}"] = [round(float(np.nanmean(v)), 4), round(float(np.nanpercentile(bs, 2.5)), 4), round(float(np.nanpercentile(bs, 97.5)), 4)]
    # AUC per subject (threshold-free)
    for pc in PCS + ["pooled"]:
        for key in ("resub", "loo"):
            a = []
            for s in subj:
                cc = [r for r in cells if r["subject"] == s and (pc == "pooled" or r["pc"] == pc)]
                y = np.concatenate([r["is_corrupted"] for r in cc]); sc = np.concatenate([r[key] for r in cc])
                a.append(roc_auc_score(y, sc))
            a = np.array(a); bs = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(B)]
            auc_rows.append({"level": L, "pc": pc, "scorer": key, "auc": round(a.mean(), 4), "lo": round(np.percentile(bs, 2.5), 4), "hi": round(np.percentile(bs, 97.5), 4)})
# decision rule (pooled over p_c, primary)
verdict = {}
for pol in ["A", "B", "Bprime", "Bdep"]:
    ok = {L: abs(dec[L]["pooled"][f"{pol}-D_recall"][0]) <= TOL and abs(dec[L]["pooled"][f"{pol}-D_fpr"][0]) <= TOL for L in LEVELS}
    verdict[pol] = ok
dec["verdict_within_tol_of_D_pooled"] = verdict
for f, rows in (("03_gate_transfer_detection.csv", rows_out), ("03_auc.csv", auc_rows)):
    with open(os.path.join(REV, "results", f), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
json.dump(dec, open(os.path.join(REV, "results", "03_decision.json"), "w"), indent=1)
print(json.dumps(verdict))
for r in auc_rows:
    if r["pc"] == "pooled": print("AUC", r)
for r in rows_out:
    if r["pc"] == "pooled": print(r["level"], r["policy"], r["tau"], "rec %.3f fpr %.3f flags %.2f cost %.2f [%.2f,%.2f]" % (r["recall"], r["fpr"], r["flags"], r["cost"], r["cost_lo"], r["cost_hi"]))
for L in LEVELS:
    print(L, {k: v for k, v in dec[L]["pooled"].items()})
