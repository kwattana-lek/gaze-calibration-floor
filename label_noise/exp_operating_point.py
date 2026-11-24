# exp_operating_point.py  (operating-point / interface spec for §6.5.4)
"""Turn §6.5's ROC-AUC into a threshold. Pure re-analysis of existing residual
records (resubstitution residual = score_small_loss). No retraining.

Records: structured donor-swap + δ=6 random-offset, seeds 0-9
  (exp_a_records seeds 0-2 + exp_seed_extension_records seeds 3-9).
Threshold: flag frame corrupted iff resid >= tau. Selection: LOPO (choose on the
other 14 by Youden's J, evaluate on the held-out); in-sample reported for the
optimism gap. Outputs: logs/OPERATING_POINT_REPORT.md, tradeoff figure, CSVs.
"""
import csv, json, os
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); _AG = os.path.dirname(_HERE)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
# Dataset switch (same convention as unigaze_arm.py / train_backbone.py): defaults
# reproduce the MPIIFaceGaze results byte-for-byte; a second dataset sets
# GAZE_SUBJECTS and GAZE_TAG (e.g. "_eyediap"), which suffixes every file read or
# written so the MPIIFaceGaze records and reports are never touched.
TAG = os.environ.get("GAZE_TAG", "")
SUBJ = ([x for x in os.environ["GAZE_SUBJECTS"].split(",") if x] if os.environ.get("GAZE_SUBJECTS")
        else [f"p{i:02d}" for i in range(15)])
PCS = [0.1, 0.2, 0.4]; K = 9
RNG = np.random.RandomState(0)


def load():
    recs = [r for r in (json.loads(l) for l in open(os.path.join(_HERE, f"exp_a_records{TAG}.jsonl")))
            if r["seed"] in (0, 1, 2) and ((r["model"] == "structured_swap") or (r["model"] == "random_offset" and r["delta"] == 6))]
    recs += [json.loads(l) for l in open(os.path.join(_HERE, f"exp_seed_extension_records{TAG}.jsonl"))]
    return recs


def confusion(rows, tau):  # rows: list of (is_corrupted, resid)
    tp = fp = tn = fn = 0
    for c, r in rows:
        if r is None: continue
        f = r >= tau
        if c and f: tp += 1
        elif c and not f: fn += 1
        elif (not c) and f: fp += 1
        else: tn += 1
    return tp, fp, tn, fn


def rates(tp, fp, tn, fn):
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    return prec, rec, spec, fpr


def youden_tau(rows):
    taus = sorted(set(r for _, r in rows if r is not None))
    best_t, best_j = taus[0], -1
    for t in taus:
        _, rec, spec, _ = rates(*confusion(rows, t))
        if rec == rec and spec == spec:
            j = rec + spec - 1
            if j > best_j: best_j, best_t = j, t
    return best_t


def rows_of(recs, col="score_small_loss", model=None, pc=None, subj=None):
    out = []
    for r in recs:
        if model and r["model"] != model: continue
        if pc is not None and abs(r["pc"] - pc) > 1e-9: continue
        if subj and r["subject"] != subj: continue
        out.append((r["is_corrupted"], r.get(col)))
    return out


L = []; w = lambda *x: L.append(" ".join(str(t) for t in x))
recs = load()
STRUCT = [r for r in recs if r["model"] == "structured_swap"]
w("# Operating point — resubstitution-residual interface spec (§6.5.4)\n")
w("Resubstitution residual = score_small_loss. Flag frame corrupted iff resid ≥ τ. "
  f"Structured donor-swap, seeds 0-9, {len(SUBJ)} subjects{TAG}.\n")

# ---- 1. LOPO threshold selection (structured, pc-pooled) ----
lopo, insamp, taus = {}, {}, {}
for s in SUBJ:
    train = rows_of(STRUCT, subj=None); train = [(c, r) for (subj, c, r) in
            [(r["subject"], r["is_corrupted"], r.get("score_small_loss")) for r in STRUCT] if subj != s]
    t = youden_tau(train); taus[s] = t
    held = [(r["is_corrupted"], r.get("score_small_loss")) for r in STRUCT if r["subject"] == s]
    lopo[s] = rates(*confusion(held, t)); insamp[s] = rates(*confusion(train, t))
ship = float(np.median(list(taus.values())))
w("## 1. LOPO threshold selection (Youden's J on the other 14; structured, pc-pooled)\n")
w(f"Per-subject τ: median **{ship:.3f}°**, range [{min(taus.values()):.3f}, {max(taus.values()):.3f}], "
  f"IQR [{np.percentile(list(taus.values()),25):.3f}, {np.percentile(list(taus.values()),75):.3f}].")
def agg(d, i): v = np.array([d[s][i] for s in SUBJ], float); return np.nanmean(v), np.nanstd(v)
w("\n| metric | LOPO (held-out, honest) | in-sample (optimistic) |")
w("|---|---|---|")
for i, nm in enumerate(["precision", "recall", "specificity", "FPR (clean flagged)"]):
    lm, ls = agg(lopo, i); im, isd = agg(insamp, i)
    w(f"| {nm} | {lm:.3f} ± {ls:.3f} | {im:.3f} ± {isd:.3f} |")
w(f"\nOptimism gap (in-sample − LOPO) recall/FPR shows selection bias; LOPO is what ships.\n")

# ---- 2. per-pc metrics at the shipped τ, paired per-subject with spread ----
w(f"## 2. Metrics at shipped τ={ship:.3f}° — structured, per p_c (paired per-subject)\n")
w("| p_c | precision | recall | specificity | FPR (clean) |  (mean ± sd across 15 subj) |")
w("|---|---|---|---|---|---|")
per_pc = {}
for pc in PCS:
    M = np.array([rates(*confusion([(r["is_corrupted"], r.get("score_small_loss"))
                 for r in STRUCT if r["subject"] == s and abs(r["pc"] - pc) < 1e-9], ship)) for s in SUBJ], float)
    per_pc[pc] = np.nanmean(M, 0)
    m, sd = np.nanmean(M, 0), np.nanstd(M, 0)
    w(f"| {pc} | {m[0]:.2f}±{sd[0]:.2f} | {m[1]:.2f}±{sd[1]:.2f} | {m[2]:.2f}±{sd[2]:.2f} | {m[3]:.2f}±{sd[3]:.2f} | |")
w("")

# ---- 3. user cost at K=9 ----
w("## 3. User cost at K=9 (one flagged frame = 11.1% of a 9-frame session)\n")
w("| p_c | falsely-flagged clean | correctly-flagged corrupt | total recaptures | wasted frac | total as % session |")
w("|---|---|---|---|---|---|")
for pc in PCS:
    prec, rec, spec, fpr = per_pc[pc]
    fflag = 9 * (1 - pc) * fpr; corr = 9 * pc * rec; tot = fflag + corr
    wasted = fflag / tot if tot else np.nan
    w(f"| {pc} | {fflag:.2f} | {corr:.2f} | {tot:.2f} | {wasted*100:.0f}% | {tot/9*100:.0f}% |")
w("")

# ---- 4. trade-off curve ----
def curve(rows, pc):
    taus = np.linspace(min(r for _, r in rows if r is not None), max(r for _, r in rows if r is not None), 120)
    wasted, missed = [], []
    for t in taus:
        _, rec, spec, fpr = rates(*confusion(rows, t))
        wasted.append(9 * (1 - pc) * (1 - spec)); missed.append(9 * pc * (1 - rec))
    return np.array(wasted), np.array(missed)
plt.figure(figsize=(6, 4.5))
for pc in PCS:
    rows = [(r["is_corrupted"], r.get("score_small_loss")) for r in STRUCT if abs(r["pc"] - pc) < 1e-9]
    wst, mis = curve(rows, pc); plt.plot(wst, mis, label=f"pc={pc}")
    _, rec, spec, _ = rates(*confusion(rows, ship)); plt.scatter([9 * (1 - pc) * (1 - spec)], [9 * pc * (1 - rec)], marker="*", s=120, zorder=5)
plt.xlabel("expected wasted recaptures / session (clean flagged)"); plt.ylabel("missed corrupted frames / session")
plt.title(f"Deployment trade-off (★ = LOPO τ={ship:.2f}°)"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(_AG, "logs", f"operating_point_tradeoff{TAG}.png"), dpi=120); plt.close()
w(f"## 4. Trade-off curve → logs/operating_point_tradeoff.png (★ = shipped τ={ship:.3f}°)\n")

# ---- 5. LOO vs resub gain at the significant cells, in frames/session ----
w("## 5. Concrete detection gain of LOO over resubstitution (the K vs 1 adaptations cost)\n")
w("| cell | scorer | recall | FPR | corrupt caught/session | clean flagged/session |")
w("|---|---|---|---|---|---|")
for model, delta, pc, lbl in [("random_offset", 6, 0.2, "δ=6/pc0.2"), ("random_offset", 6, 0.4, "δ=6/pc0.4"), ("structured_swap", None, 0.1, "struct/pc0.1")]:
    sub = [r for r in recs if r["model"] == model and (delta is None or r["delta"] == delta) and abs(r["pc"] - pc) < 1e-9]
    for col, nm in [("score_small_loss", "resub"), ("score_loo", "LOO")]:
        # LOPO threshold for this scorer on structured pool (ship-comparable): reuse per-scorer Youden on all sub
        rows = [(r["is_corrupted"], r.get(col)) for r in sub]
        t = youden_tau(rows); _, rec, spec, fpr = rates(*confusion(rows, t))
        w(f"| {lbl} | {nm} | {rec:.2f} | {fpr:.2f} | {9*pc*rec:.2f} | {9*(1-pc)*fpr:.2f} |")
w("\n(At matched Youden operating points, LOO's extra corrupted-caught per session is the concrete "
  "return on 9× the adaptation cost.)\n")

# ---- 6. ship paragraph placeholder computed numbers ----
prec2, rec2, spec2, fpr2 = per_pc[0.2]
w("## 6. §6.5.4 ship recommendation\n")
w(f"> Ship the resubstitution-residual gate at τ ≈ **{ship:.2f}°** (LOPO-selected, Youden's J). "
  f"At the mid operating rate (p_c=0.2) it recovers ~{9*0.2*rec2:.1f} corrupted frames/session at a cost of "
  f"~{9*0.8*fpr2:.1f} wrongly-recaptured clean frames — i.e. ≈{(9*0.8*fpr2)/9*100:.0f}% of a nine-frame session wasted, "
  f"with the wasted fraction of all recaptures at {100*(9*0.8*fpr2)/max(1e-9,9*0.8*fpr2+9*0.2*rec2):.0f}%. "
  f"FPR varies across subjects (±{np.nanstd([rates(*confusion([(r['is_corrupted'],r.get('score_small_loss')) for r in STRUCT if r['subject']==s and abs(r['pc']-0.2)<1e-9],ship))[3] for s in SUBJ]):.2f}), "
  f"a direct consequence of the σ spread (§6.3): an absolute threshold flags high-error subjects more.")

# CSVs
with open(os.path.join(_AG, "logs", f"operating_point_per_subject{TAG}.csv"), "w", newline="") as f:
    ww = csv.writer(f); ww.writerow(["subject", "lopo_tau", "lopo_prec", "lopo_rec", "lopo_spec", "lopo_fpr"])
    for s in SUBJ: ww.writerow([s, taus[s], *lopo[s]])
open(os.path.join(_AG, "logs", f"OPERATING_POINT_REPORT{TAG}.md"), "w").write("\n".join(L) + "\n")
print("wrote logs/OPERATING_POINT_REPORT.md + tradeoff figure + CSV"); print("\n".join(L))
