# exp_sigma_threshold.py  — σ-normalized threshold (Contribution 3 payoff)
"""Pure re-analysis. Test whether τ_s = k·σ_s equalizes the per-subject FPR that
an absolute threshold (τ=3.424°, b182219) leaves varying with σ. σ_s = s0 P1
support-LOO mean at pc=0 (reliable estimator, bb26eef). Normalized threshold =
operating-point machinery on r_norm = resid/σ_s. Outputs: dated markdown,
per-subject FPR figure, CSV."""
import csv, json, os
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
_HERE = os.path.dirname(os.path.abspath(__file__)); _AG = os.path.dirname(_HERE)
SUBJ = [f"p{i:02d}" for i in range(15)]; PCS = [0.1, 0.2, 0.4]; TAU_ABS = 3.424; RNG = np.random.RandomState(0)


def youden(labels, scores):
    labels = np.asarray(labels, bool); scores = np.asarray(scores, float)
    ok = ~np.isnan(scores); labels, scores = labels[ok], scores[ok]
    o = np.argsort(-scores); labels, scores = labels[o], scores[o]
    P = labels.sum(); N = len(labels) - P
    if P == 0 or N == 0: return np.nan
    tp = np.cumsum(labels); fp = np.cumsum(~labels)
    J = tp / P + (N - fp) / N - 1
    return float(scores[int(np.argmax(J))])


def rates(labels, scores, t):
    labels = np.asarray(labels, bool); scores = np.asarray(scores, float)
    ok = ~np.isnan(scores); labels, scores = labels[ok], scores[ok]
    f = scores >= t
    tp = int((labels & f).sum()); fp = int((~labels & f).sum())
    fn = int((labels & ~f).sum()); tn = int((~labels & ~f).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    return prec, rec, fpr


# σ_s = s0 P1 support-LOO mean
d = defaultdict(list)
for r in csv.DictReader(open(os.path.join(_AG, "logs", "sigma_procA", "P1_support_loo.csv"))):
    if r["backbone"] == "s0": d[r["subject"]].append(float(r["residual_deg"]))
sig = {s: float(np.mean(d[s])) for s in SUBJ}

recs = [r for r in (json.loads(l) for l in open(os.path.join(_HERE, "exp_a_records.jsonl")))
        if r["model"] == "structured_swap" and r["seed"] in (0, 1, 2)]
recs += [r for r in (json.loads(l) for l in open(os.path.join(_HERE, "exp_seed_extension_records.jsonl")))
         if r["model"] == "structured_swap"]
for r in recs:
    r["rnorm"] = None if r.get("score_small_loss") is None else r["score_small_loss"] / sig[r["subject"]]


def cols(rows, key):
    return [r["is_corrupted"] for r in rows], [r.get(key) if key == "score_small_loss" else r[key] for r in rows]


L = []; w = lambda *x: L.append(" ".join(str(t) for t in x))
w("# σ-normalized threshold — Contribution 3 design-rule test (§6.5.4)\n")
w("τ_s = k·σ_s, σ_s = s0 P1 support-LOO mean (bb26eef). Normalized = Youden on resid/σ_s. "
  "Baseline = absolute τ=3.424° (b182219). Structured donor-swap, seeds 0-9, 15 subjects.\n")

# 1. LOPO k
ks = []
for s in SUBJ:
    tr = [r for r in recs if r["subject"] != s]
    ks.append(youden([r["is_corrupted"] for r in tr], [r["rnorm"] for r in tr]))
k = float(np.median(ks))
w(f"## 1. LOPO-selected k : median **{k:.3f}**, range [{min(ks):.3f}, {max(ks):.3f}], "
  f"IQR [{np.percentile(ks,25):.3f}, {np.percentile(ks,75):.3f}]  (τ_s = {k:.2f}·σ_s)\n")

# 2. per-subject per-pc, both thresholds
def per(pc, mode):
    key = "score_small_loss" if mode == "abs" else "rnorm"; t = TAU_ABS if mode == "abs" else k
    out = {}
    for s in SUBJ:
        rows = [r for r in recs if r["subject"] == s and abs(r["pc"] - pc) < 1e-9]
        y, v = cols(rows, key); out[s] = rates(y, v, t)
    return out


w("## 2. Absolute vs σ-normalized — per p_c (mean, and the FPR spread)\n")
w("| p_c | thr | recall | precision | FPR mean | **FPR SD** | FPR range | SD ratio norm/abs |")
w("|---|---|---|---|---|---|---|---|")
for pc in PCS:
    r = {}
    for mode in ("abs", "norm"):
        M = np.array([per(pc, mode)[s] for s in SUBJ], float)  # prec,rec,fpr
        r[mode] = M
        w(f"| {pc} | {mode} | {np.nanmean(M[:,1]):.2f} | {np.nanmean(M[:,0]):.2f} | {np.nanmean(M[:,2]):.3f} | "
          f"{np.nanstd(M[:,2]):.3f} | [{np.nanmin(M[:,2]):.2f},{np.nanmax(M[:,2]):.2f}] | "
          f"{(np.nanstd(r['norm'][:,2])/np.nanstd(r['abs'][:,2]) if mode=='norm' else '') } |")
w("")

# 3. detection cost pooled
w("## 3. Detection cost of normalization (pooled p_c)\n")
w("| threshold | mean recall | mean FPR |"); w("|---|---|---|")
for mode in ("abs", "norm"):
    rec = np.nanmean([per(pc, mode)[s][1] for pc in PCS for s in SUBJ])
    fpr = np.nanmean([per(pc, mode)[s][2] for pc in PCS for s in SUBJ])
    w(f"| {mode} | {rec:.3f} | {fpr:.3f} |")
w("")

# 4. FPR ~ sigma regression
w("## 4. FPR ~ σ_s slope (pc=0.2) — absolute should be +, normalized should flatten\n")
w("| threshold | slope | 95% CI |"); w("|---|---|---|")
x = np.array([sig[s] for s in SUBJ])
for mode in ("abs", "norm"):
    y = np.array([per(0.2, mode)[s][2] for s in SUBJ]); m = ~np.isnan(y); xx, yy = x[m], y[m]
    sl = np.polyfit(xx, yy, 1)[0]
    bs = [np.polyfit(xx[i], yy[i], 1)[0] for i in [RNG.randint(0, len(xx), len(xx)) for _ in range(3000)]]
    w(f"| {mode} | {sl:+.4f} | [{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}] |")
w("")

# figure + csv
plt.figure(figsize=(6, 4))
for mode, mk, c in [("abs", "o", "C0"), ("norm", "s", "C1")]:
    y = [per(0.2, mode)[s][2] for s in SUBJ]; plt.scatter(x, y, marker=mk, color=c, label=f"{mode} (pc0.2)")
    sl, ic = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 20); plt.plot(xs, sl*xs+ic, c+"--")
plt.xlabel("σ_s = mean personalized error (deg)"); plt.ylabel("per-subject FPR (pc0.2)")
plt.legend(); plt.title("FPR vs σ: absolute vs σ-normalized threshold"); plt.tight_layout()
plt.savefig(os.path.join(_AG, "logs", "sigma_threshold_fpr.png"), dpi=120); plt.close()
with open(os.path.join(_AG, "logs", "sigma_threshold_per_subject.csv"), "w", newline="") as f:
    ww = csv.writer(f); ww.writerow(["subject", "sigma_s", "pc", "abs_fpr", "abs_recall", "norm_fpr", "norm_recall"])
    for pc in PCS:
        a, n = per(pc, "abs"), per(pc, "norm")
        for s in SUBJ: ww.writerow([s, round(sig[s], 3), pc, round(a[s][2], 3), round(a[s][1], 3), round(n[s][2], 3), round(n[s][1], 3)])

open(os.path.join(_AG, "logs", "SIGMA_THRESHOLD_REPORT.md"), "w").write("\n".join(L) + "\n")
print("wrote logs/SIGMA_THRESHOLD_REPORT.md + figure + CSV")
print("\n".join(L))
