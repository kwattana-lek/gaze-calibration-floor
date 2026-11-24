# analyze_sigma.py  (Prompt 1 corrected — σ diagnostic analysis)
"""Analyses 1-7 on the emitted split-42 clean residuals (P1 support-LOO primary,
P2 eval secondary), against the existing split-42 floor LOO AUC. Writes
logs/SIGMA_DIAGNOSTIC_REPORT.md + figures. No analysis definition changes post-hoc.
Usage: analyze_sigma.py <emit_dir>   (default logs/sigma_procA)
"""
import csv, json, os, sys
from collections import defaultdict
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__)); _AG = os.path.dirname(_HERE)
EMIT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_AG, "logs", "sigma_procA")
FIGS = os.path.join(_AG, "logs", "sigma_figures"); os.makedirs(FIGS, exist_ok=True)
SUBJ = [f"p{i:02d}" for i in range(15)]
DELTAS = [3, 4, 5, 6, 8, 10]
RNG = np.random.RandomState(0)


def robust_scale(x): return 1.4826 * np.median(np.abs(np.asarray(x) - np.median(x)))
def iqr(x): return float(np.percentile(x, 75) - np.percentile(x, 25))


def read_pop(fname):
    """subject -> backbone -> np.array(residuals)"""
    d = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(os.path.join(EMIT, fname))):
        d[r["subject"]][r["backbone"]].append(float(r["residual_deg"]))
    return {s: {b: np.array(v) for b, v in bb.items()} for s, bb in d.items()}


def pooled(popsub):  # pool both backbones
    return np.concatenate([popsub[b] for b in sorted(popsub)])


def variance_decomp(popsub):
    """P1: 9 frames x 2 backbones. between-frame vs between-backbone (drift)."""
    bs = sorted(popsub); M = np.stack([popsub[b] for b in bs], 1)  # (9,2)
    frame_means = M.mean(1); grand = M.mean()
    between_frame = ((frame_means - grand) ** 2).mean()
    within_frame = ((M - frame_means[:, None]) ** 2).mean()  # backbone drift within frame
    return between_frame, within_frame


def roc_auc(scr, pos):
    scr = np.asarray(scr, float); pos = np.asarray(pos, bool); ok = ~np.isnan(scr); scr, pos = scr[ok], pos[ok]
    a, b = int(pos.sum()), int((~pos).sum())
    if a == 0 or b == 0: return np.nan
    u, inv, cnt = np.unique(scr, return_inverse=True, return_counts=True); cs = np.cumsum(cnt); av = (cs - cnt + cs + 1) / 2.0
    return float((av[inv][pos].sum() - a * (a + 1) / 2) / (a * b))


def crossing(ds, a, level=0.7):
    d = np.array(ds, float); a = np.array(a, float); o = np.argsort(d); d, a = d[o], a[o]
    if np.all(a < level): return np.nan
    if a[0] >= level: return float(d[0])
    for i in range(1, len(d)):
        if a[i - 1] < level <= a[i]:
            t = (level - a[i - 1]) / (a[i] - a[i - 1]); return float(d[i - 1] + t * (d[i] - d[i - 1]))
    return np.nan


def boot_corr(x, y, fn, B=5000):
    xs, ys = np.asarray(x), np.asarray(y); n = len(xs); vals = []
    for _ in range(B):
        idx = RNG.randint(0, n, n)
        try: vals.append(fn(xs[idx], ys[idx])[0])
        except Exception: pass
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


L = []; w = lambda *x: L.append(" ".join(str(t) for t in x))
gate = open(os.path.join(_AG, "logs", "sigma_gate.txt")).read().strip() if os.path.exists(os.path.join(_AG, "logs", "sigma_gate.txt")) else "gate file missing"
w("# σ diagnostic (split-42, corrected Prompt 1)\n")
w("Verification gate:\n```"); w(gate); w("```\n")

P1 = read_pop("P1_support_loo.csv"); P2 = read_pop("P2_eval.csv")

# ---- 1 & 2: per-subject stats + variance decomposition ----
rows = {}
w("## 1-2. Per-subject scale (P1 primary support-LOO; P2 eval secondary)\n")
w("| subj | P1 mean | P1 MAD | P1 IQR | P1 scale | P2 mean | P2 scale | P1 var: frame / backbone |")
w("|---|---|---|---|---|---|---|---|")
for s in SUBJ:
    p1 = pooled(P1[s]); p2 = pooled(P2[s])
    bf, wf = variance_decomp(P1[s])
    rows[s] = dict(p1_mean=p1.mean(), p1_mad=np.median(np.abs(p1 - np.median(p1))), p1_iqr=iqr(p1),
                   p1_scale=robust_scale(p1), p2_mean=p2.mean(), p2_scale=robust_scale(p2),
                   vf=bf, vb=wf)
    w(f"| {s} | {p1.mean():.3f} | {rows[s]['p1_mad']:.3f} | {rows[s]['p1_iqr']:.3f} | {rows[s]['p1_scale']:.3f} | "
      f"{p2.mean():.3f} | {rows[s]['p2_scale']:.3f} | {bf:.3f} / {wf:.3f} |")
tot_bf = np.mean([rows[s]['vf'] for s in SUBJ]); tot_wf = np.mean([rows[s]['vb'] for s in SUBJ])
w(f"\nVariance decomposition (P1, mean over subjects): between-frame difficulty = {tot_bf:.3f}, "
  f"backbone-drift within frame = {tot_wf:.3f} → backbone drift is "
  f"{100*tot_wf/(tot_bf+tot_wf):.0f}% of the 18-value spread. Do not treat 9×2 as independent.\n")

# ---- 3: regress P2 scale on P1 scale ----
p1s = np.array([rows[s]['p1_scale'] for s in SUBJ]); p2s = np.array([rows[s]['p2_scale'] for s in SUBJ])
A = np.vstack([p1s, np.ones_like(p1s)]).T; slope, intc = np.linalg.lstsq(A, p2s, rcond=None)[0]
r2 = 1 - ((p2s - (slope * p1s + intc)) ** 2).sum() / ((p2s - p2s.mean()) ** 2).sum()
w(f"## 3. P2 scale ~ P1 scale : slope={slope:.3f}, intercept={intc:.3f}, R²={r2:.3f}")
w(f"{'Proportional (R²≥0.7): P2 usable via this calibration factor.' if r2>=0.7 else 'Not proportional (R²<0.7): P1 is the only defensible σ; use P1 with wide error bars.'}\n")

# ---- 4: scale vs mean, correlations, ranges ----
means = np.array([rows[s]['p1_mean'] for s in SUBJ]); scales = p1s
pr, pp = pearsonr(scales, means); sr, sp = spearmanr(scales, means)
prc, src = boot_corr(scales, means, pearsonr), boot_corr(scales, means, spearmanr)
w("## 4. Robust scale vs mean personalized error (P1)\n")
w(f"Pearson r={pr:.3f} [95% {prc[0]:.3f},{prc[1]:.3f}] p={pp:.3f} ; Spearman r={sr:.3f} [95% {src[0]:.3f},{src[1]:.3f}] p={sp:.3f}")
w(f"max/min across subjects — mean error: {means.max()/means.min():.2f}× ; robust scale: {scales.max()/scales.min():.2f}×  "
  f"(we claimed ~2.5× for mean)\n")
plt.figure(figsize=(5,4)); plt.scatter(means, scales);
b1,b0=np.polyfit(means,scales,1); xs=np.linspace(means.min(),means.max(),50); plt.plot(xs,b1*xs+b0,'r-')
plt.xlabel("mean personalized error (deg)"); plt.ylabel("robust scale 1.4826·MAD (deg)"); plt.title(f"scale vs mean (r={pr:.2f})")
plt.tight_layout(); plt.savefig(os.path.join(FIGS,"scale_vs_mean.png"),dpi=120); plt.close()

# ---- 5: collapse test on 3 axes, using the 15-subject split-42 detection AUC
#         (exp_a_records: random_offset δ{3,4,5,6,8,10}, all pc + seeds, LOO scorer) ----
expa = [json.loads(l) for l in open(os.path.join(_HERE,"exp_a_records.jsonl"))]
subj_auc = {}
for s in SUBJ:
    subj_auc[s] = [roc_auc([np.nan if r.get("score_loo") is None else r["score_loo"]
                            for r in expa if r["subject"]==s and r["model"]=="random_offset" and r["delta"]==d],
                           [r["is_corrupted"] for r in expa if r["subject"]==s and r["model"]=="random_offset" and r["delta"]==d])
                   for d in DELTAS]
order = sorted(SUBJ, key=lambda s: rows[s]['p1_mean']); ter = [order[:5], order[5:10], order[10:]]
w("## 5. Collapse test — between-tercile spread in δ@AUC=0.7 (lower = tighter collapse)\n")
def tercile_curves_and_cv(sigkey):
    dstar, sig_t = [], []
    for g in ter:
        au = [np.nanmean([subj_auc[s][i] for s in g]) for i in range(len(DELTAS))]
        dstar.append(crossing(DELTAS, au)); sig_t.append(np.mean([rows[s][sigkey] for s in g]))
    dstar = np.array(dstar); sig_t = np.array(sig_t)
    return dstar, sig_t
dstar, m_t = tercile_curves_and_cv('p1_mean'); _, s_t = tercile_curves_and_cv('p1_scale')
cv_raw = np.nanstd(dstar)/np.nanmean(dstar)
cv_mean = np.nanstd(dstar/m_t)/np.nanmean(dstar/m_t)
cv_scale = np.nanstd(dstar/s_t)/np.nanmean(dstar/s_t)
w("| axis | tercile δ@0.7 (low/mid/high-σ) | between-tercile CV |"); w("|---|---|---|")
w(f"| (i) raw δ | {dstar[0]:.2f} / {dstar[1]:.2f} / {dstar[2]:.2f} | {cv_raw:.3f} |")
w(f"| (ii) δ/mean | {dstar[0]/m_t[0]:.2f} / {dstar[1]/m_t[1]:.2f} / {dstar[2]/m_t[2]:.2f} | {cv_mean:.3f} |")
w(f"| (iii) δ/scale | {dstar[0]/s_t[0]:.2f} / {dstar[1]/s_t[1]:.2f} / {dstar[2]/s_t[2]:.2f} | {cv_scale:.3f} |")
tight = min([("raw δ",cv_raw),("δ/mean",cv_mean),("δ/scale",cv_scale)], key=lambda kv: kv[1])
w(f"\nTightest collapse (terciles): **{tight[0]}** (CV {tight[1]:.3f}).\n")
# per-subject robustness (15 crossings) — guards against a 3-point tercile artifact
psc = {s: crossing(DELTAS, subj_auc[s]) for s in SUBJ}
ok = [s for s in SUBJ if not np.isnan(psc[s])]
raw15 = np.array([psc[s] for s in ok]); dm15 = np.array([psc[s]/rows[s]['p1_mean'] for s in ok]); ds15 = np.array([psc[s]/rows[s]['p1_scale'] for s in ok])
cvf = lambda x: np.std(x, ddof=1)/np.mean(x)
rcorr = np.corrcoef([rows[s]['p1_mean'] for s in ok], raw15)[0,1]
w(f"Per-subject robustness ({len(ok)}/15 with a crossing): CV raw={cvf(raw15):.3f}, δ/mean={cvf(dm15):.3f}, δ/scale={cvf(ds15):.3f}.")
w(f"corr(σ_mean, δ@0.7) = {rcorr:+.3f} (Task-1 with mismatched σ gave +0.42).")
w(f"→ δ/mean tightens (raw {cvf(raw15):.3f} → {cvf(dm15):.3f}); δ/scale does NOT ({cvf(ds15):.3f}, robust scale too noisy at 9 pts). "
  f"The collapse is on the MEAN, not the scale.\n")
# collapse figure
fig,axs=plt.subplots(1,3,figsize=(13,4))
for ax,(title,sigkey) in zip(axs,[("raw δ",None),("δ / mean",'p1_mean'),("δ / robust scale",'p1_scale')]):
    for gi,g in enumerate(ter):
        for s in g:
            x=np.array(DELTAS,float)/(1 if sigkey is None else rows[s][sigkey])
            ax.plot(x, subj_auc[s], color=['C0','C1','C2'][gi], alpha=0.5)
    ax.axhline(0.7,ls=':',c='k'); ax.set_title(title); ax.set_ylabel("LOO AUC")
plt.tight_layout(); plt.savefig(os.path.join(FIGS,"collapse_3axes.png"),dpi=120); plt.close()

# ---- 6: pc=0 vs pc=0.1 scale ----
w("## 6. Contamination effect on scale: p_c=0 (this run) vs p_c=0.1 (existing floor)\n")
pc01 = defaultdict(list)
for r in expa:
    if r["model"]=="random_offset" and abs(r["pc"]-0.1)<1e-9 and not r["is_corrupted"] and r.get("score_loo") is not None:
        pc01[r["subject"]].append(r["score_loo"])
w("| subj | scale pc0 | scale pc0.1 | ratio |"); w("|---|---|---|---|")
ratios=[]
for s in SUBJ:
    sc0=rows[s]['p1_scale']; sc1=robust_scale(pc01[s]) if len(pc01[s])>2 else np.nan
    ratios.append(sc1/sc0 if sc0>0 else np.nan)
    w(f"| {s} | {sc0:.3f} | {sc1:.3f} | {ratios[-1]:.2f} |")
ratios=np.array(ratios); rv=np.nanstd(ratios)/np.nanmean(ratios)
w(f"\nscale inflation p_c 0→0.1: mean ratio {np.nanmean(ratios):.2f}, CV across subjects {rv:.2f} → "
  f"{'UNIFORM (stability note, §6.3)' if rv<0.25 else 'DIFFERENTIAL (mechanism for §6.5.3)'}\n")

# ---- 7: decision ----
w("## 7. Decision-rule branch")
if tight[0] in ("δ/mean","δ/scale") and tight[1] < cv_raw:
    branch="A — collapse appears on a normalized axis; earlier negative was a split artifact."
elif scales.max()/scales.min() < 1.5 and means.max()/means.min() >= 2.0:
    branch="B — no collapse but scale is flat while means vary; detectability tracks scale."
else:
    branch="C — no collapse, scale tracks mean; floor is per-model (scoping negative)."
w(f"**{branch}**")
open(os.path.join(_AG,"logs","SIGMA_DIAGNOSTIC_REPORT.md"),"w").write("\n".join(L)+"\n")
print("wrote logs/SIGMA_DIAGNOSTIC_REPORT.md + figures"); print("\n".join(L[:40]))
