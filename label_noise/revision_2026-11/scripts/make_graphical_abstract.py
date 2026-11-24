"""Graphical abstract for the IVC submission (one panel, data-derived).

Pooled leave-one-out ROC-AUC against label displacement in units of each backbone's support error e_hat, for the four
accuracy levels L0-L3 of the 150-step sweep (sigma_sweep_records.jsonl: MPIIFaceGaze, 15 subjects, random offset,
p_c = 0.2, seeds 0-2). e_hat = mean over subjects of the mean clean-frame leave-one-out residual (the paper's convention).
The shaded band is the ten-configuration range of the floor, 0.88-1.23 e_hat (Table tab:arch); the gate text repeats
numbers stated in the abstract. Output: manuscript/graphical_abstract.{png,pdf}, 1328 x 531 px minimum (h x w = 5.31 x
13.28 in at 300 dpi, i.e. 1593 x 3984 px).
"""
import json, os, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); REV = os.path.dirname(HERE); LN = os.path.dirname(REV)
sys.path.insert(0, LN)
from unigaze_arm import roc_auc  # the paper's ROC-AUC

R = [json.loads(l) for l in open(os.path.join(LN, "sigma_sweep_records.jsonl"))]
sc = defaultdict(list); lab = defaultdict(list); clean = defaultdict(lambda: defaultdict(list))
for r in R:
    L, d = r["sigma_level"], r["delta"]
    sc[(L, d)].append(r["loo_residual_deg"]); lab[(L, d)].append(bool(r["is_corrupted"]))
    if not r["is_corrupted"]:
        clean[L][r["subject"]].append(r["loo_residual_deg"])
levels = sorted(clean)
ehat = {L: float(np.mean([np.mean(v) for v in clean[L].values()])) for L in levels}
curves = {}
for L in levels:
    ds = sorted({d for (l, d) in sc if l == L})
    curves[L] = [(d / ehat[L], roc_auc(sc[(L, d)], lab[(L, d)])) for d in ds]
json.dump({"e_hat": ehat, "curves": curves}, open(os.path.join(REV, "results", "graphical_abstract_data.json"), "w"), indent=1)

INK, MUTED, GRID, BAND = "#1f2328", "#57606a", "#d0d7de", "#f2e6c9"
BLUES = ["#08306b", "#2171b5", "#6baed6", "#9ecae1"]          # sequential: most to least accurate
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig = plt.figure(figsize=(13.28, 5.31), dpi=300)
ax = fig.add_axes([0.065, 0.16, 0.50, 0.70])
ax.axvspan(0.88, 1.23, color=BAND, zorder=0)
ax.axhline(0.5, color=GRID, lw=1.2, zorder=1); ax.axhline(0.7, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
for L, col in zip(levels, BLUES):
    x, y = zip(*curves[L])
    ax.plot(x, y, color=col, lw=2.6, marker="o", ms=5, zorder=3, label=f"{L}  (ê = {ehat[L]:.1f}°)")
ax.set_xlim(0, 3.0); ax.set_ylim(0.45, 1.0)
ax.set_xlabel("label displacement / model's own error (δ / ê)"); ax.set_ylabel("detection ROC-AUC")
ax.grid(axis="y", color=GRID, lw=0.6); ax.spines[["top", "right"]].set_visible(False)
ax.text(1.055, 0.965, "detectability\nfloor\n0.88–1.23 ê", ha="center", va="top", fontsize=12.5, color=INK)
ax.text(2.98, 0.712, "useful detection (0.7)", ha="right", va="bottom", fontsize=11.5, color=MUTED)
ax.text(1.62, 0.507, "chance (0.5)", ha="left", va="bottom", fontsize=11.5, color=MUTED)
ax.legend(title="ResNet-18 accuracy level", loc="lower right", frameon=False, fontsize=11.5, title_fontsize=11.5)
fig.text(0.065, 0.92, "When can a corrupted calibration frame be caught?", fontsize=19, weight="bold", color=INK)
tx = 0.60
fig.text(tx, 0.80, "A wrong calibration label is detectable only\nwhen its displacement exceeds about the\nmodel's own error ê (leave-one-out residual,\nten configurations, three corpora).",
         fontsize=13.5, color=INK, va="top", linespacing=1.35)
fig.text(tx, 0.47, "A fixed-threshold gate on the cheaper residual\n(τ = 3.42°) costs one to four recaptures per\nnine-frame session and lowers held-out error by\nabout 0.16° at 20% contamination; scaling τ\nwith ê did not help.",
         fontsize=13.5, color=INK, va="top", linespacing=1.35)
out = os.path.join(REV, "manuscript", "graphical_abstract")
fig.savefig(out + ".png", dpi=300); fig.savefig(out + ".pdf")
print({L: round(v, 2) for L, v in ehat.items()}, {L: [round(a, 3) for _, a in c] for L, c in curves.items()})
