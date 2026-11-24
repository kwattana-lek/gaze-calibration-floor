"""03_threshold_comparison.png: (a) thresholds vs e_clean per level; (b) pooled per-subject ROC-AUC; (c) expected cost (w=4)."""
import json, csv, os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REV = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P = lambda f: os.path.join(REV, "results", f)
EH = json.load(open(P("step3_ehat.json"))); TH = json.load(open(P("step3_thresholds.json")))
DET = list(csv.DictReader(open(P("03_gate_transfer_detection.csv")))); AUC = list(csv.DictReader(open(P("03_auc.csv"))))
L = ["L0", "L1", "L2", "L3"]; e = np.array([EH[l]["e_clean"] for l in L])
C1, C2, C3, C4, INK, MUT = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#222", "#777"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUT, "axes.labelcolor": INK})
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
a = ax[0]
a.plot(e, [3.42] * 4, "-", color=C1, lw=2, marker="o", ms=6, label="A fixed 3.42°")
a.plot(e, [TH[l]["B"] for l in L], "-", color=C2, lw=2, marker="s", ms=6, label="B 0.96·ê")
Dm = [float(next(r["tau"] for r in DET if r["level"] == l and r["policy"] == "D" and r["pc"] == "pooled")) for l in L]
a.plot(e, Dm, "-", color=C3, lw=2, marker="^", ms=7, label="D Youden (LOPO mean)")
for pc, mk in zip(["0.1", "0.2", "0.4"], ["x", "+", "1"]):
    a.plot(e, [min(TH[l]["C"][pc], 12) for l in L], ":", color=C4, lw=1.5, marker=mk, ms=8, label=f"C oracle, p_c {pc}")
a.set_xlabel("ê_clean (°)"); a.set_ylabel("threshold τ on resub. residual (°)"); a.set_ylim(0, 12.5)
for x, l in zip(e, L): a.annotate(l, (x, 12.2), ha="center", color=MUT, fontsize=8)
a.legend(frameon=False, fontsize=7, loc="center right"); a.set_title("(a) thresholds vs model error", loc="left", fontsize=9)
a = ax[1]
for sc, col, mk, lab in (("resub", C1, "o", "resubstitution"), ("loo", C2, "s", "LOO")):
    rr = [next(r for r in AUC if r["level"] == l and r["pc"] == "pooled" and r["scorer"] == sc) for l in L]
    m = np.array([float(r["auc"]) for r in rr]); lo = np.array([float(r["lo"]) for r in rr]); hi = np.array([float(r["hi"]) for r in rr])
    a.errorbar(e, m, yerr=[m - lo, hi - m], color=col, marker=mk, ms=6, lw=2, capsize=3, label=lab)
a.axhline(0.5, color=MUT, lw=0.8, ls="--"); a.set_ylim(0.45, 1); a.set_xlabel("ê_clean (°)"); a.set_ylabel("per-subject ROC-AUC (95% CI)")
a.legend(frameon=False, fontsize=8); a.set_title("(b) donor-swap detectability", loc="left", fontsize=9)
a = ax[2]; pols = [("A", C1), ("B", C2), ("D", C3), ("C", C4)]; w = 0.2
for i, (p, col) in enumerate(pols):
    rr = [next(r for r in DET if r["level"] == l and r["policy"] == p and r["pc"] == "pooled") for l in L]
    m = np.array([float(r["cost"]) for r in rr]); lo = np.array([float(r["cost_lo"]) for r in rr]); hi = np.array([float(r["cost_hi"]) for r in rr])
    a.bar(np.arange(4) + (i - 1.5) * w, m, w * 0.9, color=col, label={"A": "A fixed", "B": "B scaled", "D": "D Youden LOPO", "C": "C oracle"}[p])
    a.errorbar(np.arange(4) + (i - 1.5) * w, m, yerr=[m - lo, hi - m], fmt="none", ecolor=INK, lw=0.8, capsize=2)
a.set_xticks(range(4)); a.set_xticklabels([f"{l}\nê={x:.1f}°" for l, x in zip(L, e)]); a.set_ylabel("expected cost / session (c_fn = 4)")
a.legend(frameon=False, fontsize=7); a.set_title("(c) cost, pooled over p_c", loc="left", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(REV, "03_threshold_comparison.png"), dpi=200); print("ok")
