"""04_deployment_tradeoffs.png: (a) paired cost difference LOO - resub (w=4) per condition and p_c; (b) the same across w;
(c) held-out MAE gain over no gate, resubstitution vs LOO (Youden rule), by level."""
import csv, json, os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = [r for r in csv.DictReader(open(os.path.join(REV, "results", "04_cost_sensitivity.csv"))) if r["scorer"] == "LOO"]
M = list(csv.DictReader(open(os.path.join(REV, "03_gate_transfer_results.csv"))))
RT = json.load(open(os.path.join(REV, "results", "04_runtime.json")))
C1, C2, C3, INK, MUT = "#2a78d6", "#eb6834", "#1baf7a", "#222", "#777"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUT})
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.5, 1, 1]})
conds = []
for r in R:
    k = (r["level"], r["condition"])
    if k not in conds: conds.append(k)
a = ax[0]; cols = {"0.1000": C1, "0.2000": C2, "0.4000": C3}
for j, pc in enumerate(["0.1000", "0.2000", "0.4000"]):
    for i, k in enumerate(conds):
        r = next(x for x in R if (x["level"], x["condition"]) == k and x["pc"] == pc and x["w"] == "4")
        y = i + (j - 1) * 0.22; m, lo, hi = float(r["dcost_LOO_minus_resub"]), float(r["dcost_lo"]), float(r["dcost_hi"])
        a.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="o", color=cols[pc], ms=5, lw=1.5, capsize=2, label=f"p_c {float(pc):.1f}" if i == 0 else None)
a.axvline(0, color=MUT, lw=0.8, ls="--"); a.set_yticks(range(len(conds))); a.set_yticklabels([f"{l} {c}" for l, c in conds]); a.invert_yaxis()
a.set_xlabel("Δ cost / session, LOO − resub (c_fn = 4)\n← LOO cheaper     resub cheaper →"); a.legend(frameon=False, fontsize=8, loc="upper left")
a.set_title("(a) which scorer is cheaper", loc="left", fontsize=9)
a = ax[1]
for k, col, mk in ((("L0", "controlled δ8–10"), C1, "o"), (("L0", "controlled δ3–6"), C2, "s"), (("L2", "structured"), C3, "^")):
    rr = sorted([x for x in R if (x["level"], x["condition"]) == k and x["pc"] == "0.2000"], key=lambda x: int(x["w"]))
    w = [int(x["w"]) for x in rr]; m = np.array([float(x["dcost_LOO_minus_resub"]) for x in rr])
    lo = np.array([float(x["dcost_lo"]) for x in rr]); hi = np.array([float(x["dcost_hi"]) for x in rr])
    a.errorbar(w, m, yerr=[m - lo, hi - m], color=col, marker=mk, lw=2, ms=5, capsize=2, label=f"{k[0]} {k[1]}")
a.axhline(0, color=MUT, lw=0.8, ls="--"); a.set_xscale("log", base=2); a.set_xticks([1, 2, 4, 8]); a.set_xticklabels(["1", "2", "4", "8"])
a.set_xlabel("weight of a missed frame, c_fn"); a.set_ylabel("Δ cost, LOO − resub (p_c 0.2)"); a.legend(frameon=False, fontsize=7)
a.set_title("(b) sensitivity to the cost weight", loc="left", fontsize=9)
a = ax[2]; L = ["L0", "L1", "L2", "L3"]; x = np.arange(4)
for i, (p, col, lab) in enumerate((("D", C1, "resub (Youden)"), ("LOO_D", C2, "LOO (Youden)"), ("oracle_true", C3, "true corrupted replaced"))):
    rr = [next(r for r in M if r["level"] == l and r["policy"] == p and r["pc"] == "pooled") for l in L]
    m = -np.array([float(r["mae_minus_naive"]) for r in rr]); lo = -np.array([float(r["mmn_hi"]) for r in rr]); hi = -np.array([float(r["mmn_lo"]) for r in rr])
    a.bar(x + (i - 1) * 0.27, m, 0.25, color=col, label=lab); a.errorbar(x + (i - 1) * 0.27, m, yerr=[m - lo, hi - m], fmt="none", ecolor=INK, lw=0.8, capsize=2)
a.axhline(0, color=MUT, lw=0.8); a.set_xticks(x); a.set_xticklabels(L); a.set_ylabel("held-out error reduced vs no gate (°)")
a.legend(frameon=False, fontsize=7); a.set_title("(c) accuracy bought", loc="left", fontsize=9)
a.text(0.02, -0.28, f"per session on CPU: resub {RT['resub']['median_s']:.1f} s, LOO {RT['loo']['median_s']:.1f} s ({RT['loo_over_resub']:.1f}×)", transform=a.transAxes, fontsize=7, color=MUT)
fig.tight_layout(); fig.savefig(os.path.join(REV, "04_deployment_tradeoffs.png"), dpi=200); print("ok")
