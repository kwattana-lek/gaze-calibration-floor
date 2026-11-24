"""Step 2: detectability floor of the resubstitution residual vs LOO, same trials.

Reads exp_a_records.jsonl (random offset, delta 3..10, pc .1/.2/.4, seeds 0-2, 15 subjects,
20-step adaptation; both scorers on identical frames). Uses the manuscript's isotonic
crossing and e_hat convention (imported from exp_isotonic_floor.py, not re-implemented).
Subject-level bootstrap, both scorers resampled jointly so the floor difference is paired.
Donor-swap displacement distribution regenerated from the stored seeding (seeds 0-9).
Writes only into revision_2026-11/results.
"""
import collections, csv, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REV = os.path.dirname(HERE); LN = os.path.dirname(REV); AG = os.path.dirname(LN)
for p in (LN, os.path.join(AG, "scripts"), AG): sys.path.insert(0, p)
from exp_isotonic_floor import curve_and_crossings, e_hat          # manuscript estimator, unchanged
from unigaze_arm import roc_auc
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

B = 2000; RS = np.random.RandomState(0); OUT = os.path.join(REV, "results")
SCORERS = [("score_loo", "LOO"), ("score_small_loss", "resub")]; PCS = [0.1, 0.2, 0.4]
TAG = os.environ.get("GAZE_TAG", "")
R = [json.loads(l) for l in open(os.path.join(LN, f"exp_a_records{TAG}.jsonl"))]


def cell(pc, key):
    out = collections.defaultdict(lambda: collections.defaultdict(lambda: ([], [])))
    for r in R:
        if r["model"] != "random_offset" or abs(r["pc"] - pc) > 1e-9 or r.get(key) is None: continue
        d = out[r["subject"]][float(r["delta"])]; d[0].append(float(r[key])); d[1].append(bool(r["is_corrupted"]))
    return {s: {d: (np.array(a), np.array(b)) for d, (a, b) in dd.items()} for s, dd in out.items()}


def eh(pc):  # manuscript convention: clean-frame LOO residual, delta<=10, this pc
    return cell(pc, "score_loo")


def lin_cross(x, y, t):  # Table 3 method: linear interpolation on raw pooled AUC
    for i in range(1, len(x)):
        if y[i - 1] < t <= y[i]: return x[i - 1] + (t - y[i - 1]) / (y[i] - y[i - 1]) * (x[i] - x[i - 1])
    return np.nan if y[0] < t else x[0]


def fl(cr, t): return cr[t][1] if cr[t][0] == "ok" else np.nan


# ---- donor-swap displacement distribution, seeds 0-9 (no model needed) ----
def donor_offsets():
    import backbone_api as api, corruption as corr, hashlib
    cache = os.path.join(OUT, "step2_donor_offsets.json")
    if TAG:  # other corpora: realised offsets stored in the Exp A structured records (seeds 0-2)
        res = collections.defaultdict(list)
        for r in R:
            if r["model"] == "structured_swap" and r["is_corrupted"]: res[r["subject"]].append(r["actual_offset_deg"])
        return {k: np.array(v) for k, v in res.items()}
    if os.path.exists(cache): return {k: np.array(v) for k, v in json.load(open(cache)).items()}
    res = collections.defaultdict(list)
    for s in sorted({r["subject"] for r in R}):
        sup = api.load_subject_support(s, K=9, seed=42)
        for sd in range(10):
            for pc in PCS:
                k = f"{s}|structured_swap|None|{pc}|{sd}".encode()
                g = np.random.RandomState(int.from_bytes(hashlib.sha256(k).digest()[:8], "big") % (2 ** 32))
                c = corr.apply_corruption(sup["labels"], model="structured_swap", rng=g, pc=pc, delta_deg=None, pool_labels=sup["pool_labels"])
                res[s] += [float(o) for o, m in zip(c["actual_offset_deg"], c["is_corrupted"]) if m]
    # validate against exp_a stored offsets (seeds 0-2)
    st = collections.defaultdict(list)
    for r in R:
        if r["model"] == "structured_swap" and r["is_corrupted"]: st[r["subject"]].append(r["actual_offset_deg"])
    json.dump({k: v for k, v in res.items()}, open(cache, "w"))
    return {k: np.array(v) for k, v in res.items()}


def main():
    rows, curves, summ = [], {}, {}
    offs = donor_offsets(); subj_all = sorted(offs)
    for pc in PCS:
        cE = eh(pc); cells = {nm: cell(pc, k) for k, nm in SCORERS}; S = sorted(cE)
        e0 = e_hat(cE, S); pt = {}
        for nm, c in cells.items():
            x, y, yf, cr = curve_and_crossings(c, S)
            pt[nm] = dict(x=x, y=y, yf=yf, f7=fl(cr, 0.7), f9=fl(cr, 0.9), f7s=cr[0.7][0], f9s=cr[0.9][0],
                          lin7=lin_cross(x, y, 0.7), lin9=lin_cross(x, y, 0.9))
        curves[pc] = pt
        # paired subject bootstrap
        bs = collections.defaultdict(list)
        for b in range(B):
            Sb = list(RS.choice(S, len(S))); e = e_hat(cE, Sb); bs["e"].append(e)
            fb = {}
            for nm, c in cells.items():
                _, _, _, cr = curve_and_crossings(c, Sb); fb[nm] = (fl(cr, 0.7), fl(cr, 0.9))
                bs[nm + "_f7"].append(fb[nm][0]); bs[nm + "_f9"].append(fb[nm][1]); bs[nm + "_c"].append(fb[nm][0] / e)
            bs["d_f7"].append(fb["resub"][0] - fb["LOO"][0]); bs["d_c"].append((fb["resub"][0] - fb["LOO"][0]) / e)
            # fraction of donor-swap corruptions below each floor (subjects resampled with the floor)
            od = np.concatenate([offs[s] for s in Sb if s in offs])
            for nm in ("LOO", "resub"): bs[nm + "_below"].append(np.mean(od < fb[nm][0]) if np.isfinite(fb[nm][0]) else np.nan)
        ci = lambda v: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)), float(np.mean(~np.isfinite(v))))
        od_all = np.concatenate([offs[s] for s in S])
        for nm in ("LOO", "resub"):
            p = pt[nm]; below = float(np.mean(od_all < p["f7"])) if np.isfinite(p["f7"]) else np.nan
            per_sub = [float(np.mean(offs[s] < p["f7"])) for s in S] if np.isfinite(p["f7"]) else [np.nan]
            row = dict(pc=pc, scorer=nm, e_hat=e0, e_ci=ci(bs["e"])[:2],
                       floor07_iso=p["f7"], floor07_status=p["f7s"], floor07_ci=ci(bs[nm + "_f7"]),
                       floor09_iso=p["f9"], floor09_status=p["f9s"], floor09_ci=ci(bs[nm + "_f9"]),
                       floor07_lin=p["lin7"], floor09_lin=p["lin9"], c=p["f7"] / e0, c_ci=ci(bs[nm + "_c"]),
                       below_floor_frac=below, below_ci=ci(bs[nm + "_below"])[:2], below_per_subject_range=(min(per_sub), max(per_sub)),
                       auc_by_delta={float(d): float(a) for d, a in zip(p["x"], p["y"])})
            rows.append(row)
        summ[pc] = dict(d_floor=pt["resub"]["f7"] - pt["LOO"]["f7"], d_floor_ci=ci(bs["d_f7"]), d_c_ci=ci(bs["d_c"]))
    # paired per-subject AUC differences per delta (pc pooled within subject? no: per pc, as in Table 5)
    paired = []
    for pc in PCS:
        for d in sorted({r["delta"] for r in R if r["model"] == "random_offset"}):
            a = {k: collections.defaultdict(lambda: ([], [])) for k, _ in SCORERS}
            for r in R:
                if r["model"] == "random_offset" and abs(r["pc"] - pc) < 1e-9 and r["delta"] == d:
                    for k, _ in SCORERS: a[k][r["subject"]][0].append(r[k]); a[k][r["subject"]][1].append(r["is_corrupted"])
            ss = sorted(a["score_loo"]); L = np.array([roc_auc(*map(np.array, a["score_loo"][s])) for s in ss]); Rr = np.array([roc_auc(*map(np.array, a["score_small_loss"][s])) for s in ss])
            ok = np.isfinite(L) & np.isfinite(Rr); dd = L[ok] - Rr[ok]
            p = wilcoxon(L[ok], Rr[ok]).pvalue if np.any(dd != 0) else np.nan
            paired.append(dict(pc=pc, delta=d, n=int(ok.sum()), auc_loo=float(L[ok].mean()), auc_resub=float(Rr[ok].mean()), diff=float(dd.mean()), p_wilcoxon=float(p)))
    # ---- outputs ----
    with open(os.path.join(OUT, f"02_floor_comparison{TAG}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pc", "scorer", "e_hat", "floor07_isotonic", "floor07_lo", "floor07_hi", "floor07_censored_frac",
                                        "floor09_isotonic", "floor09_lo", "floor09_hi", "floor07_linear", "floor09_linear", "c", "c_lo", "c_hi",
                                        "donor_below_floor07", "below_lo", "below_hi"])
        for r in rows:
            w.writerow([r["pc"], r["scorer"], f"{r['e_hat']:.3f}", f"{r['floor07_iso']:.3f}", f"{r['floor07_ci'][0]:.3f}", f"{r['floor07_ci'][1]:.3f}", f"{r['floor07_ci'][2]:.3f}",
                        f"{r['floor09_iso']:.3f}", f"{r['floor09_ci'][0]:.3f}", f"{r['floor09_ci'][1]:.3f}", f"{r['floor07_lin']:.3f}", f"{r['floor09_lin']:.3f}",
                        f"{r['c']:.3f}", f"{r['c_ci'][0]:.3f}", f"{r['c_ci'][1]:.3f}", f"{r['below_floor_frac']:.4f}", f"{r['below_ci'][0]:.4f}", f"{r['below_ci'][1]:.4f}"])
    json.dump(dict(rows=rows, diff=summ, paired=paired, B=B), open(os.path.join(OUT, f"02_floor_comparison{TAG}.json"), "w"), indent=1, default=float)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, pc in zip(axs, PCS):
        for nm, col in (("LOO", "#1f6fb4"), ("resub", "#d1562b")):
            p = curves[pc][nm]; ax.plot(p["x"], p["y"], "o", color=col, ms=4); ax.plot(p["x"], p["yf"], "-", color=col, label=f"{nm} (floor {p['f7']:.2f}°)" if np.isfinite(p['f7']) else nm)
        ax.axhline(0.7, ls=":", c="grey"); ax.axhline(0.9, ls=":", c="grey"); ax.set_title(f"p_c = {pc}"); ax.set_xlabel("label displacement δ (°)"); ax.legend(fontsize=8, loc="lower right")
    axs[0].set_ylabel("pooled ROC-AUC"); fig.suptitle("Resubstitution vs LOO residual, identical trials (Exp A records, 20-step adaptation)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"02_floor_comparison{TAG}.png"), dpi=150)
    for r in rows: print(f"pc={r['pc']} {r['scorer']:5s} ê={r['e_hat']:.2f} floor07 iso={r['floor07_iso']:.2f} {np.round(r['floor07_ci'],2)} lin={r['floor07_lin']:.2f} | floor09={r['floor09_iso']:.2f} {r['floor09_status']} lin={r['floor09_lin']:.2f} | c={r['c']:.3f} {np.round(r['c_ci'][:2],3)} | below={r['below_floor_frac']:.3f} {np.round(r['below_ci'],3)}")
    for pc, s in summ.items(): print(f"pc={pc} Δfloor(resub−LOO)={s['d_floor']:.2f} CI {np.round(s['d_floor_ci'][:2],2)} censored {s['d_floor_ci'][2]:.3f}; Δc CI {np.round(s['d_c_ci'][:2],3)}")
    for p in paired: print(p)


if __name__ == "__main__":
    main()
