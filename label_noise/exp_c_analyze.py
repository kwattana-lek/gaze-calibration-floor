# exp_c_analyze.py — Experiment C aggregation (recovery under recapture budget)
"""Aggregate exp_c_records.csv. Oracle in the SAME table. Per cell: post-adaptation
MAE, paired per-subject Δ vs naive (d_z, p), recovered fraction = (naive-policy)/
(naive-oracle), spend precision, and (two-channel) the action-vs-channel confusion
(image-corrupted DISCARDED = wasteful; label-corrupted RECAPTURED = futile).
Parallel vs cascade kept as separate rows + an explicit ordering test (§6.7).
"""
import csv, os
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon
_HERE = os.path.dirname(os.path.abspath(__file__)); _AG = os.path.dirname(_HERE)
SUBJ = [f"p{i:02d}" for i in range(15)]

rows = list(csv.DictReader(open(os.path.join(_HERE, "exp_c_records.csv"))))
def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return np.nan
for r in rows:
    r["pc"] = float(r["pc"]); r["B"] = int(r["B"]); r["n_corrupted"] = int(float(r["n_corrupted"]))
    for k in ("naive", "discard", "recapture", "oracle", "par", "cascade", "prec_discard",
              "prec_recapture", "prec_par", "img_discarded", "lab_recaptured", "n_acted_par"):
        r[k] = fnum(r.get(k))


def sel(channel, pc, B): return [r for r in rows if r["channel"] == channel and abs(r["pc"] - pc) < 1e-9 and r["B"] == B]
def persubj(cells, key):
    by = defaultdict(list)
    for r in cells: by[r["subject"]].append(r[key])
    return {s: np.nanmean(by[s]) for s in by if np.isfinite(np.nanmean(by[s]))}


def vs_naive(cells, key):
    """paired policy vs naive: mean MAE, mean Δ, d_z, p, recovered frac, n."""
    pol = persubj(cells, key); nai = persubj(cells, "naive"); ora = persubj(cells, "oracle")
    ss = [s for s in SUBJ if s in pol and s in nai and s in ora]
    P = np.array([pol[s] for s in ss]); N = np.array([nai[s] for s in ss]); O = np.array([ora[s] for s in ss])
    d = P - N; dz = np.nanmean(d) / (np.nanstd(d, ddof=1) + 1e-12)
    try: p = wilcoxon(P, N).pvalue
    except Exception: p = np.nan
    # recovered fraction per subject (guard tiny headroom), then mean
    head = N - O; rec = np.where(np.abs(head) > 0.02, (N - P) / head, np.nan)
    return dict(mae=np.nanmean(P), dmae=np.nanmean(d), dz=dz, p=p,
                rec=np.nanmean(rec), n=len(ss), headroom=np.nanmean(head))


def prec(cells, key):
    v = [r[key] for r in cells if np.isfinite(r[key])]; return np.nanmean(v) if v else np.nan


L = []; w = lambda *x: L.append(" ".join(str(t) for t in x))
w("# Experiment C — recovery under a recapture budget (15 subjects, s0)\n")
w("Scorer = resubstitution residual; residual-gate τ=3.42° (fixed, b182219); image gate = held-out LOPO "
  "detector. Recovered fraction = (naive−policy)/(naive−oracle). Δ<0 = improvement. Eval on 1000 held-out "
  "frames/subject. Oracle = perfect-knowledge routing (label→discard, image→recapture) within budget B.\n")

# ---- LABEL channel ----
w("## 1. Label channel (structured donor-swap) — policies 1–3 + oracle\n")
for pc in [0.1, 0.2, 0.4]:
    w(f"### p_c = {pc}")
    w("| B | policy | MAE° | Δ vs naive° | d_z | p | recovered frac | spend precision | n |")
    w("|---|---|---|---|---|---|---|---|---|")
    naive_mae = np.nanmean(list(persubj(sel("label", pc, 0), "naive").values()))
    w(f"| 0 | naive | {naive_mae:.3f} | — | — | — | — | — | 15 |")
    for B in [1, 2]:
        cells = sel("label", pc, B)
        for key, nm, pk in [("discard", "discard-B", "prec_discard"), ("recapture", "recapture-B", "prec_recapture"), ("oracle", "ORACLE", None)]:
            st = vs_naive(cells, key)
            pv = f"{prec(cells, pk):.2f}" if pk else "—"
            w(f"| {B} | {nm} | {st['mae']:.3f} | {st['dmae']:+.3f} | {st['dz']:+.2f} | {st['p']:.1e} | {st['rec']:.2f} | {pv} | {st['n']} |")
    w(f"\n(headroom naive−oracle at B=2: {vs_naive(sel('label',pc,2),'oracle')['headroom']:+.3f}°)\n")

# ---- TWO-CHANNEL ----
w("## 2. Two-channel (label+image mix, p_c=0.2) — policies 4 (parallel, cascade) + 5 oracle\n")
w("| B | policy | MAE° | Δ vs naive° | d_z | p | recovered frac | spend precision | n |")
w("|---|---|---|---|---|---|---|---|---|")
naive_mae = np.nanmean(list(persubj(sel("two", 0.2, 0), "naive").values()))
w(f"| 0 | naive | {naive_mae:.3f} | — | — | — | — | — | 15 |")
for B in [1, 2]:
    cells = sel("two", 0.2, B)
    for key, nm, pk in [("par", "two-ch PARALLEL", "prec_par"), ("cascade", "two-ch CASCADE", "prec_par"), ("oracle", "ORACLE", None)]:
        st = vs_naive(cells, key)
        pv = f"{prec(cells, pk):.2f}" if pk else "—"
        w(f"| {B} | {nm} | {st['mae']:.3f} | {st['dmae']:+.3f} | {st['dz']:+.2f} | {st['p']:.1e} | {st['rec']:.2f} | {pv} | {st['n']} |")
w("")

# ---- ordering test (§6.7): parallel vs cascade ----
w("## 3. Does ordering matter? Parallel vs Cascade (§6.7)\n")
w("| B | mean MAE par° | mean MAE cascade° | Δ (par−cas)° | d_z | p |"); w("|---|---|---|---|---|---|")
for B in [1, 2]:
    cells = sel("two", 0.2, B)
    par = persubj(cells, "par"); cas = persubj(cells, "cascade")
    ss = [s for s in SUBJ if s in par and s in cas]
    P = np.array([par[s] for s in ss]); C = np.array([cas[s] for s in ss]); d = P - C
    try: p = wilcoxon(P, C).pvalue
    except Exception: p = np.nan
    dz = np.nanmean(d) / (np.nanstd(d, ddof=1) + 1e-12)
    w(f"| {B} | {np.nanmean(P):.3f} | {np.nanmean(C):.3f} | {np.nanmean(d):+.3f} | {dz:+.2f} | {p:.2f} |")
w("")

# ---- confusion (the two failure modes the framework exists to prevent) ----
w("## 4. Action-vs-channel confusion (parallel; per session at p_c=0.2)\n")
w("| B | image-corrupted DISCARDED (wasteful) | label-corrupted RECAPTURED (futile) | actions taken |")
w("|---|---|---|---|")
for B in [1, 2]:
    cells = sel("two", 0.2, B)
    w(f"| {B} | {prec(cells,'img_discarded'):.2f} | {prec(cells,'lab_recaptured'):.2f} | {prec(cells,'n_acted_par'):.2f} |")
w("\n(These count the two mistakes the two-channel routing is designed to avoid — a wasteful discard of a "
  "correct-label frame, and a futile recapture spent where a free discard would do. Low counts = the routing works.)\n")

w("## 5. What Experiment C shows\n")
w("1. **Recapture ≫ discard.** Replacing flagged frames with clean pool substitutes recovers 33%→98% of the "
  "oracle headroom as p_c goes 0.1→0.4; at p_c=0.4 recapture ≈ oracle (0.97–0.98). Discarding barely helps and "
  "is net-negative at low p_c (recovered <0) — dropping frames shrinks K=9 and the residual gate mis-flags "
  "clean frames when corruption is rare.")
w("2. **Spend precision is the limiter.** Fraction of budget landing on genuinely-corrupted frames rises "
  "0.4→0.9 across p_c; the recapture recovery tracks it. The gate pays off in proportion to how much there is "
  "to find.")
w("3. **Two-channel routing is correct but the p_c=0.2 headroom is negligible.** Naive−oracle is only ~0.03° "
  "there — because image corruption is nearly harmless (Exp B ratio 0.40) and only ~1 label frame is corrupted "
  "— so parallel/cascade recover nothing significant (p>0.39). The confusion counts confirm the routing works "
  "(wasteful discards 0.07–0.16/session, futile recaptures 0.00–0.04): the framework rarely mis-routes; there "
  "is simply little to recover at this rate. Its value would surface at higher contamination or more damaging "
  "image corruption.")
w("4. **Ordering does not matter (§6.7).** Parallel vs cascade differ by <0.02° (d_z ≈ −0.25, p=0.29/0.65, ns). "
  "The gate ordering is not a design lever here.\n")
open(os.path.join(_AG, "logs", "EXP_C_REPORT.md"), "w").write("\n".join(L) + "\n")
print("wrote logs/EXP_C_REPORT.md")
print("\n".join(L))
