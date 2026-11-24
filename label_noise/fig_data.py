# fig_data.py — verified data loaders for the manuscript figures (no new computation)
"""Every figure's numbers come from committed CSVs / records, reproducing the exact
methods that produced the manuscript values. Fig 1 reproduces analyze_sigma.py's
collapse (σ = POOLED both-backbone P1 mean → matches cited CV 0.167/0.073); Fig 4
reconstructs the operating-point sweep from the same structured residual records the
operating-point analysis used (the per-threshold sweep was never persisted to CSV)."""
import csv, json, os
from collections import defaultdict
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); _AG = os.path.dirname(_HERE)
SUBJ = [f"p{i:02d}" for i in range(15)]
DELTAS_DET = [3, 4, 5, 6, 8, 10]          # exp_a random_offset deltas
PCS = [0.1, 0.2, 0.4]
FLOOR70 = (4.2, 4.8)                        # AUC-0.70 detection floor range
FLOOR90 = (8.0, 8.7)                        # AUC-0.90 crossing range
TAU_SHIP = 3.42


def _roc_auc(scr, pos):
    scr = np.asarray(scr, float); pos = np.asarray(pos, bool); ok = ~np.isnan(scr); scr, pos = scr[ok], pos[ok]
    a, b = int(pos.sum()), int((~pos).sum())
    if a == 0 or b == 0: return np.nan
    u, inv, cnt = np.unique(scr, return_inverse=True, return_counts=True); cs = np.cumsum(cnt); av = (cs - cnt + cs + 1) / 2.0
    return float((av[inv][pos].sum() - a * (a + 1) / 2) / (a * b))


def _crossing(ds, a, level=0.7):
    d = np.array(ds, float); a = np.array(a, float); o = np.argsort(d); d, a = d[o], a[o]
    if np.all(a < level): return np.nan
    if a[0] >= level: return float(d[0])
    for i in range(1, len(d)):
        if a[i - 1] < level <= a[i]:
            t = (level - a[i - 1]) / (a[i] - a[i - 1]); return float(d[i - 1] + t * (d[i] - d[i - 1]))
    return np.nan


# ---------- FIG 1: collapse (reproduces analyze_sigma.py exactly, POOLED-backbone σ) ----------
def fig1_collapse():
    P1 = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(os.path.join(_AG, "logs", "sigma_procA", "P1_support_loo.csv"))):
        P1[r["subject"]][r["backbone"]].append(float(r["residual_deg"]))
    sig = {s: float(np.concatenate([np.array(P1[s][b]) for b in sorted(P1[s])]).mean()) for s in SUBJ}  # pooled both backbones
    sig_s0 = {s: float(np.mean(P1[s]["s0"])) for s in SUBJ}   # s0-only, for the reported discrepancy
    expa = [json.loads(l) for l in open(os.path.join(_HERE, "exp_a_records.jsonl"))]
    subj_auc = {}
    for s in SUBJ:
        subj_auc[s] = [_roc_auc([r.get("score_loo") for r in expa if r["subject"] == s and r["model"] == "random_offset" and r["delta"] == d],
                                [r["is_corrupted"] for r in expa if r["subject"] == s and r["model"] == "random_offset" and r["delta"] == d])
                       for d in DELTAS_DET]
    def terciles(sg):
        order = sorted(SUBJ, key=lambda s: sg[s]); return [order[:5], order[5:10], order[10:]]
    def build(sg):
        ter = terciles(sg); curves, cross, sig_t = [], [], []
        for g in ter:
            au = [np.nanmean([subj_auc[s][i] for s in g]) for i in range(len(DELTAS_DET))]
            curves.append(au); cross.append(_crossing(DELTAS_DET, au)); sig_t.append(np.mean([sg[s] for s in g]))
        cross = np.array(cross); sig_t = np.array(sig_t)
        cv_raw = np.nanstd(cross) / np.nanmean(cross)
        cv_norm = np.nanstd(cross / sig_t) / np.nanmean(cross / sig_t)
        return dict(curves=curves, cross=cross, sig_t=sig_t, cv_raw=cv_raw, cv_norm=cv_norm)
    return dict(deltas=DELTAS_DET, pooled=build(sig), s0=build(sig_s0), floor70=FLOOR70)


# ---------- FIG 2: donor-swap displacement ----------
def fig2_displacement():
    d = np.array([float(r["displacement_deg"]) for r in csv.DictReader(open(os.path.join(_AG, "logs", "displacement_raw.csv")))])
    pct = {p: float(np.percentile(d, p)) for p in (10, 25, 50, 75, 90)}
    frac = {t: float(np.mean(d < t)) for t in (4.2, 4.8, 8.0, 8.7)}
    return dict(disp=d, pct=pct, frac=frac, floor70=FLOOR70, floor90=FLOOR90)


# ---------- FIG 3: damage ratio (from committed severity summary) ----------
def fig3_ratio():
    rows = {(r["channel"], r["severity"]): r for r in csv.DictReader(open(os.path.join(_AG, "logs", "exp_b_severity_summary.csv")))}
    deltas = [4, 5, 6, 10]
    ratio = [float(rows[("a_rot", str(d))]["agg_ratio"]) for d in deltas]
    lost = [float(rows[("a_rot", str(d))]["lost_deg"]) for d in deltas]
    mis = [float(rows[("a_rot", str(d))]["misleading_deg"]) for d in deltas]
    img_ratio = float(rows[("c_image", "ALL")]["agg_ratio"])
    # interpolated crossing of ratio=1 between the two bracketing samples
    lo = hi = None
    for a, b in zip(deltas, deltas[1:]):
        ra, rb = ratio[deltas.index(a)], ratio[deltas.index(b)]
        if ra < 1 <= rb: lo, hi, rlo, rhi = a, b, ra, rb; break
    dstar = lo + (1 - rlo) / (rhi - rlo) * (hi - lo) if lo is not None else None
    return dict(deltas=deltas, ratio=ratio, lost=lost, mis=mis, img_ratio=img_ratio,
                dstar=dstar, floor70=FLOOR70)


# ---------- FIG 4: operating-point trade-off (reconstructed sweep) ----------
def fig4_tradeoff():
    recs = [r for r in (json.loads(l) for l in open(os.path.join(_HERE, "exp_a_records.jsonl")))
            if r["model"] == "structured_swap" and r["seed"] in (0, 1, 2)]
    recs += [r for r in (json.loads(l) for l in open(os.path.join(_HERE, "exp_seed_extension_records.jsonl"))) if r["model"] == "structured_swap"]
    out = {}
    for pc in PCS:
        rows = [(r["is_corrupted"], r.get("score_small_loss")) for r in recs if abs(r["pc"] - pc) < 1e-9]
        vals = [v for _, v in rows if v is not None]
        taus = np.linspace(min(vals), max(vals), 200)
        wasted, missed = [], []
        for t in taus:
            tp = fp = tn = fn = 0
            for c, v in rows:
                if v is None: continue
                f = v >= t
                tp += c & f; fn += c & (not f); fp += (not c) & f; tn += (not c) & (not f)
            rec = tp / (tp + fn) if tp + fn else np.nan; spec = tn / (tn + fp) if tn + fp else np.nan
            wasted.append(9 * (1 - pc) * (1 - spec)); missed.append(9 * pc * (1 - rec))
        # star at tau=3.42
        tp = fp = tn = fn = 0
        for c, v in rows:
            if v is None: continue
            f = v >= TAU_SHIP
            tp += c & f; fn += c & (not f); fp += (not c) & f; tn += (not c) & (not f)
        rec = tp / (tp + fn); spec = tn / (tn + fp)
        out[pc] = dict(wasted=np.array(wasted), missed=np.array(missed),
                       star=(9 * (1 - pc) * (1 - spec), 9 * pc * (1 - rec)))
    return dict(pcs=PCS, curves=out, tau=TAU_SHIP)
