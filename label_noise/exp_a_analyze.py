# exp_a_analyze.py
"""Experiment A analysis -> logs/EXP_A_REPORT.md.
Per scorer x cell: ROC-AUC [95% bootstrap CI over (subject,seed) folds], PR-AUC,
TPR@FPR=0.1, n, corrupted. DivideMix failure rate as its own column. Highlights
δ∈{4,6} and structured (§1 protocol); plus the LOO/source 0.7 & 0.9 crossings."""
import json, os, sys
from collections import defaultdict
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)); _AG = os.path.dirname(_HERE)
RNG = np.random.RandomState(0); B = 2000
SCORERS = ["small_loss", "dividemix", "coteaching", "suge", "loo", "source"]


def roc_auc(sc, pos):
    sc = np.asarray(sc, float); pos = np.asarray(pos, bool)
    ok = ~np.isnan(sc); sc, pos = sc[ok], pos[ok]
    np_, nn = int(pos.sum()), int((~pos).sum())
    if np_ == 0 or nn == 0: return float("nan")
    u, inv, cnt = np.unique(sc, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); avg = (csum - cnt + csum + 1) / 2.0
    return float((avg[inv][pos].sum() - np_ * (np_ + 1) / 2) / (np_ * nn))


def pr_auc(sc, pos):
    sc = np.asarray(sc, float); pos = np.asarray(pos, bool)
    ok = ~np.isnan(sc); sc, pos = sc[ok], pos[ok]
    if pos.sum() == 0: return float("nan")
    order = np.argsort(-sc); pos = pos[order]
    tp = np.cumsum(pos); fp = np.cumsum(~pos)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / pos.sum()
    rec = np.concatenate([[0], rec]); prec = np.concatenate([[1], prec])
    return float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))


def tpr_at_fpr(sc, pos, fpr=0.1):
    sc = np.asarray(sc, float); pos = np.asarray(pos, bool)
    ok = ~np.isnan(sc); sc, pos = sc[ok], pos[ok]
    neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0: return float("nan")
    thr = np.quantile(sc[neg], 1 - fpr)          # score threshold giving ~fpr on negatives
    return float((sc[pos] >= thr).mean())


# Resampling unit. Seeds and frames are nested within subjects, so the SUBJECT is the
# independent unit and the default; GAZE_BOOT_UNIT=fold restores the earlier
# (subject, seed)-fold bootstrap for comparison (reviewer, major comment 6).
BOOT_UNIT = os.environ.get("GAZE_BOOT_UNIT", "subject")


def boot_auc_ci(frames_by_fold):
    if BOOT_UNIT == "subject":
        by_s = defaultdict(list)
        for (s, _seed), v in frames_by_fold.items():
            by_s[s].append(v)
        folds = [(np.concatenate([a for a, _ in v]), np.concatenate([b for _, b in v])) for v in by_s.values()]
    else:
        folds = list(frames_by_fold.values())
    aucs = []
    for _ in range(B):
        pick = [folds[i] for i in RNG.randint(0, len(folds), len(folds))]
        sc = np.concatenate([f[0] for f in pick]); pos = np.concatenate([f[1] for f in pick])
        a = roc_auc(sc, pos)
        if not np.isnan(a): aucs.append(a)
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))) if aucs else (float("nan"),) * 2


def crossing(ds, aucs, level):
    d = np.array(ds, float); a = np.array(aucs, float); o = np.argsort(d); d, a = d[o], a[o]
    if np.all(a < level): return f">{d.max():.0f}"
    if a[0] >= level: return f"<={d.min():.0f}"
    for i in range(1, len(d)):
        if a[i-1] < level <= a[i]:
            t = (level-a[i-1])/(a[i]-a[i-1]); return round(float(d[i-1]+t*(d[i]-d[i-1])), 2)
    return "n/a"


recs = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "exp_a_records.jsonl"))]
pcs = sorted({r["pc"] for r in recs})
deltas = sorted({r["delta"] for r in recs if r["model"] == "random_offset" and r["delta"] is not None})
cell_keys = [("random_offset", d) for d in deltas] + [("structured_swap", None)]

L = []; w = lambda *x: L.append(" ".join(str(t) for t in x))
w("# Experiment A — baselines that should fail at K=9\n")
w(f"{len(recs)} frame records. Scorers: {SCORERS}. deltas={deltas} pcs={pcs}. "
  f"Bootstrap B={B} over {'subjects' if BOOT_UNIT == 'subject' else '(subject,seed) folds'}. Higher score = more likely corrupted.\n")


def cell_metrics(scorer, model, delta, pc):
    by_fold = defaultdict(lambda: ([], []))
    sc_all, y_all = [], []
    for r in recs:
        if r["model"] == model and r["delta"] == delta and abs(r["pc"] - pc) < 1e-9:
            v = r.get(f"score_{scorer}")
            fk = (r["subject"], r["seed"])
            by_fold[fk][0].append(np.nan if v is None else v)
            by_fold[fk][1].append(r["is_corrupted"])
            sc_all.append(np.nan if v is None else v); y_all.append(r["is_corrupted"])
    if not sc_all: return None
    by_fold = {k: (np.array(a, float), np.array(b, bool)) for k, (a, b) in by_fold.items()}
    sc_all, y_all = np.array(sc_all, float), np.array(y_all, bool)
    if np.all(np.isnan(sc_all)): return None
    ci = boot_auc_ci(by_fold)
    return dict(auc=roc_auc(sc_all, y_all), ci=ci, pr=pr_auc(sc_all, y_all),
                tpr=tpr_at_fpr(sc_all, y_all), n=len(y_all), npos=int(y_all.sum()))


# --- headline: all scorers at δ=4, δ=6, structured ---
w("## Headline — all scorers at δ=4, δ=6, structured (ROC-AUC [95% CI])\n")
for pc in pcs:
    w(f"### pc = {pc}\n")
    w("| scorer | δ=4 | δ=6 | structured |"); w("|---|---|---|---|")
    for scorer in SCORERS:
        cellvals = []
        for model, delta in [("random_offset", 4), ("random_offset", 6), ("structured_swap", None)]:
            m = cell_metrics(scorer, model, delta, pc)
            cellvals.append("n/a" if not m or np.isnan(m["auc"]) else f"{m['auc']:.3f} [{m['ci'][0]:.2f},{m['ci'][1]:.2f}]")
        w(f"| {scorer} | " + " | ".join(cellvals) + " |")
    w("")

# --- DivideMix failure rate ---
w("## DivideMix GMM failure rate (a result, not a footnote)\n")
w("| pc | δ | frames | fit failed | reasons |")
w("|---|---|---|---|---|")
for pc in pcs:
    for model, delta in cell_keys:
        rr = [r for r in recs if r["model"] == model and r["delta"] == delta and abs(r["pc"] - pc) < 1e-9]
        if not rr: continue
        nfail = sum(1 for r in rr if r["dividemix_failed"]);
        from collections import Counter
        reasons = Counter(r["dividemix_reason"] for r in rr if r["dividemix_failed"])
        dl = "struct" if delta is None else delta
        w(f"| {pc} | {dl} | {len(rr)} | {100*nfail/len(rr):.0f}% | {dict(reasons) or '-'} |")
w("")

# --- full per-scorer tables ---
for scorer in SCORERS:
    w(f"## {scorer} — full table\n")
    w("| pc | δ | AUC [95% CI] | PR-AUC | TPR@FPR0.1 | n | corrupt |"); w("|---|---|---|---|---|---|---|")
    for pc in pcs:
        for model, delta in cell_keys:
            m = cell_metrics(scorer, model, delta, pc)
            if not m: continue
            dl = "struct" if delta is None else delta
            w(f"| {pc} | {dl} | {m['auc']:.3f} [{m['ci'][0]:.3f},{m['ci'][1]:.3f}] | {m['pr']:.3f} | "
              f"{m['tpr']:.3f} | {m['n']} | {m['npos']} |")
    w("")

# --- recomputed crossings (LOO, source) over δ 3-10, 15 subjects ---
w("## Recomputed LOO & source crossings (15 subjects, δ 3-10)\n")
w("| pc | LOO δ@0.7 | LOO δ@0.9 | source δ@0.7 | source δ@0.9 |"); w("|---|---|---|---|---|")
for pc in pcs:
    la = [cell_metrics("loo", "random_offset", d, pc)["auc"] for d in deltas]
    sa = [cell_metrics("source", "random_offset", d, pc)["auc"] for d in deltas]
    w(f"| {pc} | {crossing(deltas, la, 0.7)} | {crossing(deltas, la, 0.9)} | "
      f"{crossing(deltas, sa, 0.7)} | {crossing(deltas, sa, 0.9)} |")

out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_AG, "logs", "EXP_A_REPORT.md")
open(out, "w").write("\n".join(L) + "\n")
print("wrote", out)
