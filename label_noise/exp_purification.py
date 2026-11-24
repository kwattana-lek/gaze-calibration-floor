# exp_purification.py  (Task 2)
"""Does adaptation recover under contamination? Compare purified-LOO (drop the
top-scoring frame(s), re-adapt on the remainder, re-score — one round) against
plain LOO and the no-adaptation source residual, at δ=6 and structured, pc in
{0.1,0.2,0.4}. Two drop rules: drop-1 (loo_residuals_purified) and
drop-round(pc*K). Paired per-subject Wilcoxon + Cohen's d_z (fold=subject).

Reconstructs the exact Experiment-A corruption via the same SHA-256 seed.
Records -> label_noise/exp_purification_records.jsonl ; report -> logs/PURIFICATION_REPORT.md
"""
import hashlib, json, os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); _AG = os.path.dirname(_HERE)
for p in (_HERE, os.path.join(_AG, "scripts"), _AG):
    if p not in sys.path: sys.path.insert(0, p)
import torch
import gaze_math as gm, backbone_api as api, corruption as corr, loo_residual as lr
import train_backbone as tb
try:
    from scipy.stats import wilcoxon; HAVE = True
except Exception:
    HAVE = False

K = 9
CONDS = [("random_offset", 6), ("structured_swap", None)]
PCS = [0.1, 0.2, 0.4]
SUBJ = [f"p{i:02d}" for i in range(15)]
REC = os.path.join(_HERE, "exp_purification_records.jsonl")


def load_bb(s, device):
    m = tb.build_model(); m.load_state_dict(torch.load(os.path.join(_AG, "models", f"backbone_mpiifacegaze_{s}.pt"), map_location="cpu"))
    return m.to(device).eval()


def _rng(subject, model, delta, pc, seed):
    key = f"{subject}|{model}|{delta}|{pc}|{seed}".encode()
    return np.random.RandomState(int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2 ** 32))


def roc_auc(sc, pos):
    sc = np.asarray(sc, float); pos = np.asarray(pos, bool); ok = ~np.isnan(sc); sc, pos = sc[ok], pos[ok]
    a, b = int(pos.sum()), int((~pos).sum())
    if a == 0 or b == 0: return np.nan
    u, inv, cnt = np.unique(sc, return_inverse=True, return_counts=True); cs = np.cumsum(cnt); av = (cs - cnt + cs + 1) / 2.0
    return float((av[inv][pos].sum() - a * (a + 1) / 2) / (a * b))


def compute():
    device = api.get_device(); n = 0
    with open(REC, "w") as f:
        for s in SUBJ:
            bb = load_bb(s, device); sup = api.load_subject_support(s, K=K, seed=42)
            imgs, labels, pool = sup["images"], sup["labels"], sup["pool_labels"]
            for seed in (0, 1, 2):
                for model, delta in CONDS:
                    for pc in PCS:
                        c = corr.apply_corruption(labels, model=model, rng=_rng(s, model, delta, pc, seed),
                                                  pc=pc, delta_deg=delta, pool_labels=pool)
                        obs, mask = c["corrupted_labels"], c["is_corrupted"]
                        first = lr.loo_residuals(bb, imgs, obs)                 # plain LOO (9 adapts)
                        src = lr.source_residuals(bb, imgs, obs)
                        order = np.argsort(-np.where(np.isnan(first), -1e9, first))
                        pur1 = lr.loo_residuals(bb, imgs, obs, exclude={int(order[0])})
                        ndrop = int(round(pc * K))
                        purN = pur1 if ndrop <= 1 else lr.loo_residuals(bb, imgs, obs, exclude=set(int(x) for x in order[:ndrop]))
                        for i in range(K):
                            f.write(json.dumps({"subject": s, "seed": int(seed), "model": model,
                                "delta": (None if delta is None else int(delta)), "pc": float(pc), "frame_idx": i,
                                "is_corrupted": bool(mask[i]), "n_dropped": ndrop,
                                "score_loo": None if np.isnan(first[i]) else float(first[i]),
                                "score_loo_pur1": None if np.isnan(pur1[i]) else float(pur1[i]),
                                "score_loo_purN": None if np.isnan(purN[i]) else float(purN[i]),
                                "score_source": None if np.isnan(src[i]) else float(src[i])}) + "\n"); n += 1
                    f.flush()
            print(f"  [{s}] done ({n} records)", flush=True)
    print(f"wrote {n} records -> {REC}")


def per_subject_auc(recs, model, delta, pc, col):
    from collections import defaultdict
    by = defaultdict(lambda: ([], []))
    for r in recs:
        if r["model"] == model and r["delta"] == delta and abs(r["pc"] - pc) < 1e-9:
            v = r.get(col); by[r["subject"]][0].append(np.nan if v is None else v); by[r["subject"]][1].append(r["is_corrupted"])
    return {s: roc_auc(a, b) for s, (a, b) in by.items()}


def paired(recs, model, delta, pc, A, B):
    aa, bb = per_subject_auc(recs, model, delta, pc, A), per_subject_auc(recs, model, delta, pc, B)
    subs = sorted(set(aa) & set(bb)); x = np.array([aa[s] for s in subs]); y = np.array([bb[s] for s in subs])
    m = ~(np.isnan(x) | np.isnan(y)); x, y = x[m], y[m]; d = x - y
    dz = d.mean() / d.std(ddof=1) if len(d) > 1 and d.std(ddof=1) > 0 else np.nan
    p = np.nan
    if HAVE and len(d) >= 1 and np.any(d != 0):
        try: p = wilcoxon(x, y, zero_method="wilcox").pvalue
        except Exception: pass
    return len(d), x.mean(), y.mean(), d.mean(), dz, p


def analyze():
    recs = [json.loads(l) for l in open(REC)]
    L = ["# Task 2 — Purification under contamination\n",
         f"Paired per-subject (fold=subject, seeds pooled). scipy={HAVE}. n_dropped: pc0.1->1, 0.2->2, 0.4->4.\n"]
    comps = [("loo_pur1_vs_loo", "score_loo_pur1", "score_loo"),
             ("loo_purN_vs_loo", "score_loo_purN", "score_loo"),
             ("loo_pur1_vs_source", "score_loo_pur1", "score_source"),
             ("loo_purN_vs_source", "score_loo_purN", "score_source"),
             ("loo_vs_source", "score_loo", "score_source")]
    for name, A, B in comps:
        L.append(f"## {name}")
        L.append("| cond | pc | n | meanA | meanB | meanΔ | d_z | wilcoxon_p |"); L.append("|---|---|---|---|---|---|---|---|")
        for cond, (model, delta) in [("δ=6", ("random_offset", 6)), ("struct", ("structured_swap", None))]:
            for pc in PCS:
                n, ma, mb, md, dz, p = paired(recs, model, delta, pc, A, B)
                L.append(f"| {cond} | {pc} | {n} | {ma:.3f} | {mb:.3f} | {md:+.3f} | {dz:+.2f} | {'nan' if p!=p else f'{p:.4f}'} |")
        L.append("")
    out = os.path.join(_AG, "logs", "PURIFICATION_REPORT.md"); open(out, "w").write("\n".join(L) + "\n")
    print("wrote", out)
    print("\n".join(L))


if __name__ == "__main__":
    if "--analyze-only" not in sys.argv:
        compute()
    analyze()
