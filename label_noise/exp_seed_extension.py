# exp_seed_extension.py  (Prompt 5 revised)
"""Extend structured donor-swap AND random-offset δ=6 to seeds 3-9 (seeds 0-2
reused from exp_a_records), all 15 subjects, all p_c. Corruption seeds only —
s0 backbone, no new training. Emits LOO / resubstitution(small_loss) / source
per frame. Analysis combines seeds 0-9 for §6.5.2 (LOO vs resub) and §6.5.3
(LOO vs source), per-seed means, and the degenerate-AUC binomial audit.

Records -> label_noise/exp_seed_extension_records.jsonl (seeds 3-9)
Report  -> logs/SEED_EXTENSION_REPORT.md ; raw CSV -> logs/seed_extension_cells.csv
"""
import csv, hashlib, json, os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); _AG = os.path.dirname(_HERE)
for p in (_HERE, os.path.join(_AG, "scripts"), _AG):
    if p not in sys.path: sys.path.insert(0, p)
import torch
import backbone_api as api, corruption as corr, loo_residual as lr, exp_a_scorers as sc
import train_backbone as tb
from scipy.stats import wilcoxon

K = 9
CONDS = [("structured_swap", None), ("random_offset", 6)]
PCS = [0.1, 0.2, 0.4]
NEW_SEEDS = [3, 4, 5, 6, 7, 8, 9]
# Dataset switch (same convention as unigaze_arm.py / train_backbone.py): defaults
# reproduce the MPIIFaceGaze results byte-for-byte; a second dataset sets
# GAZE_SUBJECTS and GAZE_TAG (e.g. "_eyediap"), which suffixes every file read or
# written so the MPIIFaceGaze records and reports are never touched.
TAG = os.environ.get("GAZE_TAG", "")
SUBJ = ([x for x in os.environ["GAZE_SUBJECTS"].split(",") if x] if os.environ.get("GAZE_SUBJECTS")
        else [f"p{i:02d}" for i in range(15)])
REC = os.path.join(_HERE, f"exp_seed_extension_records{TAG}.jsonl")
EXPA = os.path.join(_HERE, f"exp_a_records{TAG}.jsonl")


def _rng(sid, model, delta, pc, seed):
    key = f"{sid}|{model}|{delta}|{pc}|{seed}".encode()
    return np.random.RandomState(int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2 ** 32))


def load_bb(s, device):
    m = tb.build_model(); m.load_state_dict(torch.load(os.path.join(_AG, "models", f"backbone_mpiifacegaze_{s}.pt"), map_location="cpu"))
    return m.to(device).eval()


def roc_auc(scr, pos):
    scr = np.asarray(scr, float); pos = np.asarray(pos, bool); ok = ~np.isnan(scr); scr, pos = scr[ok], pos[ok]
    a, b = int(pos.sum()), int((~pos).sum())
    if a == 0 or b == 0: return np.nan
    u, inv, cnt = np.unique(scr, return_inverse=True, return_counts=True); cs = np.cumsum(cnt); av = (cs - cnt + cs + 1) / 2.0
    return float((av[inv][pos].sum() - a * (a + 1) / 2) / (a * b))


def compute():
    device = api.get_device(); n = 0
    with open(REC, "w") as f:
        for s in SUBJ:
            bb = load_bb(s, device); sup = api.load_subject_support(s, K=K, seed=42)
            imgs, labels, pool = sup["images"], sup["labels"], sup["pool_labels"]
            for seed in NEW_SEEDS:
                for model, delta in CONDS:
                    for pc in PCS:
                        c = corr.apply_corruption(labels, model=model, rng=_rng(s, model, delta, pc, seed),
                                                  pc=pc, delta_deg=delta, pool_labels=pool)
                        obs, mask = c["corrupted_labels"], c["is_corrupted"]
                        small = sc.small_loss_scores(bb, imgs, obs)
                        loo = lr.loo_residuals(bb, imgs, obs)
                        src = lr.source_residuals(bb, imgs, obs)
                        for i in range(K):
                            f.write(json.dumps({"subject": s, "seed": int(seed), "model": model,
                                "delta": (None if delta is None else int(delta)), "pc": float(pc), "frame_idx": i,
                                "is_corrupted": bool(mask[i]),
                                "score_loo": None if np.isnan(loo[i]) else float(loo[i]),
                                "score_small_loss": None if np.isnan(small[i]) else float(small[i]),
                                "score_source": None if np.isnan(src[i]) else float(src[i])}) + "\n"); n += 1
                    f.flush()
            print(f"  [{s}] done ({n} records)", flush=True)
    print(f"wrote {n} records -> {REC}")


def load_all():
    """seeds 0-2 from exp_a (structured + random δ6) + seeds 3-9 from this run."""
    keep = lambda r: (r["model"] == "structured_swap") or (r["model"] == "random_offset" and r["delta"] == 6)
    a = [r for r in (json.loads(l) for l in open(EXPA)) if keep(r) and r["seed"] in (0, 1, 2)]
    b = [json.loads(l) for l in open(REC)]
    return a + b


def per_subject_auc(recs, model, delta, pc, col, seeds=None):
    from collections import defaultdict
    by = defaultdict(lambda: ([], []))
    for r in recs:
        if r["model"] == model and r["delta"] == delta and abs(r["pc"] - pc) < 1e-9 and (seeds is None or r["seed"] in seeds):
            v = r.get(col); by[r["subject"]][0].append(np.nan if v is None else v); by[r["subject"]][1].append(r["is_corrupted"])
    return {s: roc_auc(a, b) for s, (a, b) in by.items()}


def paired(recs, model, delta, pc, A, B, seeds=None):
    aa, bb = per_subject_auc(recs, model, delta, pc, A, seeds), per_subject_auc(recs, model, delta, pc, B, seeds)
    subs = sorted(set(aa) & set(bb)); x = np.array([aa[s] for s in subs]); y = np.array([bb[s] for s in subs])
    m = ~(np.isnan(x) | np.isnan(y)); x, y = x[m], y[m]; d = x - y
    dz = d.mean() / d.std(ddof=1) if len(d) > 1 and d.std(ddof=1) > 0 else np.nan
    p = wilcoxon(x, y).pvalue if len(d) >= 1 and np.any(d != 0) else np.nan
    return len(d), x.mean(), y.mean(), d.mean(), dz, p


def analyze():
    recs = load_all()
    L = ["# Prompt 5 — seed extension (seeds 0-9)\n",
         "Structured donor-swap + random-offset δ=6, s0 backbone. Paired per-subject, seeds 0-9 pooled.\n"]
    for title, A, B, conds in [("§6.5.2  LOO vs resubstitution", "score_loo", "score_small_loss",
                                 [("structured_swap", None), ("random_offset", 6)]),
                               ("§6.5.3  LOO vs source", "score_loo", "score_source",
                                 [("random_offset", 6), ("structured_swap", None)])]:
        L.append(f"## {title}")
        L.append("| cond | pc | n | meanA | meanB | meanΔ | d_z | p |"); L.append("|---|---|---|---|---|---|---|---|")
        for model, delta in conds:
            cond = "struct" if delta is None else f"δ={delta}"
            for pc in PCS:
                n, ma, mb, md, dz, p = paired(recs, model, delta, pc, A, B)
                L.append(f"| {cond} | {pc} | {n} | {ma:.3f} | {mb:.3f} | {md:+.3f} | {dz:+.2f} | {'nan' if p!=p else f'{p:.4f}'} |")
        L.append("")
    # per-seed aggregate means (LOO/resub/source), structured, to expose drift
    L.append("## Per-seed aggregate AUC (structured, subjects pooled) — drift visibility")
    L.append("| pc | seed | LOO | resub | source |"); L.append("|---|---|---|---|---|")
    for pc in PCS:
        for seed in range(10):
            a = {c: np.nanmean([roc_auc([np.nan if r.get(c) is None else r.get(c) for r in g], [r["is_corrupted"] for r in g])
                                for g in [[r for r in recs if r["model"]=="structured_swap" and abs(r["pc"]-pc)<1e-9 and r["seed"]==seed and r["subject"]==s] for s in SUBJ] if g])
                 for c in ("score_loo","score_small_loss","score_source")}
            L.append(f"| {pc} | {seed} | {a['score_loo']:.3f} | {a['score_small_loss']:.3f} | {a['score_source']:.3f} |")
    L.append("")
    # degenerate-AUC audit
    L.append("## Degenerate-AUC audit (seeds 0-9): observed vs binomial expectation")
    L.append("A subject is excluded from a cell only if it draws 0 corrupted (or 0 clean) on ALL seeds present.")
    L.append("| cond | pc | excluded subjects | expected (binomial) |"); L.append("|---|---|---|---|")
    from collections import defaultdict
    for model, delta in [("random_offset", 6), ("structured_swap", None)]:
        cond = "struct" if delta is None else f"δ={delta}"
        for pc in PCS:
            excl = []
            for s in SUBJ:
                cell = [r for r in recs if r["subject"]==s and r["model"]==model and r["delta"]==delta and abs(r["pc"]-pc)<1e-9]
                ncorr = sum(r["is_corrupted"] for r in cell); ntot = len(cell)
                if ncorr == 0 or ncorr == ntot: excl.append(s)
            nseeds = len(set(r["seed"] for r in recs if r["model"]==model and r["delta"]==delta and abs(r["pc"]-pc)<1e-9))
            p0 = (1-pc)**K
            exp_excl = 15 * (p0 ** nseeds)  # zero-corrupted on all seeds
            L.append(f"| {cond} | {pc} | {excl or 'none'} | {exp_excl:.3f} (P0={p0:.3f}/seed, {nseeds} seeds) |")
    out = os.path.join(_AG, "logs", f"SEED_EXTENSION_REPORT{TAG}.md"); open(out, "w").write("\n".join(L) + "\n")
    # raw CSV
    with open(os.path.join(_AG, "logs", f"seed_extension_cells{TAG}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["subject","seed","model","delta","pc","frame_idx","is_corrupted","score_loo","score_small_loss","score_source"])
        for r in recs: w.writerow([r["subject"],r["seed"],r["model"],r["delta"],r["pc"],r["frame_idx"],r["is_corrupted"],r.get("score_loo"),r.get("score_small_loss"),r.get("score_source")])
    print("wrote", out); print("\n".join(L))


if __name__ == "__main__":
    if "--analyze-only" not in sys.argv:
        compute()
    analyze()
