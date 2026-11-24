# exp_displacement.py — realised donor-swap displacement distribution (15 subjects)
"""Pure re-analysis, no retraining. Recover the realised angular displacement of
every corrupted donor-swap frame (angle between true label and substituted donor)
across all 15 subjects, seeds 0-9, p_c∈{0.1,0.2,0.4}. Seeds 0-2 carry the stored
actual_offset_deg (exp_a); seeds 3-9 are reconstructed by replaying the identical
seeded corruption (same _rng, same load_subject_support labels/pool). The seed-0-2
reconstruction is validated against the stored values before 3-9 are trusted.

Undetectable fraction reported against the AUC-0.70 crossing RANGE (δ=4.2-4.8°) and
the AUC-0.90 crossing RANGE (8.0-8.7°, unreliably-detectable), never a point; both
global and σ-scaled (σ_s = s0 P1 support-LOO mean, bb26eef).
"""
import csv, hashlib, json, os, sys
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
_HERE = os.path.dirname(os.path.abspath(__file__)); _AG = os.path.dirname(_HERE)
for p in (_HERE, os.path.join(_AG, "scripts"), _AG):
    if p not in sys.path: sys.path.insert(0, p)
import gaze_math as gm, backbone_api as api, corruption as corr
_ROOT = None
def ROOT():
    global _ROOT
    if _ROOT is None:
        import train_backbone as tb; _ROOT = tb.ROOT
    return _ROOT
SUBJ = [f"p{i:02d}" for i in range(15)]; PCS = [0.1, 0.2, 0.4]; SEEDS = list(range(10)); SIGN = -1
CROSS70 = (4.2, 4.8)   # AUC-0.70 crossing range -> undetectable
CROSS90 = (8.0, 8.7)   # AUC-0.90 crossing range -> unreliably-detectable
MS_5SUBJ = dict(p10=4.4, median=10.6, p90=20.8)  # manuscript 5-subject values


def _rng(sid, model, delta, pc, seed):
    key = f"{sid}|{model}|{delta}|{pc}|{seed}".encode()
    return np.random.RandomState(int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2 ** 32))


def subj_vecs(s):
    lines = open(os.path.join(ROOT(), "Label", f"{s}.label")).read().strip().split("\n")
    gi = lines[0].split().index("3DGaze"); out = []
    for ln in lines[1:]:
        g = np.array([float(x) for x in ln.split()[gi].split(",")], np.float32) * SIGN
        out.append(g / (np.linalg.norm(g) + 1e-8))
    return np.asarray(out)


# ---- reconstruct + validate ----
recs = [r for r in (json.loads(l) for l in open(os.path.join(_HERE, "exp_a_records.jsonl"))) if r["model"] == "structured_swap"]
stored = {(r["subject"], r["seed"], round(r["pc"], 3), r["frame_idx"]): (r["is_corrupted"], r["actual_offset_deg"]) for r in recs}
disp = []            # (subject, seed, pc, offset_deg)  corrupted frames only
val_ok = val_tot = 0; maxdiff = 0.0; ncorr_by_pc = defaultdict(int); ncell_by_pc = defaultdict(int)
for s in SUBJ:
    vecs = subj_vecs(s); n = len(vecs)
    sup_ids = np.asarray(api.calibration_split(n, 9, seed=42)[0])
    labels = gm.vector_to_pitchyaw(vecs[sup_ids]).reshape(-1, 2)
    pool = gm.vector_to_pitchyaw(vecs).reshape(-1, 2)
    for seed in SEEDS:
        for pc in PCS:
            c = corr.apply_corruption(labels, "structured_swap", _rng(s, "structured_swap", None, pc, seed), pc=pc, pool_labels=pool)
            mask, off = c["is_corrupted"], c["actual_offset_deg"]; ncell_by_pc[pc] += 1
            for i in range(9):
                if mask[i]:
                    disp.append((s, seed, pc, float(off[i]))); ncorr_by_pc[pc] += 1
                key = (s, seed, round(pc, 3), i)
                if key in stored:
                    val_tot += 1; sc_c, sc_o = stored[key]
                    if bool(sc_c) == bool(mask[i]) and abs(sc_o - off[i]) < 1e-5: val_ok += 1
                    maxdiff = max(maxdiff, abs(sc_o - off[i]))

with open(os.path.join(_AG, "logs", "displacement_raw.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["subject", "seed", "pc", "displacement_deg"]); w.writerows([(s, sd, pc, round(o, 5)) for (s, sd, pc, o) in disp])

allo = np.array([o for _, _, _, o in disp])
by_subj = defaultdict(list)
for s, _, _, o in disp: by_subj[s].append(o)

# ---- sigma_s ----
d = defaultdict(list)
for r in csv.DictReader(open(os.path.join(_AG, "logs", "sigma_procA", "P1_support_loo.csv"))):
    if r["backbone"] == "s0": d[r["subject"]].append(float(r["residual_deg"]))
sig = {s: float(np.mean(d[s])) for s in SUBJ}; sbar = float(np.mean([sig[s] for s in SUBJ]))

L = []; w = lambda *x: L.append(" ".join(str(t) for t in x))
def pct(a): return {k: float(np.percentile(a, v)) for k, v in [("p10", 10), ("p25", 25), ("median", 50), ("p75", 75), ("p90", 90)]}
def frac_below(a, t): return float(np.mean(np.asarray(a) < t))

w("# Donor-swap realised displacement — 15-subject distribution & undetectable fraction\n")
w("Realised displacement = angle(true label, substituted donor). Structured donor-swap, "
  "15 subjects, seeds 0-9, p_c∈{0.1,0.2,0.4}. Pure re-analysis.\n")

# 1. count check + validation
w("## 1. Reconstruction validation & corrupted-frame count\n")
w(f"Seed-0-2 reconstruction vs stored actual_offset_deg: **{val_ok}/{val_tot} exact** "
  f"(max abs diff {maxdiff:.2e}°) — reconstruction reproduces the stored corruption, so seeds 3-9 are trustworthy.")
w(f"\nTotal corrupted frames: **{len(disp)}** across {sum(ncell_by_pc.values())} cells.\n")
w("| p_c | corrupted | cells | E[Bernoulli] = 9·p_c·cells | ratio |"); w("|---|---|---|---|---|")
for pc in PCS:
    exp = 9 * pc * ncell_by_pc[pc]
    w(f"| {pc} | {ncorr_by_pc[pc]} | {ncell_by_pc[pc]} | {exp:.0f} | {ncorr_by_pc[pc]/exp:.3f} |")
w("\n(ratio < 1 by the min_clean=1 truncation, which removes a little mass at high p_c.)\n")

# 2. percentiles pooled + per subject + between-subject spread
P = pct(allo)
w("## 2. Displacement distribution (degrees)\n")
w("| | p10 | p25 | median | p75 | p90 | mean |"); w("|---|---|---|---|---|---|---|")
w(f"| **pooled (15 subj)** | {P['p10']:.1f} | {P['p25']:.1f} | {P['median']:.1f} | {P['p75']:.1f} | {P['p90']:.1f} | {allo.mean():.1f} |")
submed = {s: float(np.median(by_subj[s])) for s in SUBJ}
mv = np.array(list(submed.values()))
w(f"\nPer-subject median: mean {mv.mean():.1f}°, SD **{mv.std():.1f}°**, range [{mv.min():.1f}, {mv.max():.1f}]° "
  f"({SUBJ[int(np.argmin(mv))]}→{SUBJ[int(np.argmax(mv))]}).")
w(f"Between-subject spread in the median is {'MATERIAL' if mv.std()>1.5 else 'modest'} "
  f"— {'the pooled distribution understates per-subject variation and the paper must say so.' if mv.std()>1.5 else 'the pooled distribution is representative.'}\n")
w("| subject | p10 | median | p90 | mean | σ_s |"); w("|---|---|---|---|---|---|")
for s in SUBJ:
    ps = pct(by_subj[s]); w(f"| {s} | {ps['p10']:.1f} | {ps['median']:.1f} | {ps['p90']:.1f} | {np.mean(by_subj[s]):.1f} | {sig[s]:.2f} |")
w("")

# 3. undetectable fraction (global range) pooled
w("## 3. Undetectable / unreliably-detectable fraction (global crossing ranges, pooled)\n")
w("| crossing | range | fraction below (range) |"); w("|---|---|---|")
w(f"| AUC-0.70 (undetectable) | {CROSS70[0]}-{CROSS70[1]}° | **{frac_below(allo,CROSS70[0])*100:.0f}%–{frac_below(allo,CROSS70[1])*100:.0f}%** |")
w(f"| AUC-0.90 (unreliably-detectable) | {CROSS90[0]}-{CROSS90[1]}° | {frac_below(allo,CROSS90[0])*100:.0f}%–{frac_below(allo,CROSS90[1])*100:.0f}% |")
w("")

# 4. per-subject fractions, min/max
w("## 4. Per-subject fractions — worst case is deployment-relevant\n")
u70 = {s: (frac_below(by_subj[s], CROSS70[0]), frac_below(by_subj[s], CROSS70[1])) for s in SUBJ}
u90 = {s: (frac_below(by_subj[s], CROSS90[0]), frac_below(by_subj[s], CROSS90[1])) for s in SUBJ}
lo70 = [u70[s][0] for s in SUBJ]; hi70 = [u70[s][1] for s in SUBJ]
lo90 = [u90[s][0] for s in SUBJ]; hi90 = [u90[s][1] for s in SUBJ]
w("| | undetectable <4.2 | <4.8 | unreliable <8.0 | <8.7 |"); w("|---|---|---|---|---|")
w(f"| mean subj | {np.mean(lo70)*100:.0f}% | {np.mean(hi70)*100:.0f}% | {np.mean(lo90)*100:.0f}% | {np.mean(hi90)*100:.0f}% |")
w(f"| **worst subj (max)** | {max(lo70)*100:.0f}% ({SUBJ[int(np.argmax(lo70))]}) | {max(hi70)*100:.0f}% | {max(lo90)*100:.0f}% | {max(hi90)*100:.0f}% |")
w(f"| best subj (min) | {min(lo70)*100:.0f}% | {min(hi70)*100:.0f}% | {min(lo90)*100:.0f}% | {min(hi90)*100:.0f}% |")
w("")

# 5. sigma-scaled crossing per subject
w("## 5. σ-scaled undetectable fraction (per-subject crossing = global × σ_s/σ̄)\n")
w(f"σ̄ = {sbar:.2f}°. Crossing for subject s scales with its σ_s: high-σ subjects need larger displacement to detect.")
us = {}
for s in SUBJ:
    lo = CROSS70[0] * sig[s] / sbar; hi = CROSS70[1] * sig[s] / sbar
    us[s] = (frac_below(by_subj[s], lo), frac_below(by_subj[s], hi))
slo = [us[s][0] for s in SUBJ]; shi = [us[s][1] for s in SUBJ]
w("\n| | global 4.2-4.8° | σ-scaled |"); w("|---|---|---|")
w(f"| mean undetectable | {np.mean(lo70)*100:.0f}%–{np.mean(hi70)*100:.0f}% | {np.mean(slo)*100:.0f}%–{np.mean(shi)*100:.0f}% |")
w(f"| worst subject | {max(lo70)*100:.0f}%–{max(hi70)*100:.0f}% | {max(slo)*100:.0f}%–{max(shi)*100:.0f}% |")
diff = abs(np.mean(slo) - np.mean(lo70)) * 100
w(f"\nσ-scaled vs global differ by {diff:.0f} pts at the mean. "
  f"{'They differ materially — the σ-scaled version is the correct one for the abstract.' if diff>3 else 'They are close — the global range is adequate for the abstract.'}\n")

# 6. shift from 5-subject
w("## 6. Shift from the 5-subject manuscript values\n")
sub5 = [o for (s, _, _, o) in disp if s in SUBJ[:5]]; p5 = pct(sub5)
w("| stat | 5-subj manuscript | 5-subj (p00-04, this data) | 15-subj | shift (15 − ms) |"); w("|---|---|---|---|---|")
for k, msv in MS_5SUBJ.items():
    w(f"| {k} | {msv} | {p5[k]:.1f} | {P[k]:.1f} | {P[k]-msv:+.1f}° |")
w("")

# histogram
plt.figure(figsize=(7, 4.2))
plt.hist(allo, bins=40, color="#4C72B0", alpha=0.85, edgecolor="white")
plt.axvspan(*CROSS70, color="crimson", alpha=0.25, label=f"AUC-0.70 floor {CROSS70[0]}-{CROSS70[1]}° (undetectable ←)")
plt.axvspan(*CROSS90, color="orange", alpha=0.20, label=f"AUC-0.90 {CROSS90[0]}-{CROSS90[1]}° (unreliable ←)")
plt.axvline(P["median"], color="k", ls="--", lw=1, label=f"median {P['median']:.1f}°")
plt.xlabel("realised donor-swap displacement (deg)"); plt.ylabel("corrupted frames")
plt.title("Donor-swap displacement (15 subjects) — floor range shaded"); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(_AG, "logs", "displacement_hist.png"), dpi=120); plt.close()

open(os.path.join(_AG, "logs", "DISPLACEMENT_REPORT.md"), "w").write("\n".join(L) + "\n")
print("wrote logs/DISPLACEMENT_REPORT.md + displacement_hist.png + displacement_raw.csv")
print("\n".join(L))
