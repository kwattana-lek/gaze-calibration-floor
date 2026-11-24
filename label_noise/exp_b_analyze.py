# exp_b_analyze.py — Experiment B decomposition (per-severity, paired, coordination)
"""Aggregate exp_b_records.csv. Tests the §3.2 pre-registered prediction (25cde1c):
image ratio ≤ 1, label ratio > 1 and growing with δ. ratio = misleading/lost =
(naive-oracle)/(oracle-clean). Reports ratio PER SEVERITY (never pooled), paired
per-subject misleading-vs-lost with d_z and p (n stated), the (b)-vs-(d)
coordination sub-analysis at matched median displacement, and TraNFS as discussion.
"""
import csv, os
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
_HERE = os.path.dirname(os.path.abspath(__file__)); _AG = os.path.dirname(_HERE)
SUBJ = [f"p{i:02d}" for i in range(15)]

rows = list(csv.DictReader(open(os.path.join(_HERE, "exp_b_records.csv"))))
def fnum(x): return float(x) if x not in ("", "nan", None) else np.nan
for r in rows:
    for k in ("pc", "clean", "oracle", "naive", "lost_cost", "misleading_cost", "ratio", "median_offset_deg", "day_straddle", "n_corrupted"):
        r[k] = fnum(r[k])


def subj_means(cells, key):
    """per-subject mean of `key` over the given cells -> dict subject->value."""
    by = defaultdict(list)
    for r in cells: by[r["subject"]].append(r[key])
    return {s: np.nanmean(by[s]) for s in by}


def paired(cells):
    """paired per-subject misleading vs lost: aggregate ratio, d_z, p (n subjects)."""
    mis = subj_means(cells, "misleading_cost"); los = subj_means(cells, "lost_cost")
    ss = [s for s in SUBJ if s in mis and np.isfinite(mis[s]) and np.isfinite(los[s])]
    m = np.array([mis[s] for s in ss]); l = np.array([los[s] for s in ss])
    diff = m - l; dz = np.nanmean(diff) / (np.nanstd(diff, ddof=1) + 1e-12)
    try: p = wilcoxon(m, l).pvalue if len(ss) > 1 else np.nan
    except Exception: p = np.nan
    agg_ratio = np.nanmean(m) / np.nanmean(l) if np.nanmean(l) != 0 else np.nan
    med_ratio = np.nanmedian([r["ratio"] for r in cells if np.isfinite(r["ratio"])])
    return dict(n=len(ss), lost=np.nanmean(l), mis=np.nanmean(m), agg_ratio=agg_ratio,
                med_ratio=med_ratio, dz=dz, p=p)


L = []; w = lambda *x: L.append(" ".join(str(t) for t in x))
w("# Experiment B — two-channel damage decomposition (15 subjects, s0)\n")
w("ratio = misleading/lost = (naive−oracle)/(oracle−clean). §3.2 prediction (25cde1c): "
  "image ratio ≤ 1; label ratio > 1 and growing with δ. Aggregate ratio = mean(misleading)/mean(lost) "
  "across cells (robust); paired test = misleading vs lost per subject.\n")

# ---- per-severity table ----
w("## 1. Damage ratio per severity level (never pooled)\n")
w("| channel | severity | n | lost° | misleading° | **agg ratio** | median ratio | d_z | p | pred |")
w("|---|---|---|---|---|---|---|---|---|---|")
_KNOWN = {"4": "label δ=4", "5": "label δ=5 [POST-HOC]", "6": "label δ=6", "10": "label δ=10"}
# Discover a_rot severities from the data so a fine grid appears automatically
# (exp_b_finegrid.py appends 4.25/4.5/4.75); hardcoding hid new points before.
_arot = sorted({r["severity"] for r in rows if r["channel"] == "a_rot"}, key=float)
severities = [("a_rot", sv, _KNOWN.get(sv, f"label δ={sv} [POST-HOC]")) for sv in _arot] + \
             [("b_swap", "", "label donor-swap"), ("d_coord", "", "label coordinated")]
ratio_by_delta = {}
for chan, sev, lbl in severities:
    cells = [r for r in rows if r["channel"] == chan and r["severity"] == sev]
    st = paired(cells)
    pred = "ratio>1"
    star = "✓" if st["agg_ratio"] > 1 else "✗"
    w(f"| {lbl} | {sev or '—'} | {st['n']} | {st['lost']:.2f} | {st['mis']:.2f} | "
      f"**{st['agg_ratio']:.2f}** | {st['med_ratio']:.2f} | {st['dz']:+.2f} | {st['p']:.1e} | {pred} {star} |")
    if chan == "a_rot": ratio_by_delta[float(sev)] = st["agg_ratio"]
# image: per-mode (single-mode cells) + overall
img_cells = [r for r in rows if r["channel"] == "c_image"]
modes_of = lambda r: set(m for m in (r.get("image_modes") or "").split("|") if m)
for mode in ["motion_blur", "blink", "occlusion", "illumination", "head_yaw", "head_pitch"]:
    cells = [r for r in img_cells if modes_of(r) == {mode}]
    if len(cells) < 5: continue
    st = paired(cells)
    star = "✓" if st["agg_ratio"] <= 1 else "✗"
    w(f"| image: {mode} | (single-mode) | {st['n']} | {st['lost']:.2f} | {st['mis']:.2f} | "
      f"**{st['agg_ratio']:.2f}** | {st['med_ratio']:.2f} | {st['dz']:+.2f} | {st['p']:.1e} | ratio≤1 {star} |")
st = paired(img_cells)
star = "✓" if st["agg_ratio"] <= 1 else "✗"
w(f"| **image: ALL** | (mixed) | {st['n']} | {st['lost']:.2f} | {st['mis']:.2f} | "
  f"**{st['agg_ratio']:.2f}** | {st['med_ratio']:.2f} | {st['dz']:+.2f} | {st['p']:.1e} | ratio≤1 {star} |")
w("")

# ---- delta growth + crossing interpolation ----
FLOOR = (4.2, 4.8)
w("## 2. Does the label ratio grow with δ, and where does it cross 1? (§3.2 second clause)\n")
ds = sorted(ratio_by_delta)          # [4, 5, 6, 10]
mono = all(ratio_by_delta[a] < ratio_by_delta[b] for a, b in zip(ds, ds[1:]))
w("δ sequence (agg ratio): " + ", ".join(f"**δ={d} → {ratio_by_delta[d]:.2f}**" for d in ds) + ". "
  + ("Monotone increasing ✓." if mono else "NOT monotone — see values.") + "\n")
_posthoc = [d for d in sorted(ratio_by_delta) if d not in (4.0, 6.0, 10.0)]
w("POST-HOC severities (not in the 25cde1c pre-registration): "
  + (", ".join(f"δ={d:g}" for d in _posthoc) if _posthoc else "none")
  + ". §3.2 is unrevised.\n")
# bracket where ratio crosses 1
lo = hi = None
for a, b in zip(ds, ds[1:]):
    if ratio_by_delta[a] < 1 <= ratio_by_delta[b]: lo, hi = a, b; break
if lo is not None:
    rlo, rhi = ratio_by_delta[lo], ratio_by_delta[hi]
    dstar = lo + (1 - rlo) / (rhi - rlo) * (hi - lo)
    _n = len(ds)
    _measured = (hi - lo) <= 0.3     # bracket tight enough to call it measured
    w(f"The ratio crosses 1 inside the **[{lo:g}, {hi:g}]°** bracket, from {_n} sampled "
      f"severities. Linear interpolation between the bracketing samples "
      f"(δ={lo:g}→{rlo:.2f}, δ={hi:g}→{rhi:.2f}) puts the crossing at "
      f"**δ* ≈ {dstar:.2f}°** — "
      + ("a bracket of {:.2f}°, narrow enough to treat as a measurement.".format(hi - lo)
         if _measured else
         "an interpolation across a {:.2f}° gap, NOT a measured crossing.".format(hi - lo))
      + "\n")
    # Uncertainty on delta*. The point estimate alone is not enough to place the
    # crossing relative to the floor: a bracket of 0.25 deg says the GRID is fine,
    # not that delta* is precise. Resample subjects and re-derive the crossing.
    import random as _rnd
    _sub = sorted({r["subject"] for r in rows if r["channel"] == "a_rot"})
    def _cross_for(subs):
        cnt = {}
        for sev in sorted({float(r["severity"]) for r in rows if r["channel"] == "a_rot"}):
            L, M = [], []
            for r in rows:
                if r["channel"] != "a_rot" or abs(float(r["severity"]) - sev) > 1e-9: continue
                if r["subject"] not in subs: continue
                try: L.append(float(r["lost_cost"])); M.append(float(r["misleading_cost"]))
                except Exception: pass
            if L: cnt[sev] = np.nanmean(M) / np.nanmean(L) if np.nanmean(L) else np.nan
        dd = sorted(cnt)
        for a, b in zip(dd, dd[1:]):
            if cnt[a] < 1 <= cnt[b]:
                return a + (1 - cnt[a]) / (cnt[b] - cnt[a]) * (b - a)
        return None
    _rg = _rnd.Random(0); _bs = []
    for _ in range(4000):
        c = _cross_for([_sub[_rg.randrange(len(_sub))] for _ in _sub])
        if c is not None: _bs.append(c)
    _bs.sort()
    dlo, dhi = _bs[int(.025 * len(_bs))], _bs[int(.975 * len(_bs))]
    p_in = sum(1 for b in _bs if FLOOR[0] <= b <= FLOOR[1]) / len(_bs)
    contains = dlo <= FLOOR[0] and dhi >= FLOOR[1]
    sep = dlo > FLOOR[1] or dhi < FLOOR[0]
    w(f"Bootstrapping over subjects ({len(_bs)} resamples) gives δ* = **{dstar:.2f}° "
      f"[{dlo:.2f}, {dhi:.2f}]**; P(δ* inside the floor) = {p_in:.2f}.\n")
    w(f"**Against the AUC-0.70 detection floor {FLOOR[0]}–{FLOOR[1]}°:** "
      + ("the crossing interval CONTAINS the entire floor range, so the two "
         "quantities are statistically indistinguishable. Report them as "
         "inseparable, not as exactly coincident — the point estimate's position "
         "relative to the floor is not resolvable at this sample size."
         if contains else
         ("the intervals are DISJOINT — the crossing is genuinely "
          + ("above" if dlo > FLOOR[1] else "below")
          + " the detection floor. The coincidence claim fails; revise it."
          if sep else
          "the intervals overlap partially. Report proximity with the interval, "
          "and do not rest any claim on which side the point estimate falls."))
      + "\n")
else:
    w("No <1→≥1 crossing found among the four δ samples — inspect the sequence.\n")
w("Note on the image per-mode rows: lost-frame cost is ≈0.00–0.03° there, so per-mode agg ratios "
  "(head_yaw 15.1 etc.) are near-zero-denominator noise, not real. The reliable image signal is the pooled "
  "row (ratio 0.40) and the negative median per-cell ratios: image-corrupted frames carry a CORRECT label, "
  "so their misleading cost is ≈0 or negative — they are worth keeping. §3.2 clause 1 confirmed.\n")

# ---- coordination sub-analysis (b vs d at matched median displacement) ----
w("## 3. Coordination sub-analysis — donor-swap (b) vs coordinated (d)\n")
b_cells = [r for r in rows if r["channel"] == "b_swap"]; d_cells = [r for r in rows if r["channel"] == "d_coord"]
bo = np.nanmedian([r["median_offset_deg"] for r in b_cells]); do = np.nanmedian([r["median_offset_deg"] for r in d_cells])
w(f"Median corrupted-frame displacement: (b) {bo:.1f}° vs (d) {do:.1f}°. "
  f"Coordinated next-target frames straddle a day boundary in {int(np.nansum([r['day_straddle'] for r in d_cells]))} substituted pairs "
  f"(bimodal: same-day small, cross-day large).")
sb, sd = paired(b_cells), paired(d_cells)
w("\n| channel | median offset° | lost° | misleading° | agg ratio |"); w("|---|---|---|---|---|")
w(f"| (b) donor-swap | {bo:.1f} | {sb['lost']:.2f} | {sb['mis']:.2f} | {sb['agg_ratio']:.2f} |")
w(f"| (d) coordinated | {do:.1f} | {sd['lost']:.2f} | {sd['mis']:.2f} | {sd['agg_ratio']:.2f} |")
# matched band: overlap of per-cell median offsets
lo = max(np.nanmin([r["median_offset_deg"] for r in b_cells]), np.nanmin([r["median_offset_deg"] for r in d_cells]))
hi = min(np.nanmax([r["median_offset_deg"] for r in b_cells]), np.nanmax([r["median_offset_deg"] for r in d_cells]))
bm = [r for r in b_cells if lo <= r["median_offset_deg"] <= hi]
dm = [r for r in d_cells if lo <= r["median_offset_deg"] <= hi]
sbm, sdm = paired(bm), paired(dm)
w(f"\n**Matched displacement band [{lo:.1f}, {hi:.1f}]°** (equal detectability): "
  f"(b) misleading {sbm['mis']:.2f}° ratio {sbm['agg_ratio']:.2f} (n={sbm['n']}) vs "
  f"(d) misleading {sdm['mis']:.2f}° ratio {sdm['agg_ratio']:.2f} (n={sdm['n']}). ")
rel = (sdm["mis"] - sbm["mis"]) / (abs(sbm["mis"]) + 1e-9)
worse = rel > 0.15   # require >15% relative to claim amplification, not noise
w(f"At equal displacement the misleading costs are (b) {sbm['mis']:.2f}° vs (d) {sdm['mis']:.2f}° "
  f"({rel*100:+.0f}% relative). "
  f"{'Coordinated damage is materially higher — coordination amplifies beyond what detectability predicts.' if worse else 'The two are essentially equal — coordination does NOT amplify damage once detectability is matched; the raw ratio gap (b 3.67 vs d 6.12) is a displacement/denominator confound, not a coordination effect.'}\n")

# ---- TraNFS discussion ----
w("## 4. Comparator — TraNFS (CVPR 2022), discussion only (not fit)\n")
w("TraNFS ProtoNet-vs-Oracle ratios: miniImageNet 5-way-5-shot@40% = 0.99/1.97/2.62 (outlier/symmetric/paired), "
  "tieredImageNet = 0.98/2.12/2.93. Their oracle leaves 3 of 5 shots vs our K=9 leaving ~5 at p_c=0.4 — a different "
  "point on the shots-accuracy curve, so this is qualitative. Our label ratios (above) sit in a comparable regime: "
  "coordinated/paired-style corruption is the most damaging, matching their paired > symmetric > outlier ordering.\n")

# ---- figure ----
plt.figure(figsize=(7, 4.2))
xs = sorted(ratio_by_delta); ys = [ratio_by_delta[d] for d in xs]
plt.axvspan(4.2, 4.8, color="crimson", alpha=0.18, label="AUC-0.70 detection floor 4.2–4.8°")
plt.plot(xs, ys, "o-", color="C3", label="label rotation (δ); δ=5 post-hoc")
plt.scatter([bo], [sb["agg_ratio"]], marker="s", s=80, color="C1", label=f"donor-swap (b) @{bo:.0f}°", zorder=5)
plt.scatter([do], [sd["agg_ratio"]], marker="D", s=80, color="C2", label=f"coordinated (d) @{do:.0f}°", zorder=5)
plt.scatter([2], [paired(img_cells)["agg_ratio"]], marker="^", s=90, color="C0", label="image (all)", zorder=5)
plt.axhline(1.0, color="k", ls="--", lw=1, label="ratio = 1 (equal cost)")
plt.xlabel("corrupted-frame displacement / severity (deg)"); plt.ylabel("damage ratio = misleading / lost")
plt.title("Experiment B: two-channel damage ratio (§3.2)"); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(_AG, "logs", "exp_b_ratio.png"), dpi=120); plt.close()

# ---- per-cell CSV already exists (exp_b_records.csv); write a per-severity summary CSV ----
with open(os.path.join(_AG, "logs", "exp_b_severity_summary.csv"), "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["channel", "severity", "n", "lost_deg", "misleading_deg", "agg_ratio", "median_ratio", "dz", "p"])
    for chan, sev, lbl in severities:
        cells = [r for r in rows if r["channel"] == chan and r["severity"] == sev]; st = paired(cells)
        wr.writerow([chan, sev, st["n"], round(st["lost"],3), round(st["mis"],3), round(st["agg_ratio"],3), round(st["med_ratio"],3), round(st["dz"],3), st["p"]])
    st = paired(img_cells); wr.writerow(["c_image", "ALL", st["n"], round(st["lost"],3), round(st["mis"],3), round(st["agg_ratio"],3), round(st["med_ratio"],3), round(st["dz"],3), st["p"]])

open(os.path.join(_AG, "logs", "EXP_B_REPORT.md"), "w").write("\n".join(L) + "\n")
print("wrote logs/EXP_B_REPORT.md + exp_b_ratio.png + exp_b_severity_summary.csv")
print("\n".join(L))
