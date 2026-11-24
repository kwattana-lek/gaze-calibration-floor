"""Review item 8: stability of the damage ratio R = misleading / lost = (naive-oracle)/(oracle-clean).

Reads exp_b_records.csv (read-only). Per channel x severity reports, with a subject bootstrap (B = 2000, seed 0):
  numerator and denominator (means over cells, as in exp_b_analyze.py) with 95% CIs,
  the share of cells and of subjects whose lost-frame cost is <= 0,
  the share of bootstrap replicates whose denominator is <= 0.05 deg or <= 0,
  the ratio-of-means R with a 95% percentile CI (replicates with denominator <= 0 set to NaN and excluded; their share is reported),
  and the paired difference misleading - lost with a 95% CI, which stays defined when the denominator is small.
Writes results/07_damage_ratio_denominator.{json,csv}.
"""
import csv, json, os
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REV = os.path.dirname(HERE); LN = os.path.dirname(REV)
rows = list(csv.DictReader(open(os.path.join(LN, "exp_b_records.csv"))))
B, SEED = 2000, 0
groups = defaultdict(list)
for r in rows:
    ch = {"a_rot": "label", "b_swap": "label_donor_swap", "c_image": "image", "d_coord": "label_coordinated"}[r["channel"]]
    sev = r["severity"] if r["channel"] == "a_rot" else "all"
    groups[(ch, sev)].append(r)

out = []
for (ch, sev), cells in sorted(groups.items(), key=lambda x: (x[0][0], float(x[0][1]) if x[0][1] != "all" else 99)):
    by = defaultdict(lambda: ([], []))
    for r in cells:
        try:
            l, m = float(r["lost_cost"]), float(r["misleading_cost"])
        except ValueError:
            continue
        if np.isfinite(l) and np.isfinite(m):
            by[r["subject"]][0].append(l); by[r["subject"]][1].append(m)
    subs = sorted(by)
    Ls = np.array([np.mean(by[s][0]) for s in subs]); Ms = np.array([np.mean(by[s][1]) for s in subs])
    allL = np.concatenate([by[s][0] for s in subs])
    g = np.random.RandomState(SEED)
    bl, bm, br, bd = [], [], [], []
    for _ in range(B):
        idx = g.randint(0, len(subs), len(subs))
        l, m = Ls[idx].mean(), Ms[idx].mean()
        bl.append(l); bm.append(m); br.append(m / l if l > 0 else np.nan); bd.append(m - l)
    bl, bm, br, bd = map(np.array, (bl, bm, br, bd))
    rec = dict(channel=ch, severity=sev, n_subjects=len(subs), n_cells=int(len(allL)),
               lost_mean=float(Ls.mean()), lost_ci=[float(np.percentile(bl, 2.5)), float(np.percentile(bl, 97.5))],
               misleading_mean=float(Ms.mean()), misleading_ci=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
               R=float(Ms.mean() / Ls.mean()),
               R_ci=[float(np.nanpercentile(br, 2.5)), float(np.nanpercentile(br, 97.5))],
               diff_mis_minus_lost=float(Ms.mean() - Ls.mean()),
               diff_ci=[float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
               frac_cells_lost_le0=float((allL <= 0).mean()),
               n_subjects_lost_le0=int((Ls <= 0).sum()),
               frac_boot_denom_le0=float((bl <= 0).mean()),
               frac_boot_denom_le005=float((bl <= 0.05).mean()))
    out.append(rec)
json.dump(dict(source="exp_b_records.csv", B=B, seed=SEED, rows=out), open(os.path.join(REV, "results", "07_damage_ratio_denominator.json"), "w"), indent=1)
with open(os.path.join(REV, "results", "07_damage_ratio_denominator.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["channel", "severity", "n_subj", "n_cells", "lost", "lost_lo", "lost_hi", "mis", "mis_lo", "mis_hi", "R", "R_lo", "R_hi", "cells_lost_le0", "subj_lost_le0", "boot_den_le0", "boot_den_le0.05", "mis_minus_lost", "diff_lo", "diff_hi"])
    for r in out:
        w.writerow([r["channel"], r["severity"], r["n_subjects"], r["n_cells"], *[f"{x:.3f}" for x in (r["lost_mean"], *r["lost_ci"], r["misleading_mean"], *r["misleading_ci"], r["R"], *r["R_ci"])],
                    f"{r['frac_cells_lost_le0']:.2f}", r["n_subjects_lost_le0"], f"{r['frac_boot_denom_le0']:.3f}", f"{r['frac_boot_denom_le005']:.3f}", *[f"{x:.3f}" for x in (r["diff_mis_minus_lost"], *r["diff_ci"])]])
print(open(os.path.join(REV, "results", "07_damage_ratio_denominator.csv")).read())
