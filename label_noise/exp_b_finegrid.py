# exp_b_finegrid.py — POST-HOC fine grid for Experiment B's ratio crossing
"""Adds severities 4.25, 4.5, 4.75 (channel a_rot) to MEASURE the damage-ratio
crossing instead of interpolating it.

Why
---
The ratio crosses 1 between delta=4 (-0.28) and delta=5 (1.67), and delta* ~ 4.7
is a linear interpolation from two bracketing samples, not a measurement. That
number carries weight: the paper's coincidence argument turns on delta* falling
inside the ROC-AUC 0.70 detection floor of 4.2-4.8 deg. Three interior points
turn the interpolation into an observation, and either tighten the claim or
correct it.

Design
------
Reuses Experiment B's own Subject class verbatim (via `from exp_b import
Subject`), exactly as exp_b_delta5.py does, so every setting — full held-out
eval, cached clean arm, api.adapt 20-step/1e-5, the SHA-256 (subject, channel,
severity, p_c, seed) seeding — is identical to the existing grid and the new
rows pool with it.

Seeding note: exp_b keys the RNG on the severity's string form, so "4.25" is a
distinct draw from "4" and no existing cell is disturbed. Only fractional
severities are generated here; the integer grid is never regenerated.

Safety
------
APPENDS to exp_b_records.csv, never truncates (exp_b.py itself opens the file
with "w" and would wipe it — do not run exp_b.py to add severities). Backs the
file up first and refuses if any target severity is already present.

Cost: 15 subjects x 3 severities x 3 p_c x 3 seeds = 405 cells (~1 h at the
8.4 s/cell benchmark).

Usage:  python exp_b_finegrid.py
"""
import csv
import os
import shutil
import sys
import time
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_AG = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from exp_b import Subject            # verbatim Exp B settings
import backbone_api as api

SUBJ = [f"p{i:02d}" for i in range(15)]
PCS = [0.1, 0.2, 0.4]
SEEDS = [0, 1, 2]
DELTAS = [4.25, 4.5, 4.75]
CSV = os.path.join(_HERE, "exp_b_records.csv")
FIELDS = ["subject", "channel", "severity", "pc", "seed", "n_corrupted", "clean",
          "oracle", "naive", "lost_cost", "misleading_cost", "ratio",
          "median_offset_deg", "image_modes", "day_straddle"]


def preflight():
    if not os.path.exists(CSV):
        sys.exit(f"ABORT: {CSV} not found.")
    rows = list(csv.DictReader(open(CSV)))
    have = {r["severity"] for r in rows}
    clash = sorted(str(d) for d in DELTAS if str(d) in have)
    if clash:
        sys.exit(f"ABORT: severities already present in the records: {clash}\n"
                 f"       Appending would duplicate them. Remove them first, or "
                 f"edit DELTAS.")
    bak = f"{CSV}.bak_{datetime.now():%Y%m%d_%H%M%S}"
    shutil.copy2(CSV, bak)
    print(f"preflight: {len(rows)} existing rows; backed up -> {os.path.basename(bak)}")
    print(f"preflight: existing a_rot severities "
          f"{sorted({r['severity'] for r in rows if r['channel']=='a_rot'}, key=float)}")
    print(f"preflight: adding {DELTAS}\n")


def main():
    preflight()
    device = api.get_device()
    total = len(SUBJ) * len(DELTAS) * len(PCS) * len(SEEDS)
    n = 0
    t0 = time.time()
    with open(CSV, "a", newline="") as f:          # APPEND — never "w"
        wr = csv.DictWriter(f, fieldnames=FIELDS, restval="")
        for s in SUBJ:
            subj = Subject(s, device)
            for d in DELTAS:
                for pc in PCS:
                    for seed in SEEDS:
                        wr.writerow(subj.cell("a_rot", d, pc, seed))
                        n += 1
            f.flush()
            el = time.time() - t0
            print(f"  [{s}] {n}/{total} cells   {el/60:.1f} min elapsed, "
                  f"~{(el/n)*(total-n)/60:.1f} min left", flush=True)
    print(f"\nappended {n} a_rot rows at delta in {DELTAS} -> exp_b_records.csv")
    print("next: python exp_b_analyze.py   (regenerates logs/EXP_B_REPORT.md)")


if __name__ == "__main__":
    main()
