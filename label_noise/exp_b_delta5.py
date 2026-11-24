# exp_b_delta5.py — POST-HOC addendum to Experiment B (c569c1f)
"""Adds ONE severity level (channel a_rot, δ=5) to narrow the ratio-crossing bracket
between δ=4 and δ=6. NOT part of the 25cde1c pre-registration — post-hoc.

Reuses Experiment B's own Subject class verbatim, so every setting (full ~2991-frame
held-out eval, cached clean arm, api.adapt 20-step/1e-5, same seeding scheme) is
identical to the Exp B grid and the rows are poolable. Appends to exp_b_records.csv
with the same schema.
"""
import csv, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); _AG = os.path.dirname(_HERE)
if _HERE not in sys.path: sys.path.insert(0, _HERE)
from exp_b import Subject          # verbatim Exp B settings
import backbone_api as api

SUBJ = [f"p{i:02d}" for i in range(15)]; PCS = [0.1, 0.2, 0.4]; SEEDS = [0, 1, 2]; DELTA = 5
CSV = os.path.join(_HERE, "exp_b_records.csv")
FIELDS = ["subject", "channel", "severity", "pc", "seed", "n_corrupted", "clean", "oracle", "naive",
          "lost_cost", "misleading_cost", "ratio", "median_offset_deg", "image_modes", "day_straddle"]


def main():
    device = api.get_device(); n = 0
    with open(CSV, "a", newline="") as f:          # APPEND — do not touch existing rows
        wr = csv.DictWriter(f, fieldnames=FIELDS, restval="")
        for s in SUBJ:
            subj = Subject(s, device)
            for pc in PCS:
                for seed in SEEDS:
                    wr.writerow(subj.cell("a_rot", DELTA, pc, seed)); n += 1
            f.flush(); print(f"  [{s}] done (+{n} a_rot@5 cells)", flush=True)
    print(f"appended {n} a_rot@δ=5 rows -> exp_b_records.csv")


if __name__ == "__main__":
    main()
