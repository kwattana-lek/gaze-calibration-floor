#!/usr/bin/env python3
# exp_identifiability.py -- is the high-p_c regime a high floor, or no floor at all?
"""THE QUESTION

At p_c = 0.4 with kappa = 1 the ROC-AUC never sustains 0.7 out to delta = 50, and
the kappa = 0.75 crossing swings 18.9-36.8 across lambda. Quoting either as a
"floor" would repeat the high-end truncation artifact of Section 5. The
hypothesis is that there is no floor to find, because the problem stops being
identifiable.

A residual scorer answers "which frames disagree with the mapping the other
frames imply". That question presupposes the clean frames form the only
self-consistent set. Coherent displacement breaks the presupposition: rotating
several labels by the SAME amount leaves them mutually consistent with a rotated
mapping, so the corrupted subset is a rival hypothesis rather than a set of
outliers. When both subsets are internally self-consistent, no residual scorer
can prefer one -- not a hard detection problem, an ill-posed one.

THE MEASUREMENT

For each cell, split the 9 support frames into the clean subset C and the
corrupted subset X, and fit each subset AGAINST ITS OWN LABELS:

    within_clean   = mean LOO residual inside C, against the true labels
    within_corrupt = mean LOO residual inside X, against the CORRUPTED labels

X is scored against the corrupted labels deliberately: the rival hypothesis is
"those labels are the correct ones". The ratio

    I = within_corrupt / within_clean

is the identifiability index. I >> 1 means the corrupted set is internally
incoherent and can be recognised as noise. I -> 1 means the two subsets are
equally self-consistent and the data cannot say which is which.

Both subsets need >= 3 frames for a leave-one-out fit, so cells that fail that
are skipped and counted.

WRITES only its own summary.
"""
import argparse, json, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import corruption_temporal as ct      # noqa: E402
import exp_coherence as E             # noqa: E402
from unigaze_arm import SUBJ, FEAT_DIR   # noqa: E402

try:
    from prepare_rtgene_labels import REPEAT_SESSIONS
except Exception:
    REPEAT_SESSIONS = {}

OUT = os.path.join(_HERE, "identifiability_summary.json")
MIN_SUB = 3


def within(F, Y, lam):
    """Mean leave-one-out residual inside a subset, against that subset's own
    labels. Measures how self-consistent the subset is on its own terms."""
    if len(F) < MIN_SUB:
        return None
    return float(np.mean(E.loo_resid(E.loo_functionals(F, lam), Y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--feat", default=None,
                    help="feature cache dir; defaults to the dataset unigaze_arm selected")
    ap.add_argument("--subj", default=None, help="comma-separated subject ids")
    ap.add_argument("--label", default="MPIIFaceGaze")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--deltas", default="6,10,16,20",
                    help="severities at which to report I. The default suits "
                         "MPIIFaceGaze; a corpus with a larger ehat needs larger "
                         "deltas to probe the same relative regime.")
    a = ap.parse_args()

    subj = ([x for x in a.subj.split(",") if x] if a.subj else list(SUBJ))
    feat_dir = os.path.join(_HERE, a.feat) if a.feat else FEAT_DIR
    deltas = [float(x) if "." in x else int(x) for x in a.deltas.split(",") if x]

    # No leakage here -- I is computed inside a subject and no cross-subject map is
    # fitted -- but a repeated participant would still carry twice the weight in the
    # mean over subjects, so it is worth saying out loud.
    dup = [x for x in subj if x in REPEAT_SESSIONS and REPEAT_SESSIONS[x] in subj]
    if dup:
        print("WARNING: " + ", ".join(f"{x} repeats {REPEAT_SESSIONS[x]}" for x in dup)
              + " -- that person is weighted twice in the mean.", file=sys.stderr)

    feats = E.load_feats("unigaze", "L0", feat_dir=feat_dir, subjects=subj)
    print(f"dataset: {a.label} -- {len(subj)} subjects from "
          f"{os.path.basename(feat_dir)}")
    out = {"label": a.label, "subjects": subj, "lam": a.lam, "deltas": deltas}
    print(f"Identifiability index I = within_corrupt / within_clean, lambda={a.lam:g}")
    print("I >> 1: corrupted set is incoherent noise.  I ~ 1: rival hypothesis.\n")
    for pc in (0.2, 0.4):
        print(f"--- p_c = {pc} ---")
        print(f"{'delta':>7}" + "".join(f"{'k='+str(k):>9}" for k in E.KAPPAS)
              + f"{'  (cells used)':>16}")
        for d in deltas:
            line, used = f"{d:>7}", 0
            for kap in E.KAPPAS:
                rs = []
                for s in subj:
                    F, Y = feats[s]
                    for seed in range(a.seeds):
                        c = ct.apply_coherent(Y, s, d, pc, seed, kap,
                                              locality="clustered")
                        m = c["is_corrupted"]
                        wc = within(F[~m], Y[~m], a.lam)
                        wx = within(F[m], c["corrupted_labels"][m], a.lam)
                        if wc and wx and wc > 0:
                            rs.append(wx / wc)
                line += f"{np.mean(rs):>9.2f}" if rs else f"{'--':>9}"
                out[f"{pc}|{d}|{kap}"] = float(np.mean(rs)) if rs else None
                used = len(rs)
            print(line + f"{used:>16}")
        print()

    json.dump(out, open(a.out, "w"), indent=2)
    print(f"wrote {os.path.basename(a.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
