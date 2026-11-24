#!/usr/bin/env python
# exp_a_family_stats.py — paired per-subject tests for the FAMILY BOUNDARY claim
"""Experiment A's headline claim is that scorers separate by the kind of quantity
they compute — population-statistical versus per-frame — rather than by which
literature they come from. That claim is currently supported by ROC-AUC ranges
across contamination rates, while the narrower leave-one-out-vs-resubstitution
comparison gets full paired per-subject tests. This script gives the headline
claim the same treatment.

Families (Section 4.3 of the manuscript):
  population-statistical : dividemix (mixture over losses),
                           coteaching (cross-model disagreement),
                           suge       (neighbour consistency)
  per-frame              : loo        (leave-one-out residual),
                           small_loss (resubstitution residual),
                           source     (no-adaptation residual)

Three tests, in increasing strictness:
  (1) family means  — mean per-frame AUC vs mean population AUC, per subject
  (2) worst vs best — the WEAKEST per-frame scorer vs the STRONGEST population
                      scorer, per subject. This is the boundary claim at its most
                      demanding: if it holds, no population scorer beats any
                      per-frame scorer on that subject.
  (3) all 9 pairs   — every per-frame scorer against every population scorer.

All tests are paired per subject across 15 subjects, using an exact Wilcoxon
signed-rank test (full 2^n enumeration, no scipy) and Cohen's d_z, matching the
convention used elsewhere in the paper. Per-subject AUC pools that subject's
seeds within the condition.

Usage:  python exp_a_family_stats.py [--records PATH] [--out PATH]
"""
import argparse
import itertools
import json
import os
import statistics
from collections import defaultdict

import numpy as np

PER_FRAME = ["loo", "small_loss", "source"]
POPULATION = ["dividemix", "coteaching", "suge"]
PRETTY = {"loo": "leave-one-out", "small_loss": "resubstitution",
          "source": "source residual", "dividemix": "mixture over losses",
          "coteaching": "cross-model disagr.", "suge": "neighbour consistency"}


def roc_auc(scores, pos):
    s = np.asarray(scores, float)
    p = np.asarray(pos, bool)
    ok = ~np.isnan(s)
    s, p = s[ok], p[ok]
    a, b = int(p.sum()), int((~p).sum())
    if a == 0 or b == 0:
        return float("nan")
    u, inv, c = np.unique(s, return_inverse=True, return_counts=True)
    cs = np.cumsum(c)
    avg = (cs - c + cs + 1) / 2.0
    return float((avg[inv][p].sum() - a * (a + 1) / 2) / (a * b))


def exact_wilcoxon(x):
    """Exact two-sided Wilcoxon signed-rank against 0; zeros dropped."""
    v = [a for a in x if a != 0]
    n = len(v)
    if n == 0:
        return 1.0
    if n > 20:
        raise ValueError("exact enumeration only for n<=20")
    order = sorted(range(n), key=lambda i: abs(v[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(v[order[j + 1]]) == abs(v[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    w = sum(r for r, a in zip(ranks, v) if a > 0)
    tot = sum(ranks)
    target = abs(w - tot / 2)
    hits = sum(1 for sg in itertools.product((0, 1), repeat=n)
               if abs(sum(r for r, b in zip(ranks, sg) if b) - tot / 2) >= target - 1e-12)
    return min(1.0, hits / 2 ** n)


def dz(d):
    sd = statistics.pstdev(d) * (len(d) / (len(d) - 1)) ** 0.5 if len(d) > 1 else 0.0
    return statistics.mean(d) / sd if sd else 0.0


def subject_aucs(recs, cond, scorers):
    """subject -> {scorer: AUC} for one condition."""
    cells = defaultdict(list)
    for r in recs:
        if cond == "structured":
            if r["model"] != "structured_swap":
                continue
        else:
            if r["model"] != "random_offset" or r["delta"] != cond:
                continue
        cells[r["subject"]].append(r)
    out = {}
    for s, rows in cells.items():
        pos = [x["is_corrupted"] for x in rows]
        out[s] = {k: roc_auc([x.get(f"score_{k}") for x in rows], pos) for k in scorers}
    return out


def report(recs, conds, w):
    scorers = PER_FRAME + POPULATION
    for cond in conds:
        lbl = "structured donor-swap" if cond == "structured" else f"delta = {cond}"
        per = subject_aucs(recs, cond, scorers)
        subs = sorted(per)
        if len(subs) < 5:
            continue
        w(f"\n### {lbl}  (n = {len(subs)} subjects)\n")

        # (1) family means
        pf = [statistics.mean(per[s][k] for k in PER_FRAME) for s in subs]
        po = [statistics.mean(per[s][k] for k in POPULATION) for s in subs]
        d = [a - b for a, b in zip(pf, po)]
        w("| test | per-frame | population | mean diff | d_z | p |")
        w("|---|---|---|---|---|---|")
        w(f"| family means | {statistics.mean(pf):.3f} | {statistics.mean(po):.3f} | "
          f"{statistics.mean(d):+.3f} | {dz(d):+.2f} | {exact_wilcoxon(d):.4f} |")

        # (2) worst per-frame vs best population — the strong form
        wf = [min(per[s][k] for k in PER_FRAME) for s in subs]
        bp = [max(per[s][k] for k in POPULATION) for s in subs]
        d2 = [a - b for a, b in zip(wf, bp)]
        n_win = sum(1 for x in d2 if x > 0)
        w(f"| worst per-frame vs best population | {statistics.mean(wf):.3f} | "
          f"{statistics.mean(bp):.3f} | {statistics.mean(d2):+.3f} | {dz(d2):+.2f} | "
          f"{exact_wilcoxon(d2):.4f} |")
        w(f"\nStrong form holds for {n_win}/{len(subs)} subjects "
          f"(weakest per-frame scorer beats strongest population scorer).\n")

        # (3) all 9 pairs
        w("| per-frame | vs population | mean diff | d_z | p |")
        w("|---|---|---|---|---|")
        for a in PER_FRAME:
            for b in POPULATION:
                dd = [per[s][a] - per[s][b] for s in subs]
                w(f"| {PRETTY[a]} | {PRETTY[b]} | {statistics.mean(dd):+.3f} | "
                  f"{dz(dd):+.2f} | {exact_wilcoxon(dd):.4f} |")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=os.path.join(here, "exp_a_records.jsonl"))
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(here), "logs", "EXP_A_FAMILY_STATS.md"))
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.records)]
    L = []
    w = lambda *x: L.append(" ".join(str(t) for t in x))
    w("# Experiment A — family boundary, paired per-subject tests\n")
    w("Population-statistical: " + ", ".join(PRETTY[k] for k in POPULATION) + ".")
    w("Per-frame: " + ", ".join(PRETTY[k] for k in PER_FRAME) + ".\n")
    w("Paired per subject, exact Wilcoxon signed-rank (full enumeration) and "
      "Cohen's d_z. Per-subject AUC pools that subject's seeds.\n")
    report(recs, [4, 6, "structured"], w)
    txt = "\n".join(L) + "\n"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write(txt)
    print(txt)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
