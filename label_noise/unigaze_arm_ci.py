#!/usr/bin/env python3
# unigaze_arm_ci.py — uncertainty and validity audit for the UniGaze arm
"""The arm's driver reports a single pooled point estimate, c = delta_floor /
e_hat, against the ResNet arm's 1.02-1.12. A point estimate cannot say whether
that is the same constant or a different one, and three apparent findings in
this sweep have already turned out to be truncation artifacts. So this script
asks, in order:

  1. bootstrap CI on the pooled c (resampling the 15 subjects)
  2. per-subject floors, with an explicit account of HOW any subject fails to
     yield one — censored above the grid, censored below, or an unsustained
     crossing — because how they fail is what distinguishes a real effect from
     a grid artifact
  3. how noisy a per-subject AUC actually is, so the spread of per-subject c is
     read against its own error bar rather than taken at face value
  4. the 0.9 crossing, which the ResNet arm also reports (8.0-8.7 deg)
  5. grid resolution at each crossing: measured, or interpolated across a gap?

Usage:
  python3 unigaze_arm_ci.py                 # headline: pc = 0.2
  python3 unigaze_arm_ci.py --pc all        # pooled over all pc
  python3 unigaze_arm_ci.py --pc 0.1        # one rate
"""
import argparse, json, math, os, random
from collections import defaultdict

import numpy as np

TARGETS = (0.7, 0.9)
RESNET_BAND = (1.02, 1.12)      # c across L0-L2, ResNet-18 arm
RESNET_F07 = (4.2, 4.8)         # its 0.7 floor, degrees
RESNET_F09 = (8.0, 8.7)         # its 0.9 floor, degrees
RESNET_GRID_TOP = 10            # top of the ResNet arm's delta grid


def roc_auc(scores, pos):
    s = np.asarray(scores, float); p = np.asarray(pos, bool)
    ok = ~np.isnan(s); s, p = s[ok], p[ok]
    a, b = int(p.sum()), int((~p).sum())
    if a == 0 or b == 0:
        return float("nan")
    u, inv, c = np.unique(s, return_inverse=True, return_counts=True)
    cs = np.cumsum(c); avg = (cs - c + cs + 1) / 2.0
    return float((avg[inv][p].sum() - a * (a + 1) / 2) / (a * b))


def auc_se(auc, na, nb):
    """Hanley-McNeil standard error — the yardstick a per-subject AUC is read
    against. Without it a spread of point estimates says nothing."""
    if na < 1 or nb < 1 or not (auc == auc):
        return float("nan")
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    v = (auc * (1 - auc) + (na - 1) * (q1 - auc ** 2)
         + (nb - 1) * (q2 - auc ** 2)) / (na * nb)
    return math.sqrt(max(v, 0.0))


def crossing(ds, aa, target):
    """First SUSTAINED upward crossing of `target`, linearly interpolated.

    status:
      ok           crossing found and the top of the grid stays >= target
      unsustained  crossed, then fell back below target by the top of the grid
      censored_hi  never reaches target inside the grid (floor above the grid)
      censored_lo  already >= target at the smallest delta (floor below it)
    Returns (delta, status, bracket_lo, bracket_hi).
    """
    ds, aa = list(ds), list(aa)
    top = aa[-1]
    if not (top == top) or top < target:
        return float("nan"), "censored_hi", None, None
    if aa[0] >= target:
        return float("nan"), "censored_lo", None, None
    for i in range(1, len(ds)):
        if aa[i] >= target > aa[i - 1]:
            f = (target - aa[i - 1]) / (aa[i] - aa[i - 1])
            return ds[i - 1] + f * (ds[i] - ds[i - 1]), "ok", ds[i - 1], ds[i]
    return float("nan"), "unsustained", None, None


def curve(recs, deltas):
    g = defaultdict(list)
    for x in recs:
        g[x["delta"]].append(x)
    return [roc_auc([x["loo_residual_deg"] for x in g.get(d, [])],
                    [x["is_corrupted"] for x in g.get(d, [])]) for d in deltas]


def ehat(recs):
    v = [x["loo_residual_deg"] for x in recs if not x["is_corrupted"]]
    return float(np.mean(v)) if v else float("nan")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=os.path.join(here, "unigaze_arm_records.jsonl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--pc", default="0.2", help="'0.2' (default), a rate, or 'all'")
    ap.add_argument("--lam", type=float, default=0.01,
                    help="ridge penalty to select. REQUIRED once the records "
                         "hold a lam sweep: pooling penalties mixes four "
                         "different estimators into one ROC curve, and R and c "
                         "both depend on lam (see logs/ARM_SHAPE_COMPARE.md).")
    ap.add_argument("--boot", type=int, default=4000)
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(here), "logs",
                                "UNIGAZE_ARM_CI.md" if a.pc == "0.2"
                                else f"UNIGAZE_ARM_CI_pc{a.pc}.md")

    allrecs = [json.loads(l) for l in open(a.records)]
    pcs = sorted({r["pc"] for r in allrecs})
    lams = sorted({r["lam"] for r in allrecs if "lam" in r})
    recs = allrecs if a.pc == "all" else [r for r in allrecs
                                          if abs(r["pc"] - float(a.pc)) < 1e-9]
    if lams:
        recs = [r for r in recs if abs(r.get("lam", a.lam) - a.lam) < 1e-12]
    if not recs:
        raise SystemExit(f"no records with pc={a.pc}, lam={a.lam:g}; "
                         f"present: pc {pcs}, lam {lams}")

    by_s = defaultdict(list)
    for r in recs:
        by_s[r["subject"]].append(r)
    subs = sorted(by_s)
    deltas = sorted({r["delta"] for r in recs})

    L = []
    w = lambda *x: L.append(" ".join(str(t) for t in x))
    w("# UniGaze arm — uncertainty and validity audit\n")
    w(f"Filter: pc = {a.pc} (rates present: "
      f"{', '.join(f'{p:g}' for p in pcs)}); lam = {a.lam:g}"
      + (f" (penalties present: {', '.join(f'{x:g}' for x in lams)} — "
         f"c and R both move with lam, so a figure without this filter is "
         f"four estimators averaged together)." if len(lams) > 1 else "."))
    w(f"{len(recs)} records, {len(subs)} subjects, "
      f"{len(deltas)} deltas {[f'{d:g}' for d in deltas]}, "
      f"{len({r['seed'] for r in recs})} seeds.\n")

    # ---------------- pooled ----------------
    pooled = curve(recs, deltas)
    e_all = ehat(recs)
    npos = sum(1 for r in recs if r["is_corrupted"])
    w("## Pooled curve\n")
    w("| delta | AUC | +/- SE | n corrupted |")
    w("|---|---|---|---|")
    for d, v in zip(deltas, pooled):
        cell = [r for r in recs if r["delta"] == d]
        na = sum(1 for r in cell if r["is_corrupted"])
        w(f"| {d:g} | {v:.3f} | {auc_se(v, na, len(cell) - na):.3f} | {na} |")
    w("")
    # --- e_hat, three ways. The estimator is not innocent: see below. ---
    e_match = ehat([r for r in recs if r["delta"] <= RESNET_GRID_TOP])
    e_lo = ehat([r for r in recs if r["delta"] == min(deltas)])
    clean = None
    cj = os.path.join(os.path.dirname(a.records), "unigaze_clean_ehat.json")
    if os.path.exists(cj):
        clean = json.load(open(cj))
        if abs(float(clean.get("lam", a.lam)) - a.lam) > 1e-12:
            w(f"(!) unigaze_clean_ehat.json was computed at lam="
              f"{clean['lam']:g} but this run selects lam={a.lam:g}; the clean "
              f"column below is therefore NOT comparable. Re-run "
              f"`python unigaze_clean_ehat.py --lam {a.lam:g}`.\n")
            clean = None
    w("### e_hat is an estimator, and the choice matters\n")
    w("A 'clean' frame inside a sweep cell is not a frame from a clean fit — its "
      "LOO residual comes from an adaptation whose OTHER support labels are "
      "corrupted, and that contamination grows with delta. So pooling e_hat over "
      "a longer grid inflates it and deflates c, with no change in the "
      "underlying model. Both arms use the pooled definition "
      "(exp_sigma_sweep.e_hat), but only this arm's grid was extended past 10 "
      "deg, so the pooled comparison is not like-for-like.\n")
    w("| definition | e_hat (deg) | note |")
    w("|---|---|---|")
    w(f"| pooled over this whole grid | {e_all:.3f} | matches the driver's number; "
      f"inflated by the delta<={deltas[-1]:g} tail |")
    w(f"| pooled over delta <= {RESNET_GRID_TOP:g} | {e_match:.3f} | grid-matched to "
      f"the ResNet arm — **the like-for-like comparator** |")
    w(f"| mildest delta only ({min(deltas):g} deg) | {e_lo:.3f} | least contaminated "
      f"sweep-based estimate |")
    if clean:
        w(f"| clean fit, true labels | {clean['e_hat_clean']:.3f} | no corruption "
          f"anywhere; the uncontaminated scale |")
    else:
        w("| clean fit, true labels | — | run `python3 unigaze_clean_ehat.py` |")
    w("")

    w("## Crossings (pooled)\n")
    w("| target | delta_floor | resolution | c (grid-matched e_hat) | c (whole-grid) |"
      + (" c (clean) |" if clean else ""))
    w("|---|---|---|---|---|" + ("---|" if clean else ""))
    for t, ref in zip(TARGETS, (RESNET_F07, RESNET_F09)):
        f, st, lo, hi = crossing(deltas, pooled, t)
        if st == "ok":
            res = (f"MEASURED, bracket [{lo:g}, {hi:g}] ({hi - lo:g} deg)"
                   if hi - lo <= 0.5 else
                   f"INTERPOLATED across [{lo:g}, {hi:g}] ({hi - lo:g} deg)")
            row = (f"| {t} | {f:.2f} deg | {res} | **{f / e_match:.3f}** | "
                   f"{f / e_all:.3f} |")
            if clean:
                row += f" {f / clean['e_hat_clean']:.3f} |"
        else:
            row = (f"| {t} | — (`{st}`) | top of grid delta={deltas[-1]:g} reaches "
                   f"AUC {pooled[-1]:.3f} | — | — |" + (" — |" if clean else ""))
        w(row)
    w(f"\nResNet arm for reference: 0.7 floor {RESNET_F07[0]}-{RESNET_F07[1]} deg, "
      f"0.9 floor {RESNET_F09[0]}-{RESNET_F09[1]} deg, "
      f"c = {RESNET_BAND[0]}-{RESNET_BAND[1]}.\n")

    # ---------------- per subject ----------------
    w("## Per subject\n")
    ex = [r for r in recs if r["subject"] == subs[0] and r["delta"] == deltas[0]]
    exa = sum(1 for r in ex if r["is_corrupted"])
    w(f"A subject-delta cell holds ~{exa} corrupted and ~{len(ex) - exa} clean "
      f"frames, so a per-subject AUC near 0.7 carries a standard error of about "
      f"{auc_se(0.7, max(exa,1), max(len(ex)-exa,1)):.3f}. Read the spread of c "
      f"below against that number before treating it as variation.\n")
    w("| subject | e_hat (grid-matched) | e_hat (clean) | AUC@top | floor@0.7 | "
      "status | c | floor@0.9 | status |")
    w("|---|---|---|---|---|---|---|---|---|")
    cs, fails = [], defaultdict(list)
    for s in subs:
        cv = curve(by_s[s], deltas)
        e = ehat([x for x in by_s[s] if x["delta"] <= RESNET_GRID_TOP])
        ec = clean["per_subject"].get(s, float("nan")) if clean else float("nan")
        f7, s7, _, _ = crossing(deltas, cv, 0.7)
        f9, s9, _, _ = crossing(deltas, cv, 0.9)
        c = f7 / e if s7 == "ok" else float("nan")
        (cs.append(c) if s7 == "ok" else fails[s7].append(s))
        w(f"| {s} | {e:.2f} | {ec:.2f} | {cv[-1]:.3f} | {f7:.2f} | {s7} | "
          f"{c:.3f} | {f9:.2f} | {s9} |".replace("nan", "—"))
    w("")
    w(f"Usable 0.7 floor for {len(cs)}/{len(subs)} subjects.")
    for k, v in sorted(fails.items()):
        w(f"- `{k}`: {', '.join(v)}")
    if cs:
        w(f"\nPer-subject c: mean {np.mean(cs):.3f}, median {np.median(cs):.3f}, "
          f"sd {np.std(cs, ddof=1):.3f}, range [{min(cs):.3f}, {max(cs):.3f}].\n")

    # ---------------- bootstrap ----------------
    rng = random.Random(0)
    bf, nbad = [], 0
    bc = {"grid-matched": [], "whole-grid": []}
    if clean:
        bc["clean"] = []
    for _ in range(a.boot):
        pick = [rng.choice(subs) for _ in subs]
        rr = [x for s in pick for x in by_s[s]]
        f, st, _, _ = crossing(deltas, curve(rr, deltas), 0.7)
        if st != "ok":
            nbad += 1
            continue
        bf.append(f)
        bc["whole-grid"].append(f / ehat(rr))
        bc["grid-matched"].append(
            f / ehat([x for x in rr if x["delta"] <= RESNET_GRID_TOP]))
        if clean:
            bc["clean"].append(
                f / float(np.mean([clean["per_subject"][s] for s in pick])))
    w("## Bootstrap over subjects\n")
    w(f"{a.boot} replicates; {nbad} ({100 * nbad / a.boot:.1f}%) gave no usable "
      f"0.7 crossing and are dropped. Subjects are resampled with replacement; "
      f"e_hat is recomputed inside each replicate, so the interval carries the "
      f"uncertainty of the ratio, not of the floor alone.\n")
    q = lambda v, p: float(np.percentile(v, p))
    if bf:
        w(f"- delta_floor@0.7: {np.median(bf):.2f} deg, 95% CI "
          f"[{q(bf, 2.5):.2f}, {q(bf, 97.5):.2f}]\n")
        w(f"ResNet-18 arm: c = {RESNET_BAND[0]}-{RESNET_BAND[1]} (L0-L2).\n")
        w("| e_hat definition | c | 95% CI | overlaps ResNet band | P(inside) | "
          "P(below) | P(above) |")
        w("|---|---|---|---|---|---|---|")
        verdicts = {}
        for k in ("grid-matched", "whole-grid", "clean"):
            v = bc.get(k)
            if not v:
                continue
            lo, hi = q(v, 2.5), q(v, 97.5)
            ov = not (hi < RESNET_BAND[0] or lo > RESNET_BAND[1])
            verdicts[k] = ov
            w(f"| {k} | {np.median(v):.3f} | [{lo:.3f}, {hi:.3f}] | "
              f"**{'YES' if ov else 'NO'}** | "
              f"{np.mean([RESNET_BAND[0] <= x <= RESNET_BAND[1] for x in v]):.2f} | "
              f"{np.mean([x < RESNET_BAND[0] for x in v]):.2f} | "
              f"{np.mean([x > RESNET_BAND[1] for x in v]):.2f} |")
        w("")
        prim = bc["grid-matched"]
        plo, phi = q(prim, 2.5), q(prim, 97.5)
        agree = len(set(verdicts.values())) == 1
        w("**Reading.** The grid-matched row is the like-for-like comparator; the "
          "others are shown so the estimator's contribution is visible.")
        if not agree:
            w("The three definitions do NOT agree on whether the interval covers "
              "the ResNet band. That means the comparison is decided by how "
              "e_hat is estimated, not by the data, and no claim about "
              "architecture invariance can be made until the two arms use one "
              "definition on comparable grids. Report this, do not pick the "
              "favourable row.")
        elif verdicts["grid-matched"]:
            below = float(np.mean([x < RESNET_BAND[0] for x in prim]))
            above = float(np.mean([x > RESNET_BAND[1] for x in prim]))
            side = ("below" if below > above else "above")
            w(f"The interval [{plo:.3f}, {phi:.3f}] does not exclude the ResNet "
              f"band, which licenses 'not distinguishable' and nothing stronger — "
              f"an interval this wide would also fail to exclude a c that differed "
              f"by {100 * max(phi / np.mean(RESNET_BAND) - 1, 1 - plo / np.mean(RESNET_BAND)):.0f}%. "
              f"{max(below, above):.0%} of the mass sits {side} the band.")
        else:
            w(f"The interval [{plo:.3f}, {phi:.3f}] excludes the ResNet band.")
        w("\nIn every case: this arm changes adaptation depth (frozen backbone + "
          "ridge head) at the same time as architecture, so architecture and "
          "adaptation depth are confounded and neither can be assigned the "
          "effect. Say so wherever the number is reported.")
    w("")

    # ---------------- grid audit ----------------
    w("## Grid audit\n")
    for t, ref in zip(TARGETS, (RESNET_F07, RESNET_F09)):
        f, st, lo, hi = crossing(deltas, pooled, t)
        if st == "ok":
            w(f"- {t} crossing sits between delta={lo:g} and delta={hi:g} "
              f"({hi - lo:g} deg apart). "
              + ("Measured." if hi - lo <= 0.5 else
                 "INTERPOLATED across that gap — add interior points before "
                 "quoting the figure to two decimals."))
        else:
            w(f"- {t} crossing: `{st}`; top of grid delta={deltas[-1]:g} reaches "
              f"{pooled[-1]:.3f}. "
              + (f"Proportionality with the ResNet arm would place it near "
                 f"{e_all * np.mean(ref) * np.mean(RESNET_BAND) / np.mean(RESNET_F07):.1f}"
                 f" deg (that arm's {t} floor expressed model-relatively, scaled "
                 f"by this arm's e_hat) — extend the grid rather than reporting "
                 f"the crossing as absent."
                 if st == "censored_hi" else ""))
    w("- AUC below 0.5 at small delta is the chance plateau of Section 5.1, not "
      "a defect: below the floor the scorer has no signal and its sampling "
      "error straddles chance. The crossing search starts from the low end and "
      "requires the top of the grid to stay above target, so a dip there cannot "
      "produce a spurious floor.")

    txt = "\n".join(L) + "\n"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write(txt)
    print(txt)
    print("wrote", out)


if __name__ == "__main__":
    main()
