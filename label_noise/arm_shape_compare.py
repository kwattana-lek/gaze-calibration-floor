#!/usr/bin/env python3
# arm_shape_compare.py — do the two arms share the SHAPE of the detectability curve?
"""The finding this script exists to test.

The UniGaze arm agrees with the ResNet arm at the 0.7 floor — c = 1.162,
95% CI [0.952, 1.254] against 1.02-1.12 — and disagrees badly at 0.9, where its
floor lands at 22.5 deg (c9 = 3.40) against the ResNet arm's 8.0-8.7.

Both of those readings are ratios to e_hat, and e_hat is an estimator whose
value moves with how it is measured (unigaze_clean_ehat.py exists because of
that). A disagreement that appears in one ratio and not the other could easily
be the estimator rather than the model.

So test the thing that needs no e_hat at all:

    R = delta_floor@0.9 / delta_floor@0.7

R is dimensionless and scale-free. Every normalization cancels: if the two arms
differ only in accuracy — if one is simply a rescaled copy of the other, which
is exactly what 'the floor is model-relative' asserts — then R is the same in
both, whatever e_hat is or how it is estimated. If R differs, the arms have
DIFFERENT-SHAPED detectability curves, and the single constant c cannot describe
both: it would have to be quoted per operating point.

That is a sharper and more falsifiable claim than the c comparison, and it is
the one the paper should make.

Method
------
Per arm, per level: pool records, sweep the delta grid for ROC-AUC of the LOO
residual against is_corrupted, take the first SUSTAINED upward crossing of each
target by linear interpolation, form R. Bootstrap over subjects (2000 reps) for
a CI on R and on the difference between arms. A replicate is dropped only if a
crossing is unavailable, and the drop rate is reported, because a high drop rate
is itself a censoring warning.

Note the grids differ: the ResNet sweep tops out at delta=10, the UniGaze arm now
runs to 26. If the ResNet 0.9 crossing were censored at 10 this comparison would
be meaningless, so the script reports the censoring status of every crossing and
refuses to compare a censored one.

Usage:  python3 arm_shape_compare.py
"""
import argparse, json, os, random
from collections import defaultdict

import numpy as np

TARGETS = (0.7, 0.9)


def roc_auc(scores, pos):
    s = np.asarray(scores, float); p = np.asarray(pos, bool)
    ok = ~np.isnan(s); s, p = s[ok], p[ok]
    a, b = int(p.sum()), int((~p).sum())
    if a == 0 or b == 0:
        return float("nan")
    u, inv, c = np.unique(s, return_inverse=True, return_counts=True)
    cs = np.cumsum(c); avg = (cs - c + cs + 1) / 2.0
    return float((avg[inv][p].sum() - a * (a + 1) / 2) / (a * b))


def crossing(ds, aa, target):
    ds, aa = list(ds), list(aa)
    if not (aa[-1] == aa[-1]) or aa[-1] < target:
        return float("nan"), "censored_hi"
    if aa[0] >= target:
        return float("nan"), "censored_lo"
    for i in range(1, len(ds)):
        if aa[i] >= target > aa[i - 1]:
            f = (target - aa[i - 1]) / (aa[i] - aa[i - 1])
            return ds[i - 1] + f * (ds[i] - ds[i - 1]), "ok"
    return float("nan"), "unsustained"


def curve(recs, deltas):
    g = defaultdict(list)
    for x in recs:
        g[x["delta"]].append(x)
    return [roc_auc([x["_r"] for x in g.get(d, [])],
                    [x["_c"] for x in g.get(d, [])]) for d in deltas]


def ratio(recs, deltas):
    cv = curve(recs, deltas)
    f7, s7 = crossing(deltas, cv, 0.7)
    f9, s9 = crossing(deltas, cv, 0.9)
    R = f9 / f7 if (s7 == "ok" and s9 == "ok") else float("nan")
    return f7, s7, f9, s9, R, cv


def load(path, resid_key, corr_key="is_corrupted", model="random_offset",
         lam=None):
    """lam: when the file carries a `lam` field, keep only that value. A file
    holding a lam sweep MUST be filtered — pooling ridge penalties into one
    ROC curve mixes different estimators and is meaningless."""
    out = []
    for l in open(path):
        r = json.loads(l)
        if model and r.get("model") != model:
            continue
        if lam is not None and "lam" in r and abs(float(r["lam"]) - lam) > 1e-12:
            continue
        v = r.get(resid_key)
        if v is None:
            continue
        r["_r"], r["_c"] = float(v), bool(r[corr_key])
        out.append(r)
    return out


def analyse(name, recs, w, boot, rng):
    subs = sorted({r["subject"] for r in recs})
    deltas = sorted({r["delta"] for r in recs})
    by_s = defaultdict(list)
    for r in recs:
        by_s[r["subject"]].append(r)
    f7, s7, f9, s9, R, cv = ratio(recs, deltas)
    w(f"\n### {name}\n")
    w(f"{len(recs)} records, {len(subs)} subjects, grid "
      f"{deltas[0]:g}-{deltas[-1]:g} ({len(deltas)} points), "
      f"top-of-grid AUC {cv[-1]:.3f}.\n")
    w(f"- floor@0.7 = {f7:.2f} deg (`{s7}`)")
    w(f"- floor@0.9 = {f9:.2f} deg (`{s9}`)")
    if R != R:
        w(f"- **R = delta_0.9 / delta_0.7: unavailable** — a crossing is censored, "
          f"so this arm cannot enter the shape comparison. Extend its grid.")
        return None
    bs, nbad = [], 0
    for _ in range(boot):
        pick = [rng.choice(subs) for _ in subs]
        rr = [x for s in pick for x in by_s[s]]
        r = ratio(rr, deltas)[4]
        (bs.append(r) if r == r else None)
        nbad += (r != r)
    q = lambda p: float(np.percentile(bs, p))
    w(f"- **R = {R:.3f}**, bootstrap median {np.median(bs):.3f}, 95% CI "
      f"[{q(2.5):.3f}, {q(97.5):.3f}]  ({nbad}/{boot} replicates dropped)")
    return {"name": name, "R": R, "boot": bs, "f7": f7, "f9": f9}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--pc", type=float, default=0.2)
    ap.add_argument("--lam", type=float, default=0.01,
                    help="ridge penalty to select in the frozen arms; files "
                         "holding a lam sweep are filtered to this value")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(here), "logs",
                                                  "ARM_SHAPE_COMPARE.md"))
    a = ap.parse_args()
    rng = random.Random(0)

    L = []
    w = lambda *x: L.append(" ".join(str(t) for t in x))
    w("# Do the two arms share the shape of the detectability curve?\n")
    w("R = delta_floor@0.9 / delta_floor@0.7 is dimensionless: it cancels e_hat "
      "and every choice about how e_hat is estimated. If the model-relative "
      "framing is right and the arms differ only in accuracy, R is the same in "
      "both. If R differs, the curves have different shapes and no single "
      "constant c describes both — c would have to be quoted per operating "
      "point.\n")

    arms = []

    # ---- ResNet-18 sigma sweep, per level ----
    p = os.path.join(here, "sigma_sweep_records.jsonl")
    if os.path.exists(p):
        rs = load(p, "loo_residual_deg")
        rs = [r for r in rs if abs(r.get("pc", a.pc) - a.pc) < 1e-9] or rs
        lv = defaultdict(list)
        for r in rs:
            lv[r.get("sigma_level", "?")].append(r)
        for k in sorted(lv):
            got = analyse(f"ResNet-18 {k} (full fine-tune)", lv[k], w, a.boot, rng)
            if got:
                arms.append(got)
    else:
        w(f"\n(!) {os.path.basename(p)} not found — ResNet arm skipped.")

    # ---- ResNet-18 FROZEN + ridge — the 2x2 cell that breaks the confound ----
    p = os.path.join(here, "resnet_frozen_records.jsonl")
    if os.path.exists(p):
        fs = load(p, "loo_residual_deg", lam=a.lam)
        fs = [r for r in fs if abs(r["pc"] - a.pc) < 1e-9]
        lv = defaultdict(list)
        for r in fs:
            lv[r.get("level", "?")].append(r)
        for k in sorted(lv):
            got = analyse(f"ResNet-18 {k} (FROZEN + ridge head)", lv[k], w,
                          a.boot, rng)
            if got:
                arms.append(got)
    else:
        w(f"\n(!) {os.path.basename(p)} not found — the ResNet frozen arm has "
          f"not been run. Without it the architecture/adaptation confound "
          f"stays open: run `python resnet_frozen_arm.py` on the Mac.")

    # ---- UniGaze arm ----
    p = os.path.join(here, "unigaze_arm_records.jsonl")
    if os.path.exists(p):
        us = load(p, "loo_residual_deg", lam=a.lam)
        us = [r for r in us if abs(r["pc"] - a.pc) < 1e-9]
        got = analyse("UniGaze ViT-H/14 (frozen + ridge head)", us, w, a.boot, rng)
        if got:
            arms.append(got)
    else:
        w(f"\n(!) {os.path.basename(p)} not found — UniGaze arm skipped.")

    # ---- lambda robustness: this GATES every verdict below ----
    # Learned the hard way. At lam=0.01 this script reported "adaptation depth
    # carries it" (p = 0.0038). At lam=1.0 both contrasts are null and all four
    # cells agree. R is a function of the ridge penalty, so a verdict from a
    # single lam is not a finding about models — it is a finding about lam.
    lam_tab = []
    for path, lbl in ((os.path.join(here, "resnet_frozen_records.jsonl"),
                       "ResNet-18 frozen"),
                      (os.path.join(here, "unigaze_arm_records.jsonl"),
                       "UniGaze frozen")):
        if not os.path.exists(path):
            continue
        rows = load(path, "loo_residual_deg")
        rows = [r for r in rows if abs(r["pc"] - a.pc) < 1e-9]
        for lm in sorted({r["lam"] for r in rows if "lam" in r}):
            sel = [r for r in rows if abs(r["lam"] - lm) < 1e-12]
            for k in sorted({r.get("level", "-") for r in sel}):
                ss = [r for r in sel if r.get("level", "-") == k]
                ds = sorted({r["delta"] for r in ss})
                f7, s7, f9, s9, R, cv = ratio(ss, ds)
                lam_tab.append((lbl, k, lm, f7, f9, R, s9, cv[-1], max(ds)))
    if lam_tab:
        w("\n## Lambda robustness (read this before the verdict)\n")
        w("R is not a fixed property of an arm: it depends on how strongly the "
          "ridge head is regularized. A weakly-regularized 9-point fit in a "
          "high-dimensional feature space nearly interpolates, so leaving one "
          "frame out still lets the remaining corrupted labels drag the "
          "prediction — which masks large displacements and flattens the top of "
          "the AUC curve. Stronger penalties resist that. Any verdict below "
          "holds only at the lambda it was computed at.\n")
        w("| arm | level | lambda | floor@0.7 | floor@0.9 | R | AUC at top of grid |")
        w("|---|---|---|---|---|---|---|")
        for lbl, k, lm, f7, f9, R, s9, top, dmax in lam_tab:
            rr = f"{R:.3f}" if R == R else f"— (`{s9}`, so R > {dmax / f7:.2f})"
            w(f"| {lbl} | {k} | {lm:g} | {f7:.2f} | "
              f"{f9:.2f} | {rr} | {top:.3f} |".replace("nan", "—"))
        spread = [x[5] for x in lam_tab if x[5] == x[5]]
        if spread and max(spread) - min(spread) > 0.4:
            w(f"\n**R moves by {max(spread) - min(spread):.2f} across the lambda "
              f"grid ({min(spread):.2f} to {max(spread):.2f}) — more than the "
              f"between-arm difference this script is testing.** Do not report a "
              f"single-lambda contrast as a result about architecture or "
              f"adaptation depth. Run every lambda and report only what holds "
              f"across all of them.")
        w("")

    # ---- comparison ----
    w(f"\n## Comparison (lambda = {a.lam:g} ONLY)\n")
    ug = next((x for x in arms if x["name"].startswith("UniGaze")), None)
    rn = [x for x in arms if x["name"].startswith("ResNet")
          and "full fine-tune" in x["name"]]
    rf = [x for x in arms if x["name"].startswith("ResNet") and "FROZEN" in x["name"]]

    def band(xs):
        """Bootstrap draws for the mean R of a group of levels."""
        n_ = min(len(x["boot"]) for x in xs)
        return [float(np.mean([x["boot"][i] for x in xs])) for i in range(n_)]

    def diff(A, B, label):
        n_ = min(len(A), len(B))
        d = [A[i] - B[i] for i in range(n_)]
        lo = float(np.percentile(d, 2.5)); hi = float(np.percentile(d, 97.5))
        p = 2 * min(np.mean([v <= 0 for v in d]), np.mean([v >= 0 for v in d]))
        sig = lo > 0 or hi < 0
        w(f"- **{label}**: dR = {np.mean(d):+.3f}, 95% CI "
          f"[{lo:+.3f}, {hi:+.3f}], p = {min(p, 1.0):.4f} -> "
          f"{'DIFFERENT' if sig else 'not distinguishable'}")
        return sig, float(np.mean(d))

    # ---- the 2x2, when all three groups are present ----
    if ug and rn and rf:
        w("### The 2x2\n")
        w("| | full fine-tune | frozen + ridge head |")
        w("|---|---|---|")
        w(f"| **ResNet-18** | R = {np.mean(band(rn)):.3f} | "
          f"R = {np.mean(band(rf)):.3f} |")
        w(f"| **UniGaze ViT-H/14** | infeasible (~4700 h/level) | "
          f"R = {np.median(ug['boot']):.3f} |")
        w("")
        w("Two contrasts, each holding one factor fixed:\n")
        s_adapt, d_adapt = diff(
            band(rf), band(rn),
            "ADAPTATION DEPTH, architecture fixed: ResNet frozen - ResNet full")
        s_arch, d_arch = diff(
            ug["boot"], band(rf),
            "ARCHITECTURE, adaptation fixed: UniGaze frozen - ResNet frozen")
        w("")
        if s_arch and not s_adapt:
            w("**Architecture carries it.** Freezing the ResNet did not move the "
              "shape, but swapping the feature extractor did. The confound is "
              "broken in favour of the architecture reading, and the paper can "
              "say the curve shape depends on what you adapt.")
        elif s_adapt and not s_arch:
            w("**Adaptation depth carries it.** The shape moved when the ResNet "
              "was frozen and did not move further when the architecture "
              "changed. The UniGaze arm contributes nothing beyond being "
              "frozen; the paper must state the effect as being about HOW "
              "DEEPLY you adapt, not about architecture. Rewrite any sentence "
              "attributing it to the backbone.")
        elif s_adapt and s_arch:
            w(f"**Both contribute** ({d_adapt:+.3f} from freezing, "
              f"{d_arch:+.3f} from architecture). Report the decomposition and "
              f"claim neither factor alone.")
        else:
            w("**Neither contrast is individually significant.** With 15 "
              "subjects the intervals are too wide to decompose the effect, "
              "even though the pooled UniGaze-vs-full-fine-tune difference "
              "reached p < 0.05. Report the 2x2 as underpowered rather than "
              "assigning the effect to either factor — this is a real outcome, "
              "not a failed experiment, and it is the one that should be "
              "stated if it happens.")
        w("")

    if not ug or not (rn or rf):
        w("Not enough arms with an uncensored R to compare.")
    else:
        rn = rn or rf
        w("| arm | floor@0.7 | floor@0.9 | R | 95% CI |")
        w("|---|---|---|---|---|")
        for x in arms:
            q = lambda p_: float(np.percentile(x["boot"], p_))
            w(f"| {x['name']} | {x['f7']:.2f} | {x['f9']:.2f} | {x['R']:.3f} | "
              f"[{q(2.5):.3f}, {q(97.5):.3f}] |")
        w("")
        n = min(len(ug["boot"]), min(len(x["boot"]) for x in rn))
        for x in rn:
            d = [ug["boot"][i] - x["boot"][i] for i in range(n)]
            lo = float(np.percentile(d, 2.5)); hi = float(np.percentile(d, 97.5))
            pne = 2 * min(np.mean([v <= 0 for v in d]), np.mean([v >= 0 for v in d]))
            w(f"- UniGaze minus {x['name']}: dR = {np.mean(d):+.3f}, 95% CI "
              f"[{lo:+.3f}, {hi:+.3f}], two-sided p = {min(pne, 1.0):.4f} "
              f"-> **{'DIFFERENT shape' if (lo > 0 or hi < 0) else 'not distinguishable'}**")
        # --- pooled ResNet reference: one test, not three ---
        if len(rn) > 1:
            m = [float(np.mean([x["boot"][i] for x in rn])) for i in range(n)]
            d = [ug["boot"][i] - m[i] for i in range(n)]
            lo = float(np.percentile(d, 2.5)); hi = float(np.percentile(d, 97.5))
            p = 2 * min(np.mean([v <= 0 for v in d]), np.mean([v >= 0 for v in d]))
            rlist = ", ".join(f"{x['R']:.2f}" for x in rn)
            w(f"\n**Pooled reference.** The sigma sweep already established that the "
              f"ResNet levels agree with each other, and their R values here "
              f"({rlist}) bear that out. Averaging "
              f"them gives one comparator and one test instead of three, avoiding "
              f"the multiplicity that makes a single p = 0.035 among three "
              f"unconvincing:\n")
            w(f"- R(ResNet, mean of levels) = {np.mean(m):.3f}, 95% CI "
              f"[{np.percentile(m, 2.5):.3f}, {np.percentile(m, 97.5):.3f}]")
            w(f"- R(UniGaze) = {np.median(ug['boot']):.3f}")
            w(f"- **dR = {np.mean(d):+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}], "
              f"two-sided p = {min(p, 1.0):.4f} -> "
              f"{'DIFFERENT shape' if (lo > 0 or hi < 0) else 'not distinguishable'}**")
        w("\nThe two bootstraps are independent, so pairing them index-wise is a "
          "convenience for forming the difference distribution, not a paired "
          "design; the interval is a percentile interval on that difference.")
        w("\nIf the shapes differ: the arms also differ in adaptation depth "
          "(frozen+ridge vs full fine-tune), so this is not yet attributable to "
          "architecture. What it does establish is narrower and safer — that a "
          "single model-relative constant does not transport across both arms, "
          "and that agreement measured at one operating point (0.7) does not "
          "imply agreement at another (0.9). That is a claim about the "
          "GENERALITY of the constant, and it is the honest headline.")

    txt = "\n".join(L) + "\n"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write(txt)
    print(txt)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
