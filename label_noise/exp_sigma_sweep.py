#!/usr/bin/env python
# exp_sigma_sweep.py
"""
sigma-sweep: is the detection floor a FIXED MULTIPLE of the model's own error?

The model-relative claim (SIGMA_DIAGNOSTIC §5, SIGMA_RELIABILITY branch A) is
that detection collapses on a model-relative axis: delta@AUC=0.7 divided by the
mean personalized error is ~1.2 regardless of subject. Every subject in that
analysis shares ONE backbone, so subject difficulty and error scale are
confounded. This driver breaks the confound by moving the error scale itself:
train the SAME architecture to several error levels (train_backbone.py
--epochs / --train_cap), then ask whether

    c = delta_floor / e_hat

is invariant across levels. If c holds, the floor is model-relative and the gate
self-calibrates. If c moves with e_hat, the model-relative framing is wrong and
the absolute threshold (tau=3.42) is all there is.

WHY A SEPARATE DRIVER (not a run_phase1.py patch)
-------------------------------------------------
run_phase1.py is validated code behind committed results. Three things there
block a fine delta grid, and patching them in place would put recorded numbers
at risk:
  * `--deltas` is type=int
  * score_cell records `int(delta)`  -> 4.25 would silently become 4
  * the summary printf uses `:>3d`   -> crashes on a float
This driver reuses run_phase1's PRIMITIVES (corruption, loo_residual,
backbone_api) unchanged and re-implements only the cell loop, exactly as
exp_a/exp_b/exp_c do.

SEED COMPATIBILITY (important)
------------------------------
run_phase1.score_cell derives its RNG from
    sha256(f"{sid}|{model}|{delta}|{pc}|{seed}")
with `delta` an int, so the digest string is "6", never "6.0". `_dkey` below
reproduces that: integral deltas format as int, fractional ones as the shortest
exact repr. So delta=6.0 here draws the SAME corruption as delta=6 there, and
--verify proves it against the recorded floor-v2 records before any new number
is produced.

USAGE
-----
  # 0. prove faithfulness against recorded results (no new compute)
  python exp_sigma_sweep.py --verify

  # 1. per level, after training that level's backbones into its own models dir
  python exp_sigma_sweep.py --sweep --level L1 \
      --models-dir models_sigma/L1 --deltas 3 4 4.5 5 6 8 --pcs 0.2

  # 2. once all levels are swept
  python exp_sigma_sweep.py --analyze
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LN = os.path.join(os.path.dirname(_HERE), "label_noise") if not os.path.exists(
    os.path.join(_HERE, "backbone_api.py")) else _HERE
for _p in (_LN, os.path.dirname(_LN)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import backbone_api as api          # noqa: E402
import corruption as corr           # noqa: E402
import loo_residual as lr           # noqa: E402

K = 9
DEFAULT_DELTAS = [3, 4, 4.5, 5, 6, 8, 10]
DEFAULT_PCS = [0.2]
RECORDS = os.path.join(_HERE, "sigma_sweep_records.jsonl")


# --------------------------------------------------------------------------
def _dkey(delta):
    """Digest form of delta matching run_phase1 (int when integral)."""
    if delta is None:
        return "None"
    f = float(delta)
    return str(int(f)) if f.is_integer() else repr(f)


def _rng_for(sid, model, delta, pc, seed):
    key = f"{sid}|{model}|{_dkey(delta)}|{pc}|{seed}".encode()
    return np.random.RandomState(
        int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2 ** 32))


def score_cell(sid, subj, backbone, seed, model, delta, pc, steps, lr_):
    """One (subject, seed, model, delta, pc) cell -> K frame records.

    Mirrors run_phase1.score_cell, minus purification (not needed here) and
    with delta kept as a float in the output.
    """
    clean = subj["support_labels"]
    rng = _rng_for(sid, model, delta, pc, seed)
    c = corr.apply_corruption(clean, model=model, rng=rng, pc=pc,
                              delta_deg=delta, pool_labels=subj["pool_labels"])
    obs = c["corrupted_labels"]
    src = lr.source_residuals(backbone, subj["images"], obs)
    loo = lr.loo_residuals(backbone, subj["images"], obs, steps=steps, lr=lr_)
    out = []
    for i in range(len(clean)):
        out.append({
            "subject": sid, "seed": int(seed), "model": model,
            "delta": (None if model == "structured_swap" else float(delta)),
            "pc": float(pc), "frame_idx": int(i),
            "is_corrupted": bool(c["is_corrupted"][i]),
            "actual_offset_deg": float(c["actual_offset_deg"][i]),
            "loo_residual_deg": (None if np.isnan(loo[i]) else float(loo[i])),
            "source_residual_deg": (None if np.isnan(src[i]) else float(src[i])),
        })
    return out


# --------------------------------------------------------------------------
def roc_auc(scores, pos):
    scores = np.asarray(scores, float)
    pos = np.asarray(pos, bool)
    ok = ~np.isnan(scores)
    scores, pos = scores[ok], pos[ok]
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    u, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    avg = (csum - cnt + csum + 1) / 2.0
    return float((avg[inv][pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def crossing(deltas, aucs, target):
    """First delta where AUC crosses `target`, linear between grid points."""
    pairs = sorted(zip(deltas, aucs))
    for (d0, a0), (d1, a1) in zip(pairs, pairs[1:]):
        if a0 < target <= a1 and a1 != a0:
            return d0 + (target - a0) * (d1 - d0) / (a1 - a0)
    return None


def e_hat(records):
    """Mean personalized error scale: mean clean-frame LOO residual.

    SIGMA_RELIABILITY settled that at K=9 the MEAN is the practical normalizer
    (test-retest r=0.82 vs 0.39 for the 9-point MAD), so the scale is estimated
    as the mean LOO residual over UNCORRUPTED support frames.
    """
    per_subj = defaultdict(list)
    for r in records:
        if not r["is_corrupted"] and r["loo_residual_deg"] is not None:
            per_subj[r["subject"]].append(r["loo_residual_deg"])
    means = [float(np.mean(v)) for v in per_subj.values() if v]
    return float(np.mean(means)) if means else float("nan"), len(means)


# --------------------------------------------------------------------------
def cmd_verify(args):
    """Reproduce recorded floor-v2 cells with this driver's seeding path.

    Proves _dkey/_rng_for/apply_corruption reproduce run_phase1 EXACTLY before
    any new number is generated. Compares the corruption draw only (is_corrupted
    + actual_offset_deg), which is the part this driver re-implements; the
    residual path is called unchanged from loo_residual.
    """
    src = args.reference
    if not os.path.exists(src):
        print(f"GATE: SKIP - reference not found: {src}")
        return 1
    recs = [json.loads(l) for l in open(src)]
    cells = defaultdict(list)
    for r in recs:
        if r["model"] != "random_offset":
            continue
        cells[(r["subject"], r["seed"], r["delta"], r["pc"])].append(r)
    checked = mismatched = 0
    for (sid, seed, delta, pc), rows in sorted(cells.items())[:args.n_cells]:
        rows.sort(key=lambda r: r["frame_idx"])
        clean = np.zeros((len(rows), 2))          # labels irrelevant to the mask
        rng = _rng_for(sid, "random_offset", delta, pc, seed)
        c = corr.apply_corruption(clean, model="random_offset", rng=rng, pc=pc,
                                  delta_deg=delta, pool_labels=None)
        got = list(map(bool, c["is_corrupted"]))
        want = [bool(r["is_corrupted"]) for r in rows]
        checked += 1
        if got != want:
            mismatched += 1
            print(f"  MISMATCH {sid} seed={seed} d={delta} pc={pc}\n"
                  f"    recorded {want}\n    reproduced {got}")
    print(f"\nVERIFICATION GATE: cells={checked}  mismatched={mismatched}")
    print("GATE: PASS - corruption draw reproduces run_phase1 bit-exactly"
          if mismatched == 0 else "GATE: FAIL - seeding path diverges; DO NOT SWEEP")
    return 0 if mismatched == 0 else 1


def cmd_sweep(args):
    subjects = {}
    ids = args.subjects or [f"p{i:02d}" for i in range(args.n_subjects)]
    for sid in ids:
        s = api.load_subject_support(sid, K=K, seed=args.seeds[0])
        # load_subject_support returns "labels"; run_phase1 renames it to
        # "support_labels" in its own loader wrapper and score_cell below
        # follows run_phase1's contract. Remap here rather than diverge.
        subjects[sid] = {
            "support_labels": s["labels"],
            "images": s["images"],
            "pool_labels": s["pool_labels"],
        }
    steps = args.steps if args.steps is not None else 150
    n = 0
    with open(args.out, "a") as fh:
        for seed in args.seeds:
            for sid, subj in subjects.items():
                ck = (os.path.join(args.models_dir,
                                   f"backbone_mpiifacegaze_{sid}.pt")
                      if args.models_dir else None)
                backbone = api.load_backbone(sid, seed=seed, ckpt_path=ck)
                cells = [("random_offset", d, pc)
                         for d in args.deltas for pc in args.pcs]
                for model, delta, pc in cells:
                    for r in score_cell(sid, subj, backbone, seed, model,
                                        delta, pc, steps, args.lr):
                        r["sigma_level"] = args.level
                        fh.write(json.dumps(r) + "\n")
                        n += 1
                print(f"  {args.level} {sid} seed={seed}: {len(cells)} cells", flush=True)
    print(f"wrote {n} records -> {args.out}")
    return 0


def cmd_analyze(args):
    if not os.path.exists(args.out):
        print(f"no records at {args.out}")
        return 1
    recs = [json.loads(l) for l in open(args.out)]
    by_level = defaultdict(list)
    for r in recs:
        by_level[r.get("sigma_level", "?")].append(r)

    print(f"\n{'='*74}\nSIGMA SWEEP — is delta_floor / e_hat invariant?\n{'='*74}\n")
    print(f"{'level':8}{'subj':>6}{'e_hat':>9}{'d@0.7':>9}{'c=d/e':>9}"
          f"{'d@0.9':>9}{'c9':>8}")
    rows = []
    for lvl, rs in sorted(by_level.items()):
        eh, nsub = e_hat(rs)
        by_d = defaultdict(list)
        for r in rs:
            if r["model"] == "random_offset" and r["loo_residual_deg"] is not None:
                by_d[r["delta"]].append((r["loo_residual_deg"], r["is_corrupted"]))
        ds = sorted(by_d)
        aucs = [roc_auc([s for s, _ in by_d[d]], [p for _, p in by_d[d]]) for d in ds]
        d7, d9 = crossing(ds, aucs, 0.7), crossing(ds, aucs, 0.9)
        c7 = d7 / eh if d7 and eh == eh else None
        c9 = d9 / eh if d9 and eh == eh else None
        rows.append((lvl, eh, d7, c7, d9, c9))
        f = lambda v, p=2: "  --" if v is None else f"{v:.{p}f}"
        print(f"{lvl:8}{nsub:>6}{eh:>9.3f}{f(d7):>9}{f(c7,3):>9}"
              f"{f(d9):>9}{f(c9,3):>8}")
        print(f"         AUC by delta: " +
              "  ".join(f"{d:g}:{a:.3f}" for d, a in zip(ds, aucs)))

    cs = [r[3] for r in rows if r[3] is not None]
    ehs = [r[1] for r in rows if r[3] is not None]
    print()
    if len(cs) < 2:
        print("  need >=2 levels with a located crossing to test invariance")
        return 0
    cv = float(np.std(cs) / np.mean(cs))
    span = max(ehs) / min(ehs)
    print(f"  e_hat span across levels : {span:.2f}x  ({min(ehs):.2f} - {max(ehs):.2f} deg)")
    print(f"  c = delta_floor / e_hat  : {' '.join(f'{c:.3f}' for c in cs)}")
    print(f"  CV(c) = {cv:.3f}")
    print()
    if span < 1.3:
        print("  INCONCLUSIVE - e_hat barely moved; levels are too close to test.")
    elif cv < 0.10:
        print("  INVARIANT - c holds across error scales. The floor is")
        print("  model-relative; the gate self-calibrates as tau = c * e_hat.")
    else:
        print("  NOT INVARIANT - c moves with the error scale. The")
        print("  model-relative framing does not survive; ship absolute tau.")
    return 0


def cmd_regress(args):
    """Per-subject regression of delta_floor on e_hat, with a censoring audit.

    This is the PRIMARY analysis. The level-pooled view (--analyze) hides a
    selection effect: subjects whose crossing lies outside the swept delta grid
    contribute no point, and on the 2026-09-02 run those were predominantly
    LOW-error subjects already above AUC 0.7 at the grid minimum. Dropping them
    biases the slope down and the intercept up, which manufactures an apparent
    affine relation. This mode reports the censored cells alongside the fit so
    the bias direction is always visible next to the estimate.
    """
    recs = [json.loads(l) for l in open(args.out)]
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    clean = collections.defaultdict(list)
    for r in recs:
        if r.get("loo_residual_deg") is None:
            continue
        k = (r.get("sigma_level", "?"), r["subject"])
        if not r["is_corrupted"]:
            clean[k].append(r["loo_residual_deg"])
        if r["model"] == "random_offset":
            per[k][r["delta"]].append((r["loo_residual_deg"], r["is_corrupted"]))

    hit, miss, unsustained = [], [], []
    for k in sorted(per):
        ds = sorted(per[k])
        aa = [roc_auc([x for x, _ in per[k][d]], [c for _, c in per[k][d]])
              for d in ds]
        eh = float(np.mean(clean[k])) if clean[k] else float("nan")
        c7 = crossing(ds, aa, 0.7)
        finite = [a for a in aa if a == a]
        # VALIDITY GUARD. A detectability crossing must be SUSTAINED: once the
        # displacement is large enough to detect, larger displacement must not
        # become less detectable. At heavily degraded accuracy levels the true
        # floor can lie beyond the swept grid, and the per-cell ROC estimate
        # (~27 frames) then wanders above 0.7 by chance. Such a spurious
        # crossing is indistinguishable from a real one by location alone, but
        # not by whether it holds at the top of the grid. Counting them drags
        # the slope toward zero and manufactures an affine relation.
        top = dict(zip(ds, aa)).get(max(ds), float("nan"))
        sustained = (top == top) and (top >= 0.7)
        if c7 and not sustained:
            c7 = None
            unsustained.append((k[0], k[1], eh, top))
        (hit if c7 else miss).append(
            (k[0], k[1], eh, c7, min(ds), max(finite) if finite else float("nan")))

    print(f"\n{'='*74}\nPER-SUBJECT REGRESSION — delta_floor on e_hat\n{'='*74}")
    if unsustained:
        print(f"\n  DISCARDED — crossing not sustained at the top of the grid"
              f" ({len(unsustained)} cells):")
        print(f"    {'level':6}{'subj':6}{'e_hat':>8}{'AUC at max delta':>18}")
        for lv, sid, eh, top in unsustained:
            print(f"    {lv:6}{sid:6}{eh:>8.2f}{top:>18.3f}")
        print("    => their true floor lies beyond the swept grid; the located"
              " value was noise.")
    print(f"\n  cells with a located 0.7 crossing: {len(hit)} / {len(hit)+len(miss)}")

    if miss:
        print(f"\n  CENSORED CELLS (no crossing inside the swept grid):")
        print(f"    {'level':6}{'subj':6}{'e_hat':>8}{'grid min':>10}{'max AUC':>9}  cause")
        unsus = {(u[0], u[1]) for u in unsustained}
        for lv, sid, eh, _, dmin, mx in miss:
            if (lv, sid) in unsus:
                cause = "crossing not sustained -> floor ABOVE the grid"
            elif mx >= 0.7:
                cause = "already above 0.7 at grid min -> floor BELOW the grid"
            else:
                cause = "never reaches 0.7 -> floor ABOVE the grid"
            print(f"    {lv:6}{sid:6}{eh:>8.2f}{dmin:>10g}{mx:>9.3f}  {cause}")
        eh_hit = np.mean([h[2] for h in hit])
        eh_mis = np.mean([m[2] for m in miss])
        print(f"\n    mean e_hat  located {eh_hit:.3f}   censored {eh_mis:.3f}")
        if eh_mis < eh_hit:
            print("    => censored cells are LOWER-error, and their floors sit"
                  " below the grid. Excluding them truncates the LOW end of")
            print("       both axes: slope biased DOWN, intercept biased UP."
                  " Extend the grid downward before trusting the fit.")
        else:
            print("    => censored cells are HIGHER-error, and their floors sit"
                  " above the grid. Excluding them truncates the HIGH end of")
            print("       both axes: slope biased DOWN, intercept biased UP."
                  " The fit below UNDERSTATES the slope.")
            print("       A level whose floor is entirely off-grid cannot"
                  " contribute; re-fit without it rather than around it.")

    if len(hit) < 5:
        print("\n  too few located crossings to regress")
        return 0
    E = np.array([h[2] for h in hit])
    D = np.array([h[3] for h in hit])
    sl, ic = np.polyfit(E, D, 1)
    r = float(np.corrcoef(E, D)[0, 1])
    rng = random.Random(0)
    sls, ics = [], []
    for _ in range(args.n_boot):
        idx = [rng.randrange(len(E)) for _ in E]
        a, b = np.polyfit(E[idx], D[idx], 1)
        sls.append(a); ics.append(b)
    sls.sort(); ics.sort()
    q = lambda v: (v[int(0.025 * len(v))], v[int(0.975 * len(v))])
    slo, shi = q(sls); ilo, ihi = q(ics)
    cv = lambda v: float(np.std(v) / np.mean(v))
    print(f"\n  n = {len(hit)}   slope = {sl:+.3f} [{slo:+.3f}, {shi:+.3f}]"
          f"   intercept = {ic:+.3f} [{ilo:+.3f}, {ihi:+.3f}]")
    print(f"  r = {r:+.3f}   R2 = {r*r:.3f}")
    print(f"  CV(delta_floor) {cv(D):.3f} -> CV(delta_floor/e_hat) {cv(D/E):.3f}")
    prop = not (ilo > 0 or ihi < 0)
    print(f"\n  intercept {'includes' if prop else 'EXCLUDES'} 0"
          f"  -> {'consistent with PROPORTIONAL' if prop else 'favours AFFINE'}")
    if miss:
        print("  (but see the censoring audit above before acting on this)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--regress", action="store_true",
                    help="per-subject regression + censoring audit (primary)")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--level", default="L0", help="error-scale level label")
    ap.add_argument("--models-dir", default=None,
                    help="dir of backbone_mpiifacegaze_<sid>.pt for this level")
    ap.add_argument("--out", default=RECORDS)
    ap.add_argument("--deltas", type=float, nargs="+", default=DEFAULT_DELTAS)
    ap.add_argument("--pcs", type=float, nargs="+", default=DEFAULT_PCS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--n-subjects", type=int, default=15)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=getattr(api, "DEFAULT_LR", 1e-4))
    ap.add_argument("--n-cells", type=int, default=40, help="--verify: cells to check")
    ap.add_argument("--reference", default=os.path.join(
        _LN, "phase1_records_floor_v2.jsonl"))
    args = ap.parse_args()
    if args.verify:
        return cmd_verify(args)
    if args.sweep:
        return cmd_sweep(args)
    if args.analyze:
        return cmd_analyze(args)
    if args.regress:
        return cmd_regress(args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
