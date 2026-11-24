#!/usr/bin/env python3
# exp_sigma_coherent.py -- the fine-tuned arm under COHERENT corruption.
"""WHAT THIS IS FOR

The coherence result is established on four frozen arms, where every adaptation
is closed-form. The one arm it has not been measured on is full fine-tuning,
which is also the arm where the personalized model is strongest and where the
residual scorers are expected to be hardest to beat. This runs the sigma sweep
protocol unchanged except that corruption is coherent.

COMPARABILITY. The corruption MASK is drawn with exp_sigma_sweep's own keying --
_rng_for(sid, "random_offset", delta, pc, seed) then draw_corruption_mask -- so
the cells line up frame-for-frame with sigma_sweep_records.jsonl, whose
independent-azimuth corruption IS kappa = 0. Only the azimuths differ, and they
come from a kappa-independent stream so a kappa sweep stays paired. A gate
verifies the mask reproduces the recorded is_corrupted before any cell runs; if
it does not, nothing is written.

GRID. The base sweep stops at delta = 10. The floor RISES with coherence on
every frozen arm, so that grid would censor kappa = 1 from above -- the
high-end truncation artifact of Section 5. The grid here extends to 20.

RESUMABLE. Records are appended, never truncated, and cells already present are
skipped. A six-hour job must survive the laptop sleeping.

COST. exp_sigma_sweep took ~45 s/cell (logs/sigma_timings.tsv: 22321 s for 495
cells at L0). One level, one kappa, 10 deltas x 15 subjects x 3 seeds = 450
cells is roughly 5.6 hours.

Needs torch + MPS: run in the agentgaze env on the Mac, not the analysis VM.

Usage:
  python3 exp_sigma_coherent.py --gate
  python3 exp_sigma_coherent.py --level L0 --kappa 1.0
"""
import argparse, hashlib, json, os, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import corruption as corr                  # noqa: E402
import corruption_temporal as ct           # noqa: E402
# loo_residual and backbone_api pull in torch, and so does exp_sigma_sweep at
# module level. They are imported lazily inside main() so that --gate -- the
# check that decides whether a night is worth spending -- runs anywhere numpy
# runs, including the analysis VM.
#
# _dkey and _rng_for below are copied VERBATIM from exp_sigma_sweep rather than
# imported, for the same reason. The copy is not taken on trust: gate() checks
# the masks they produce against the recorded sweep, and nothing is written
# unless they reproduce it exactly.
K = 9


def _dkey(delta):
    if delta is None:
        return "None"
    f = float(delta)
    return str(int(f)) if f.is_integer() else repr(f)


def _rng_for(sid, model, delta, pc, seed):
    key = f"{sid}|{model}|{_dkey(delta)}|{pc}|{seed}".encode()
    return np.random.RandomState(
        int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2 ** 32))

OUT = os.path.join(_HERE, "sigma_coherent_records.jsonl")
REF = os.path.join(_HERE, "sigma_sweep_records.jsonl")
DELTAS = [3, 4, 5, 6, 8, 10, 12, 14, 16, 20]
SUBJ = [f"p{i:02d}" for i in range(15)]


def sweep_mask(sid, delta, pc, seed):
    """The sweep's own mask, bit-for-bit."""
    return corr.draw_corruption_mask(
        K, pc, _rng_for(sid, "random_offset", delta, pc, seed))


def gate(n_cells=60):
    """Prove the mask path reproduces the recorded sweep before spending a night."""
    if not os.path.exists(REF):
        print("GATE: SKIP - no reference records"); return 1
    cells = {}
    for line in open(REF):
        r = json.loads(line)
        if r["model"] != "random_offset":
            continue
        # sigma_level MUST be part of the cell key: the same
        # (subject, seed, delta, pc) appears once per level, and collapsing them
        # compares a 36-row bucket against a 9-frame mask.
        cells.setdefault((r["subject"], r["seed"], r["delta"], r["pc"],
                          r.get("sigma_level")), []).append(r)
    checked = bad = 0
    for (sid, seed, d, pc, _lv), rows in sorted(
            cells.items(), key=lambda kv: tuple(map(str, kv[0])))[:n_cells]:
        rows.sort(key=lambda r: r["frame_idx"])
        got = list(map(bool, sweep_mask(sid, d, pc, seed)))
        want = [bool(r["is_corrupted"]) for r in rows]
        checked += 1
        if got != want:
            bad += 1
            print(f"  MISMATCH {sid} seed={seed} d={d}")
    print(f"\nGATE: cells={checked} mismatched={bad}")
    print("GATE: PASS - masks reproduce the sweep; kappa=0 comparison is paired"
          if bad == 0 else "GATE: FAIL - DO NOT SWEEP")
    return 0 if bad == 0 else 1


def done_cells(path):
    if not os.path.exists(path):
        return set()
    out = set()
    for line in open(path):
        r = json.loads(line)
        out.add((r["sigma_level"], r["subject"], r["seed"], r["delta"],
                 r["pc"], r["kappa"], r.get("lr"), r.get("models_dir")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--level", default="L0")
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--pc", type=float, default=0.2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--deltas", type=float, nargs="+", default=None,
                    help="override the severity grid. At pc=0 every delta gives "
                         "an identical (uncorrupted) cell, so pass a single value "
                         "or the run is 10x longer for no extra information.")
    # These MUST match how sigma_sweep_run.sh invoked exp_sigma_sweep, or the
    # cells are not comparable to sigma_sweep_records.jsonl. That script passes
    # --models-dir models_sigma/<level> and leaves --steps/--lr at their
    # defaults, which resolve to 150 steps and api.DEFAULT_LR (1e-5). An earlier
    # version of this file hardcoded lr=1e-4 and models-dir=None, which silently
    # swept the canonical LOPO backbones at a 10x learning rate and produced
    # 9 hours of results that had to be thrown away.
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=None,
                    help="default: api.DEFAULT_LR, matching exp_sigma_sweep")
    ap.add_argument("--models-dir", default=None,
                    help="default: ../models_sigma/<level>, matching "
                         "sigma_sweep_run.sh. Pass 'canonical' for models/.")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.gate:
        return gate()
    if gate(30) != 0:
        sys.exit("ABORT: seeding gate failed.")

    import loo_residual as lr          # noqa: E402  (torch)
    import backbone_api as api         # noqa: E402  (torch)

    if a.lr is None:
        a.lr = getattr(api, "DEFAULT_LR", 1e-5)
    if a.models_dir is None:
        a.models_dir = os.path.join(os.path.dirname(_HERE), "models_sigma", a.level)
    elif a.models_dir == "canonical":
        a.models_dir = None
    if a.models_dir and not os.path.isdir(a.models_dir):
        sys.exit(f"ABORT: {a.models_dir} not found.")
    print(f"protocol: steps={a.steps} lr={a.lr:g} "
          f"models_dir={a.models_dir or 'models/ (canonical LOPO)'}")
    print("  (must match sigma_sweep_run.sh: steps=150 lr=1e-05 "
          "models_dir=models_sigma/<level>)\n", flush=True)

    have = done_cells(a.out)
    deltas = a.deltas if a.deltas else DELTAS
    if a.pc == 0 and len(deltas) > 1:
        sys.exit("ABORT: pc=0 makes every delta identical. Pass --deltas with a "
                 "single value; running the full grid wastes hours for nothing.")
    tag = (float(a.lr), os.path.basename(a.models_dir or "models"))
    todo = [(sd, s, d) for sd in a.seeds for s in SUBJ for d in deltas
            if (a.level, s, sd, float(d), a.pc, a.kappa) + tag not in have]
    print(f"\nlevel {a.level}  kappa {a.kappa}  pc {a.pc}")
    print(f"{len(todo)} cells to run ({len(have)} already present)")
    print(f"estimated {len(todo)*45/3600:.1f} h at 45 s/cell\n", flush=True)
    if not todo:
        print("nothing to do"); return 0

    subjects = {s: api.load_subject_support(s, K=K, seed=a.seeds[0]) for s in SUBJ}
    t0 = time.time()
    with open(a.out, "a") as fh:
        for n, (seed, sid, d) in enumerate(todo, 1):
            subj = subjects[sid]
            ck = (os.path.join(a.models_dir, f"backbone_mpiifacegaze_{sid}.pt")
                  if a.models_dir else None)
            backbone = api.load_backbone(sid, seed=seed, ckpt_path=ck)
            clean = subj["labels"]
            mask = sweep_mask(sid, d, a.pc, seed)
            r2 = ct.rng_for(sid, "azim", d, a.pc, seed)
            theta = r2.uniform(0.0, 2.0 * np.pi)
            eps = r2.uniform(-np.pi, np.pi, size=K)
            obs, offs = ct.coherent_offset(clean, mask, d, a.kappa, theta, eps)
            src = lr.source_residuals(backbone, subj["images"], obs)
            loo = lr.loo_residuals(backbone, subj["images"], obs,
                                   steps=a.steps, lr=a.lr)
            for i in range(K):
                fh.write(json.dumps({
                    "subject": sid, "seed": int(seed), "model": "coherent_offset",
                    "lr": float(a.lr), "steps": int(a.steps),
                    "models_dir": os.path.basename(a.models_dir or "models"),
                    "kappa": float(a.kappa), "delta": float(d), "pc": float(a.pc),
                    "sigma_level": a.level, "frame_idx": int(i),
                    "is_corrupted": bool(mask[i]),
                    "actual_offset_deg": float(offs[i]),
                    "loo_residual_deg": (None if np.isnan(loo[i]) else float(loo[i])),
                    "source_residual_deg": (None if np.isnan(src[i]) else float(src[i])),
                }) + "\n")
            fh.flush(); os.fsync(fh.fileno())      # survive a sleep or a crash
            el = time.time() - t0
            print(f"  [{n}/{len(todo)}] {sid} seed={seed} d={d}  "
                  f"{el/n:.0f}s/cell  ETA {(len(todo)-n)*el/n/3600:.1f}h", flush=True)
    print(f"\ndone in {(time.time()-t0)/3600:.2f} h -> {os.path.basename(a.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
