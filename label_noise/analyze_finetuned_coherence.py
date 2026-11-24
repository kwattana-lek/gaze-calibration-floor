#!/usr/bin/env python3
# analyze_finetuned_coherence.py -- the fine-tuned arm's c(kappa), done properly.
"""WHY THE OBVIOUS ANALYSIS IS WRONG

The sweep records carry an e_hat measured as the mean clean-frame LOO residual
INSIDE CORRUPTED CELLS. Under coherent corruption a full fine-tune absorbs the
displacement by bending the whole mapping, which degrades the clean frames too,
so that e_hat rises with kappa -- 3.43 -> 4.82 deg on L0. Divide the floor by it
and c looks invariant (1.120 -> 1.129) purely because numerator and denominator
moved together.

The frozen-arm table uses CLEAN-FIT e_hat: a fit on true labels with no
corruption anywhere, which cannot depend on kappa by construction. For the two
to be comparable the fine-tuned arm needs the same denominator, which is what
the pc=0 pass measures.

This script reports both so the estimator's contribution is visible rather than
hidden, in the spirit of the three-definition table in the earlier manuscript.

Inputs (all produced on the Mac):
  sigma_sweep_records.jsonl        kappa = 0   (independent azimuths)
  sigma_coherent_records.jsonl     kappa = 1
  sigma_clean_ehat_records.jsonl   pc = 0      -> clean-fit e_hat

Needs no torch.
"""
import json, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import exp_coherence as E              # noqa: E402
from unigaze_arm import roc_auc        # noqa: E402

LEVEL = "L0"


def load(fn, **f):
    if not os.path.exists(os.path.join(_HERE, fn)):
        return None
    rs = [json.loads(l) for l in open(os.path.join(_HERE, fn))]
    for k, v in f.items():
        rs = [r for r in rs if r.get(k) == v]
    return rs


def floor(rs, field):
    ds = sorted({r["delta"] for r in rs if r["delta"] > 0})
    au = []
    for d in ds:
        c = [r for r in rs if r["delta"] == d and r[field] is not None]
        au.append(roc_auc([r[field] for r in c], [r["is_corrupted"] for r in c]))
    return E.crossing(ds, au, 0.7)


def per_subject_mean(rs, clean_only=True):
    per = {}
    for r in rs:
        if clean_only and r["is_corrupted"]:
            continue
        if r["loo_residual_deg"] is not None:
            per.setdefault(r["subject"], []).append(r["loo_residual_deg"])
    return {s: float(np.mean(v)) for s, v in per.items()}


def main():
    k0 = load("sigma_sweep_records.jsonl", sigma_level=LEVEL)
    k1 = load("sigma_coherent_records.jsonl", sigma_level=LEVEL)
    cl = load("sigma_clean_ehat_records.jsonl", sigma_level=LEVEL)
    if k0 is None or k1 is None:
        sys.exit("ABORT: sweep records missing.")
    if cl is None:
        sys.exit("ABORT: sigma_clean_ehat_records.jsonl not found -- run\n"
                 "'Run Clean Ehat (fine-tuned).command' on the Mac first (~51 min).")

    # clean-fit e_hat, with the MPS training-drift spread across seeds
    by_seed = {}
    for sd in sorted({r["seed"] for r in cl}):
        g = [r for r in cl if r["seed"] == sd]
        by_seed[sd] = float(np.mean(list(per_subject_mean(g, False).values())))
    e_clean = float(np.mean(list(by_seed.values())))
    spread = max(by_seed.values()) - min(by_seed.values())
    print(f"Clean-fit e_hat (pc=0, no corruption anywhere): {e_clean:.2f} deg")
    print(f"  per seed: " + ", ".join(f"{s}:{v:.2f}" for s, v in by_seed.items())
          + f"   spread {spread:.2f} deg")
    print("  (MPS training is nondeterministic; that spread bounds every c below)\n")

    e_cont = {"kappa=0": float(np.mean(list(per_subject_mean(k0).values()))),
              "kappa=1": float(np.mean(list(per_subject_mean(k1).values())))}
    fl = {"kappa=0": floor(k0, "loo_residual_deg")[0],
          "kappa=1": floor(k1, "loo_residual_deg")[0]}

    print(f"{'':>9} {'floor':>8} {'e_cont':>8} {'c (cont)':>10} "
          f"{'e_clean':>9} {'c (clean)':>11}")
    print("-" * 60)
    for nm in ("kappa=0", "kappa=1"):
        f_, ec = fl[nm], e_cont[nm]
        print(f"{nm:>9} {f_:>8.2f} {ec:>8.2f} {f_/ec:>10.3f} "
              f"{e_clean:>9.2f} {f_/e_clean:>11.3f}")
    r_cont = fl["kappa=1"] / e_cont["kappa=1"] / (fl["kappa=0"] / e_cont["kappa=0"])
    r_clean = fl["kappa=1"] / fl["kappa=0"]
    print(f"\nc(1)/c(0):  contaminated {r_cont:.2f}x   clean-fit {r_clean:.2f}x")
    print("The frozen arms gave 1.48x on clean-fit e_hat. Only the clean-fit")
    print("column is comparable to that; the contaminated column hides the")
    print("effect by moving its own denominator.")

    # the unexplained inversion, flagged rather than tabulated
    print("\nResubstitution floors (UNEXPLAINED -- do not tabulate yet):")
    for nm, rs in (("kappa=0", k0), ("kappa=1", k1)):
        v = floor(rs, "source_residual_deg")
        print(f"  {nm}: {v[0]:.2f}" if v[0] else f"  {nm}: {v[1]}")

    json.dump({"level": LEVEL, "e_clean": e_clean, "e_clean_by_seed": by_seed,
               "e_clean_spread": spread, "e_contaminated": e_cont,
               "floor_loo": fl, "c_clean": {k: fl[k] / e_clean for k in fl},
               "c_contaminated": {k: fl[k] / e_cont[k] for k in fl}},
              open(os.path.join(_HERE, "finetuned_coherence_summary.json"), "w"),
              indent=2)
    print("\nwrote finetuned_coherence_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
