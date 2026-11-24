# test_smoke.py  (agentgaze 112px port)
"""
Validates the ported Phase-1 pipeline. Synthetic path needs no data/checkpoint:
    python label_noise/test_smoke.py
If a real backbone (models/backbone_mpiifacegaze_p00.pt) and the MPIIFaceGaze
dataset are present, an extra check runs the real 112px backbone end-to-end.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

import gaze_math as gm
import backbone_api as api
import corruption as corr
import loo_residual as lr
import run_phase1 as rp

_passed = _failed = _skipped = 0
def check(cond, msg):
    global _passed, _failed
    if cond: _passed += 1; print(f"  PASS: {msg}")
    else: _failed += 1; print(f"  FAIL: {msg}")
def skip(msg):
    global _skipped; _skipped += 1; print(f"  SKIP: {msg}")


def test_gaze_math_and_split():
    print("[gaze_math + calibration_split]")
    rng = np.random.RandomState(0)
    maxerr = 0.0
    for _ in range(2000):
        py = gm.random_unit_pitchyaw(rng)
        v = gm.pitchyaw_to_vector(py)
        # exact vec->pitchyaw->vec round-trip (the "no axis bridge" identity)
        maxerr = max(maxerr, float(np.abs(gm.pitchyaw_to_vector(gm.vector_to_pitchyaw(v)) - v).max()))
        for d in (5, 15, 30):
            npy, real = gm.rotate_by_angle(py, d, rng=rng)
            maxerr = max(maxerr, abs(gm.angular_error_deg(py, npy) - d))
    check(maxerr < 1e-6, f"vec round-trip + exact rotation < 1e-6 (max {maxerr:.2e})")
    c9, rest = api.calibration_split(105, 9, seed=42)
    check(len(c9) == 9 and len(rest) == 96 and set(c9).isdisjoint(rest), "K=9 split disjoint")
    check(api.calibration_ids(105, 5, 42) == c9[:5], "K=5 prefix-nested in K=9")


def test_corruption():
    print("[corruption]")
    rng = np.random.RandomState(1)
    labels = np.stack([gm.random_unit_pitchyaw(rng) for _ in range(9)], 0)
    pool = np.stack([gm.random_unit_pitchyaw(rng) for _ in range(105)], 0)
    before = labels.copy()
    o = corr.apply_corruption(labels, "random_offset", np.random.RandomState(2), pc=0.4, delta_deg=15)
    m = o["is_corrupted"]
    check(np.allclose(o["actual_offset_deg"][m], 15.0, atol=1e-6), "random_offset realised == delta")
    check(np.array_equal(labels, before), "input not mutated")
    o2 = corr.apply_corruption(labels, "structured_swap", np.random.RandomState(3), pc=0.4, pool_labels=pool)
    check(np.all(o2["actual_offset_deg"][o2["is_corrupted"]] > 0.5), "structured_swap real offsets recorded")


def test_synthetic_backbone_and_auc():
    print("[synthetic backbone + LOO + AUC]")
    subj = rp.make_synthetic_subjects(n_subjects=1, K=9, seed=0)["S00"]
    bb = api.make_synthetic_backbone()
    v = api.predict(bb, subj["images"])
    check(v.shape == (9, 3) and np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-9), "predict -> unit (9,3)")
    mask = np.zeros(9, bool); mask[[1, 6]] = True
    obs = corr.apply_corruption(subj["support_labels"], "random_offset",
                                np.random.RandomState(0), mask=mask, delta_deg=30)["corrupted_labels"]
    loo = lr.loo_residuals(bb, subj["images"], obs, steps=1, lr=0.0)
    check(np.nanmin(loo[mask]) > np.nanmax(loo[~mask]), "LOO separates corrupted from clean")
    check(lr.roc_auc(loo, mask) == 1.0, "LOO AUC == 1.0 on clean separation")


def test_run_sweep_schema():
    print("[run_phase1 sweep + schema + AUC]")
    subjects = rp.make_synthetic_subjects(n_subjects=3, K=9, seed=0)
    synth = api.make_synthetic_backbone()
    out = os.path.join(_HERE, "_smoke_records.jsonl")
    recs = rp.run_sweep(subjects, lambda seed, sid: synth, seeds=[0],
                        deltas=[10, 30], pcs=[0.1, 0.2], steps=1, lr_=0.0,
                        purify=True, out_path=out)
    req = {"subject", "seed", "delta", "pc", "frame_idx", "is_corrupted",
           "actual_offset_deg", "loo_residual_deg", "source_residual_deg"}
    check(recs and all(req <= set(r) for r in recs), "records carry §5 schema")
    summ = rp.summarize_auc(recs)
    hard = [s for s in summ if s["model"] == "random_offset" and s["delta"] == 30 and s["pc"] == 0.1]
    check(bool(hard) and hard[0]["auc_loo"] >= 0.9, "AUC high at delta=30, pc=0.1")
    if os.path.exists(out): os.remove(out)


def test_real_backbone_optional():
    print("[real 112px backbone (optional)]")
    try:
        import train_backbone as tb
    except Exception as e:
        skip(f"train_backbone deps unavailable ({type(e).__name__}: {e})"); return
    ckpt = os.path.join(os.path.dirname(_HERE), "models", "backbone_mpiifacegaze_p00.pt")
    label = os.path.join(tb.ROOT, "Label", "p00.label")
    if not (os.path.exists(ckpt) and os.path.exists(label)):
        skip("backbone p00 checkpoint or MPIIFaceGaze dataset absent"); return
    s = api.load_subject_support("p00", K=9, seed=42)
    check(s["images"].shape == (9, api.DEFAULT_IMG_SIZE, api.DEFAULT_IMG_SIZE, 3), "support frames (9,112,112,3)")
    check(s["labels"].shape == (9, 2), "support labels (9,2) pitchyaw")
    model = api.load_backbone("p00")
    v = api.predict(model, s["images"])
    check(v.shape == (9, 3) and np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-4), "real predict -> unit (9,3)")
    src = lr.source_residuals(model, s["images"], s["labels"])
    check(np.all(np.isfinite(src)) and src.mean() < 30, f"source residual finite & sane (mean {src.mean():.2f} deg)")
    # one real BN-frozen calibration fold: adapt on 8, predict the held-out 1
    adapted = api.adapt(model, s["images"][1:], s["labels"][1:], steps=api.DEFAULT_STEPS, lr=api.DEFAULT_LR)
    pv = api.predict(adapted, s["images"][:1])
    r = gm.angular_error_vec_deg(pv[0], gm.pitchyaw_to_vector(s["labels"][0]))
    check(np.isfinite(r), f"BN-frozen adapt+predict fold OK (residual {r:.2f} deg)")


def main():
    print(f"convention={api.CONVENTION}  img={api.DEFAULT_IMG_SIZE}  device={api.get_device()}\n")
    test_gaze_math_and_split()
    test_corruption()
    test_synthetic_backbone_and_auc()
    test_run_sweep_schema()
    test_real_backbone_optional()
    print(f"\n{_passed} passed, {_failed} failed, {_skipped} skipped")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
