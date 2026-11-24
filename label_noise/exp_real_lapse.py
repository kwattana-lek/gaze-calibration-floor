#!/usr/bin/env python3
# exp_real_lapse.py -- does the detector find REAL attention lapses, not simulated ones?
"""EXPLORATORY. Declared as such in PREREG_eyediap.md section 4 (commit 86235d4),
before any of this was run. One directional expectation was recorded there:

    post-jump NL frames with SinceJump < 6 and LapseDeg above the measured floor
    should be flagged at a rate no lower than synthetic corruption of matched
    displacement.

THE QUESTION

Every corruption in this project so far is simulated: a clean frame whose label we
moved. EYEDIAP's gaze-state annotation gives the real thing -- frames hand-labelled
"not looking at the target" (NL), paired with the target the system believes was
fixated. Image clean, label wrong, nobody chose the displacement. If the residual
detector only works on labels WE displaced, the paper's deployment claim is hollow.

FOUR KINDS OF REAL FRAME, plus one control
  PJ_EARLY  NL, first 6 frames (200 ms) after the target jumps to a new trajectory.
            Saccade latency: the eyes are still on the previous target. LapseDeg is
            the angle between the two targets from the current face centre -- the
            true label displacement, to within fixation error.
  PJ_LATE   NL, frames 6-11 of the same window. The saccade is under way or done;
            LapseDeg is only an upper bound. Reported separately for that reason.
  MANUAL    NL, >= 15 frames into a trajectory: the annotator saw the participant
            look away. Real distraction; displacement unknown.
  BK        blink. Real IMAGE-channel corruption: the label is nominally right.
  CLEAN_SWAP (control) a fresh OK frame with its correct label. A substituted frame
            is a different image from the one it replaces whatever its label, so
            this is the AUC that "being substituted" earns on its own. It should be
            ~0.5; whatever it is, it is the baseline the other kinds are read against.

THE MATCHED SYNTHETIC TWIN
For every PJ trial the same support set is corrupted a second way: the clean frames
stay, and the same positions have their labels rotated by exactly the natural
frames' LapseDeg (gaze_math.rotate_by_angle, as everywhere else in the project).
Same subject, same positions, same displacement; only real-vs-simulated differs.

Protocol is otherwise the transport arm's: UniGaze h14_cross_X frozen, ridge
lam = 1e-2, K = 9 (the SAME nine clean support frames), LOO residual, 2 of 9 frames
corrupted (p_c = 0.2), draws seeded by SHA-256, bootstrap over subjects.

Stage 1 costs ~45 min of forward passes (cached afterwards); stage 2 is seconds.
Writes only eyediap_lapse_features/, real_lapse_records.jsonl, real_lapse_summary.json.

Usage:  python3 exp_real_lapse.py [--trials 300] [--boot 2000]
"""
import argparse, hashlib, json, os, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
ROOT = os.path.abspath(os.path.join(_HERE, "..", "data", "EYEDIAP_Normalized"))
SUBJ = "e01,e02,e03,e04,e05,e06,e07,e08,e09,e10,e11,e14,e15,e16".split(",")
CLEAN_DIR = os.path.join(_HERE, "eyediap_unigaze_features")
LAPSE_DIR = os.path.join(_HERE, "eyediap_lapse_features")
LAM, NCORR = 1e-2, 2
QUOTA = {"PJ_EARLY": 300, "PJ_LATE": 150, "MANUAL": 150, "BK": 150}
BINS = [(0, 5), (5, 8), (8, 12), (12, 16), (16, 24), (24, 90)]
KINDS = ["PJ_EARLY", "PJ_LATE", "MANUAL", "BK", "CLEAN_SWAP"]


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def rng_for(*key):
    k = "|".join(str(x) for x in key).encode()
    return np.random.RandomState(int.from_bytes(hashlib.sha256(k).digest()[:8], "big") % 2**32)


def kind_of(state, since):
    if state == "BK":
        return "BK"
    if since < 6:
        return "PJ_EARLY"
    if since < 12:
        return "PJ_LATE"
    if since >= 15:
        return "MANUAL"
    return None                                   # frames 12-14: window edge, ambiguous


def select(sid):
    L = open(os.path.join(ROOT, "Label_lapse", f"{sid}.label")).read().strip().split("\n")
    h = L[0].split()
    ix = {c: h.index(c) for c in ("Face", "2DGaze", "State", "SinceJump", "LapseDeg", "Origin")}
    pools = {k: [] for k in QUOTA}
    for ln in L[1:]:
        p = ln.split()
        k = kind_of(p[ix["State"]], int(p[ix["SinceJump"]]))
        lapse = float(p[ix["LapseDeg"]])
        if k is None or (k.startswith("PJ") and not np.isfinite(lapse)):
            continue
        pools[k].append((p[ix["Face"]], [float(v) for v in p[ix["2DGaze"]].split(",")],
                         lapse, int(p[ix["SinceJump"]]), p[ix["Origin"]]))
    out = []
    for k, rows in pools.items():
        pick = rng_for("select", sid, k).permutation(len(rows))[:QUOTA[k]]
        out += [(k,) + rows[i] for i in sorted(pick)]
    return out


def stage1():
    todo = [s for s in SUBJ if not os.path.exists(os.path.join(LAPSE_DIR, f"{s}.npz"))]
    if not todo:
        log("stage 1: all lapse features cached"); return
    import torch, unigaze
    import unigaze_arm as U
    from PIL import Image
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = unigaze.load(U.MODEL, device=str(dev)); model.eval()
    U._dev.append(dev)
    extract = U.build_extractor(model, torch)
    os.makedirs(LAPSE_DIR, exist_ok=True)
    for s in todo:
        t, rows, feats = time.time(), select(s), []
        for i in range(0, len(rows), 8):
            arr = np.stack([np.asarray(Image.open(os.path.join(ROOT, "Image", r[1])).convert("RGB"))
                            for r in rows[i:i + 8]])                  # same input path as the arm
            feats.append(extract(torch.from_numpy(arr).permute(0, 3, 1, 2).float().div(255).to(dev)))
        np.savez_compressed(os.path.join(LAPSE_DIR, f"{s}.npz"),
                            F=np.concatenate(feats).astype(np.float32),
                            Y=np.array([r[2] for r in rows]), kind=np.array([r[0] for r in rows]),
                            lapse=np.array([r[3] for r in rows]), since=np.array([r[4] for r in rows]),
                            origin=np.array([r[5] for r in rows]))
        log(f"  {s}: {len(rows)} lapse frames in {time.time()-t:.0f}s")


def stage2(trials):
    import gaze_math as gm
    from unigaze_arm import K, residuals
    recs = []
    for s in SUBJ:
        zc = np.load(os.path.join(CLEAN_DIR, f"{s}.npz"))
        zl = np.load(os.path.join(LAPSE_DIR, f"{s}.npz"))
        Fs, Ys = zc["F"][:K].astype(np.float64), zc["Y"][:K].astype(np.float64)
        pool = {k: np.where(zl["kind"] == k)[0] for k in QUOTA}
        for kind in KINDS:
            for t in range(trials):
                r = rng_for(s, kind, t)
                pos = r.choice(K, NCORR, replace=False)
                F, Y = Fs.copy(), Ys.copy()
                if kind == "CLEAN_SWAP":
                    src = K + r.choice(len(zc["F"]) - K, NCORR, replace=False)
                    F[pos], Y[pos] = zc["F"][src], zc["Y"][src]
                    lap = np.full(NCORR, np.nan)
                else:
                    if len(pool[kind]) < NCORR:
                        break
                    src = r.choice(pool[kind], NCORR, replace=False)
                    F[pos], Y[pos] = zl["F"][src], zl["Y"][src]
                    lap = zl["lapse"][src]
                _, loo = residuals(F, Y, LAM)
                tw = None
                if kind.startswith("PJ"):                      # matched synthetic twin
                    Yt = Ys.copy()
                    for p, d in zip(pos, lap):
                        Yt[p] = gm.rotate_by_angle(Ys[p], float(d), rng=r)[0]
                    _, tw = residuals(Fs, Yt, LAM)
                for i in range(K):
                    hit = i in pos
                    recs.append({"subject": s, "kind": kind, "trial": t, "pos": i, "is_lapse": bool(hit),
                                 "lapse_deg": float(lap[list(pos).index(i)]) if hit else None,
                                 "loo": float(loo[i]), "loo_twin": None if tw is None else float(tw[i])})
        log(f"  {s} done")
    with open(os.path.join(_HERE, "real_lapse_records.jsonl"), "w") as fh:
        fh.write("\n".join(json.dumps(r) for r in recs) + "\n")
    return recs


def auc_ci(recs, kind, key, lo=None, hi=None, boot=2000):
    """Pooled ROC-AUC of lapse vs clean positions, bootstrap over subjects. With a
    displacement bin, positives are lapses in the bin; negatives are the clean
    positions of the same kind's trials (all of them -- a clean frame has no bin)."""
    from unigaze_arm import roc_auc
    by = {}
    for r in recs:
        if r["kind"] != kind or r[key] is None:
            continue
        if r["is_lapse"] and lo is not None and not (lo <= r["lapse_deg"] < hi):
            continue
        by.setdefault(r["subject"], ([], []))
        by[r["subject"]][0].append(r[key]); by[r["subject"]][1].append(r["is_lapse"])
    subj = sorted(by)
    if not subj:
        return None
    pool = lambda ss: roc_auc(sum((by[s][0] for s in ss), []), sum((by[s][1] for s in ss), []))
    rs = np.random.RandomState(0)
    bs = [pool(list(rs.choice(subj, len(subj)))) for _ in range(boot)]
    bs = [b for b in bs if np.isfinite(b)]
    return {"auc": pool(subj), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "n_lapse": int(sum(sum(by[s][1]) for s in subj)), "boot": bs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    stage1()
    log("stage 2: substitution trials")
    recs = stage2(a.trials)
    out = {"protocol": {"lam": LAM, "K": 9, "n_corrupted": NCORR, "trials": a.trials,
                        "prereg": "PREREG_eyediap.md section 4 (exploratory), commit 86235d4"},
           "overall": {}, "by_displacement": {}}
    print(f"\n{'kind':<11} {'AUC':>6}  {'95% CI':<16} {'n':>6}")
    for k in KINDS:
        r = auc_ci(recs, k, "loo", boot=a.boot)
        out["overall"][k] = {x: r[x] for x in ("auc", "ci95", "n_lapse")}
        print(f"{k:<11} {r['auc']:6.3f}  [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]  {r['n_lapse']:>6}")
    for k in ("PJ_EARLY", "PJ_LATE"):
        print(f"\n{k}: real lapse vs matched synthetic twin, by label displacement")
        print(f"  {'bin (deg)':<10} {'real':>6} {'95% CI':<16} {'synthetic':>9} {'95% CI':<16} {'real-synth':>10} {'95% CI':<17} {'n':>5}")
        out["by_displacement"][k] = {}
        for lo, hi in BINS:
            rn, sy = (auc_ci(recs, k, key, lo, hi, a.boot) for key in ("loo", "loo_twin"))
            if not rn or rn["n_lapse"] < 40:
                continue
            # The two arms share a subject list and a RandomState(0), so replicate i
            # is the SAME resample on both sides and the difference is paired. That
            # holds only while neither side drops a replicate: auc_ci filters
            # non-finite AUCs, and if the two sides dropped different ones, index i
            # would no longer be the same draw and a silent truncation would pair
            # unrelated replicates. Verified not to occur here (every subject carries
            # both classes in every bin, so no resample is degenerate), but the cost
            # of being wrong is a wrong interval, so it is asserted rather than cut.
            if len(rn["boot"]) != len(sy["boot"]):
                sys.exit(f"ABORT [{k} {lo}-{hi}]: {len(rn['boot'])} real vs "
                         f"{len(sy['boot'])} synthetic replicates survived. Index "
                         f"pairing is no longer valid; pair on the draw, not the index.")
            d = np.array(rn["boot"]) - np.array(sy["boot"])
            dci = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
            out["by_displacement"][k][f"{lo}-{hi}"] = {
                "real": rn["auc"], "real_ci95": rn["ci95"], "synthetic": sy["auc"],
                "synthetic_ci95": sy["ci95"], "diff": rn["auc"] - sy["auc"], "diff_ci95": dci,
                "n_lapse": rn["n_lapse"]}
            print(f"  {f'{lo}-{hi}':<10} {rn['auc']:6.3f} [{rn['ci95'][0]:.3f}, {rn['ci95'][1]:.3f}]  "
                  f"{sy['auc']:9.3f} [{sy['ci95'][0]:.3f}, {sy['ci95'][1]:.3f}]  "
                  f"{rn['auc']-sy['auc']:+10.3f} [{dci[0]:+.3f}, {dci[1]:+.3f}] {rn['n_lapse']:>5}")
    json.dump(out, open(os.path.join(_HERE, "real_lapse_summary.json"), "w"), indent=2)
    print("\nwrote real_lapse_records.jsonl, real_lapse_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
