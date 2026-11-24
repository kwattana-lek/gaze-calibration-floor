"""Step 3 runner: gate transfer across L0-L3 (see ../PREREG_step3_gate_transfer.md).

pass `score`: per (level, subject): clean-support stats; per (seed, pc): structured donor
swap -> resub + LOO residuals for the 9 frames. Append-only JSONL, resume-safe.
pass `mae`:   given a thresholds JSON {level: {policy: tau or {pc: tau}}}, re-adapt with flagged
frames replaced by clean substitutes and evaluate held-out MAE. Append-only, resume-safe.

Never writes outside label_noise/revision_2026-11/. Reads models_sigma/ read-only.
Run from repo root with PYTHONPATH=.:scripts PYTHONHASHSEED=0.
"""
import argparse, hashlib, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REV = os.path.dirname(HERE)
LN = os.path.dirname(REV); AG = os.path.dirname(LN)
for p in (LN, os.path.join(AG, "scripts"), AG):
    if p not in sys.path: sys.path.insert(0, p)
import backbone_api as api, corruption as corr, loo_residual as lr, exp_a_scorers as sc, gaze_math as gm
import train_backbone as tb

LEVELS = ["L0", "L1", "L2", "L3"]; SUBJ = [f"p{i:02d}" for i in range(15)]
SEEDS = list(range(10)); PCS = [0.1, 0.2, 0.4]; K = 9; EVAL_N = 1000
PREREG = os.path.join(REV, "PREREG_step3_gate_transfer.md")
HASHF = os.path.join(REV, "logs", "prereg_hash.txt")


def check_prereg():
    h = hashlib.sha256(open(PREREG, "rb").read()).hexdigest()
    if not os.path.exists(HASHF):
        open(HASHF, "w").write(f"{h}  PREREG_step3_gate_transfer.md  recorded {time.strftime('%Y-%m-%d %H:%M:%S %z')}\n")
    elif open(HASHF).read().split()[0] != h:
        sys.exit("pre-registration file changed since its hash was recorded; refusing to run")
    return h


def rng_key(*parts):
    k = "|".join(str(p) for p in parts).encode()
    return np.random.RandomState(int.from_bytes(hashlib.sha256(k).digest()[:8], "big") % (2 ** 32))


class Subj:
    def __init__(self, s):
        self.s = s
        imgs, vecs = tb.load_images(s, cap=None, rng=np.random.default_rng(42))
        n = len(imgs); sup = np.asarray(api.calibration_split(n, K, seed=42)[0])
        ref = api.load_subject_support(s, K=K, seed=42)          # guard: identical to the loader the paper used
        assert np.array_equal(sup, np.asarray(ref["support_ids"])), "support split mismatch"
        self.sup_imgs = imgs[sup]; self.sup_lab = gm.vector_to_pitchyaw(vecs[sup]).reshape(-1, 2)
        assert np.allclose(self.sup_lab, ref["labels"]), "support labels mismatch"
        self.pool_lab = gm.vector_to_pitchyaw(vecs).reshape(-1, 2)
        rest = np.setdiff1d(np.arange(n), sup)
        ev = np.sort(np.random.RandomState(1000).choice(rest, min(EVAL_N, len(rest)), replace=False))
        self.ev_imgs = imgs[ev]; self.ev_vec = gm.pitchyaw_to_vector(gm.vector_to_pitchyaw(vecs[ev]).reshape(-1, 2))
        self.sub_idx = np.setdiff1d(rest, ev)                        # substitutes never enter the eval set
        self.imgs, self.vecs = imgs, vecs

    def mae(self, model):
        out = [gm.angular_error_vec_deg(api.predict(model, self.ev_imgs[i:i + 500]), self.ev_vec[i:i + 500])
               for i in range(0, len(self.ev_imgs), 500)]
        return float(np.concatenate(out).mean())

    def corrupt(self, pc, seed):
        c = corr.apply_corruption(self.sup_lab, model="structured_swap", rng=rng_key(self.s, "structured_swap", None, pc, seed),
                                  pc=pc, delta_deg=None, pool_labels=self.pool_lab)
        return np.asarray(c["corrupted_labels"]), np.asarray(c["is_corrupted"], bool), np.asarray(c.get("actual_offset_deg", [np.nan] * K), float)


def load_bb(level, s):
    return api.load_backbone(s, ckpt_path=os.path.join(AG, "models_sigma", level, f"backbone_mpiifacegaze_{s}.pt"))


def done_keys(path, fields):
    if not os.path.exists(path): return set()
    return {tuple(json.loads(l)[f] for f in fields) for l in open(path) if l.strip()}


def cmd_score(a):
    h = check_prereg(); out = os.path.join(REV, "results", "gate_transfer_scores.jsonl")
    done = done_keys(out, ["level", "subject", "kind", "seed", "pc"])
    t0 = time.time()
    with open(out, "a") as f:
        for s in a.subjects:
            S = None
            for L in a.levels:
                if all((L, s, "cell", sd, pc) in done for sd in SEEDS for pc in PCS) and (L, s, "clean", -1, 0.0) in done:
                    continue
                S = S or Subj(s); bb = load_bb(L, s)
                if (L, s, "clean", -1, 0.0) not in done:
                    loo = lr.loo_residuals(bb, S.sup_imgs, S.sup_lab); res = sc.small_loss_scores(bb, S.sup_imgs, S.sup_lab)
                    m_ad = api.adapt(bb, S.sup_imgs, S.sup_lab)
                    f.write(json.dumps({"level": L, "subject": s, "kind": "clean", "seed": -1, "pc": 0.0,
                                        "loo": [float(x) for x in loo], "resub": [float(x) for x in res],
                                        "heldout_mae": S.mae(m_ad), "cross_mae": S.mae(bb), "prereg_sha": h}) + "\n"); f.flush()
                for sd in SEEDS:
                    for pc in PCS:
                        if (L, s, "cell", sd, pc) in done: continue
                        obs, mask, off = S.corrupt(pc, sd)
                        res = sc.small_loss_scores(bb, S.sup_imgs, obs); loo = lr.loo_residuals(bb, S.sup_imgs, obs)
                        f.write(json.dumps({"level": L, "subject": s, "kind": "cell", "seed": sd, "pc": pc,
                                            "is_corrupted": mask.tolist(), "offset": [None if np.isnan(x) else float(x) for x in off],
                                            "resub": [float(x) for x in res], "loo": [float(x) for x in loo]}) + "\n")
                    f.flush()
                print(f"[score] {L} {s} done  t={time.time()-t0:.0f}s", flush=True)


def cmd_mae(a):
    check_prereg(); thr = json.load(open(a.thresholds))
    scores = [json.loads(l) for l in open(os.path.join(REV, "results", "gate_transfer_scores.jsonl"))]
    cells = {(r["level"], r["subject"], r["seed"], r["pc"]): r for r in scores if r["kind"] == "cell"}
    out = os.path.join(REV, "results", "gate_transfer_mae.jsonl")
    done = done_keys(out, ["level", "subject", "seed", "pc"]); t0 = time.time()
    with open(out, "a") as f:
        for s in a.subjects:
            S = None
            for L in a.levels:
                todo = [(sd, pc) for sd in a.seeds for pc in PCS if (L, s, sd, pc) not in done and (L, s, sd, pc) in cells]
                if not todo: continue
                S = S or Subj(s); bb = load_bb(L, s)
                for sd, pc in todo:
                    r = cells[(L, s, sd, pc)]; obs, mask, _ = S.corrupt(pc, sd)
                    assert mask.tolist() == r["is_corrupted"], "corruption draw changed"
                    res = np.asarray(r["resub"]); g = rng_key(s, "subst", pc, sd)
                    subs = g.choice(S.sub_idx, K, replace=False)
                    memo = {}
                    def run(flag):
                        key = tuple(sorted(int(i) for i in flag))
                        if key not in memo:
                            lab = obs.copy(); im = S.sup_imgs.copy()
                            for j, i in enumerate(key):
                                im[i] = S.imgs[subs[j]]; lab[i] = gm.vector_to_pitchyaw(S.vecs[subs[j]][None]).reshape(2)
                            memo[key] = (S.mae(api.adapt(bb, im, lab)), len(key))
                        return memo[key]
                    row = {"level": L, "subject": s, "seed": sd, "pc": pc, "n_corrupted": int(mask.sum())}
                    row["naive"] = run([])[0]; row["oracle_true"] = run(np.where(mask)[0])[0]
                    loo = np.asarray(r["loo"])
                    for pol, t in thr[L].items():
                        tau = t[str(pc)] if isinstance(t, dict) else t
                        sc_ = loo if pol.startswith("LOO_") else res      # LOO_* policies gate on the LOO residual
                        m, n = run(np.where(sc_ >= tau)[0]); row[pol] = m; row[pol + "_n"] = n
                    f.write(json.dumps(row) + "\n"); f.flush()
                print(f"[mae] {L} {s} done  t={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["score", "mae"])
    ap.add_argument("--levels", nargs="+", default=LEVELS); ap.add_argument("--subjects", nargs="+", default=SUBJ)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS); ap.add_argument("--thresholds")
    a = ap.parse_args(); (cmd_score if a.cmd == "score" else cmd_mae)(a)
