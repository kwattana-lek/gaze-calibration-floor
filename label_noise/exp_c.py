# exp_c.py — Experiment C: recovery under a recapture budget (reduced design)
"""s0 backbone only, seeds 0-2, budgets B∈{0,1,2}. Scorer = resubstitution residual
(sc.small_loss_scores). Residual-gate threshold FIXED at tau=3.42° (b182219) — never
re-selected here. Image gate = pre-trained held-out LOPO detector (detector_lopo_s0_pXX).

Policies (per cell, per B):
  1 naive        adapt on corrupted K as-is
  2 discard-B    drop top-B residual frames, adapt on remainder
  3 recapture-B  replace top-B residual frames with clean pool substitutes, adapt on K
  4 two-channel  residual gate (discard) + image gate (recapture) within budget B,
                 run PARALLEL and CASCADE (separate rows)
  5 oracle       policy-4 routing with TRUE identities+channels (label→discard,
                 image→recapture) — the ceiling, in the same table.

Channels: policies 1-3 label-only (structured donor-swap), p_c∈{.1,.2,.4}. Two-channel
rows (4,5) mix label+image corruption, p_c=0.2 only. Shared adaptations memoised per cell.
`--benchmark N` first. Raw per-cell CSV before aggregation.
"""
import argparse, csv, hashlib, os, sys, time
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); _AG = os.path.dirname(_HERE)
for p in (_HERE, os.path.join(_AG, "scripts"), _AG):
    if p not in sys.path: sys.path.insert(0, p)
import cv2, torch
import gaze_math as gm, backbone_api as api, corruption as corr
import exp_a_scorers as sc
import train_backbone as tb
from train_detector import build_resnet, to_batch as det_batch

SUBJ = [f"p{i:02d}" for i in range(15)]
PCS_LABEL = [0.1, 0.2, 0.4]; PC_TWO = 0.2; BUDGETS = [0, 1, 2]
TAU = 3.42; DET_THR = 0.5; SIGN = -1; EVAL_SUBSAMPLE = 1000
_cqs = None
def cqs():
    global _cqs
    if _cqs is None:
        import cqs_detection as c; _cqs = c
    return _cqs


def _rng(sid, tag, pc, seed):
    key = f"{sid}|{tag}|{pc}|{seed}".encode()
    return np.random.RandomState(int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2 ** 32))


def eval_mae(model, images, labvecs, chunk=512):
    out = []
    for i in range(0, len(images), chunk):
        out.append(gm.angular_error_vec_deg(api.predict(model, images[i:i + chunk]), labvecs[i:i + chunk]))
    return float(np.concatenate(out).mean())


class Subject:
    def __init__(self, s, device):
        self.s = s; self.device = device
        self.bb = self._load(tb.build_model(), os.path.join(_AG, "models", f"backbone_mpiifacegaze_{s}.pt"))
        self.det = self._load(build_resnet(), os.path.join(_AG, "models", "detectors", f"detector_lopo_s0_{s}.pt"))
        imgs, vecs = tb.load_images(s, None, np.random.default_rng(0))
        self.imgs, self.n = imgs, len(imgs)
        self.sup_ids = np.asarray(api.calibration_split(self.n, 9, seed=42)[0])
        self.sup_imgs = imgs[self.sup_ids]
        self.sup_lab = gm.vector_to_pitchyaw(vecs[self.sup_ids]).reshape(-1, 2)   # clean labels
        em = np.ones(self.n, bool); em[self.sup_ids] = False
        self.ev_idx = np.where(em)[0]
        ev_all = np.where(em)[0]
        # fixed 1000-frame eval subsample per subject (deterministic; relative/paired comparisons unaffected)
        if len(ev_all) > EVAL_SUBSAMPLE:
            ev_all = np.sort(np.random.RandomState(1000).choice(ev_all, EVAL_SUBSAMPLE, replace=False))
        self.ev_imgs = imgs[ev_all]
        self.ev_labvec = gm.pitchyaw_to_vector(gm.vector_to_pitchyaw(vecs[ev_all]).reshape(-1, 2))
        self.pool_lab = gm.vector_to_pitchyaw(vecs).reshape(-1, 2)
        self._faces = None

    def _load(self, m, path):
        m.load_state_dict(torch.load(path, map_location="cpu")); return m.to(self.device).eval()

    def det_scores(self, imgs112):
        with torch.no_grad():
            return torch.sigmoid(self.det(det_batch(imgs112, self.device))).cpu().numpy().reshape(-1)

    def corrupt_image_112(self, pos, rng):
        if self._faces is None: self._faces = cqs().face_paths(self.s)
        gen = np.random.default_rng(int(rng.randint(0, 2 ** 32)))
        im = cqs().load_img(self._faces[pos]); pts = cqs().landmarks_xy(im)
        ci, mode = cqs().corrupt(im, pts, gen)
        return cv2.resize(ci, (tb.IMG, tb.IMG), interpolation=cv2.INTER_AREA), mode

    # ---------- one cell ----------
    def cell(self, channel, pc, seed):
        """Build observed support + scores + substitutes for this (channel,pc,seed).
        channel: 'label' or 'two'. Returns a dict of per-B policy results."""
        rng = _rng(self.s, channel, pc, seed)
        mask = corr.draw_corruption_mask(9, pc, rng)
        cor = np.where(mask)[0]
        obs_imgs = self.sup_imgs.copy(); obs_lab = self.sup_lab.copy()
        true_ch = ["none"] * 9   # per-frame true channel
        if channel == "label":
            nl, _ = corr.structured_swap(self.sup_lab, mask, self.pool_lab, rng); obs_lab = nl
            for i in cor: true_ch[i] = "label"
        else:  # two-channel: each corrupted frame -> label OR image
            for i in cor:
                if rng.random_sample() < 0.5:
                    donor, _ = corr.structured_swap(self.sup_lab, (np.arange(9) == i), self.pool_lab, rng)
                    obs_lab[i] = donor[i]; true_ch[i] = "label"
                else:
                    obs_imgs[i], _ = self.corrupt_image_112(int(self.sup_ids[i]), rng); true_ch[i] = "image"
        # scores on observed support
        r = np.asarray(sc.small_loss_scores(self.bb, obs_imgs, obs_lab), float)   # residual gate (label-suspect)
        d = self.det_scores(obs_imgs)                                             # image gate (image-suspect)
        # substitute clean frames (distinct, from eval pool)
        sub_ids = rng.choice(self.ev_idx, size=3, replace=False)
        sub_imgs = self.imgs[sub_ids]; sub_lab = self.pool_lab[sub_ids]
        memo = {}

        def ae(kept, subk):   # kept: obs indices to keep; subk: # substitutes to append
            key = (tuple(sorted(kept)), int(subk))
            if key in memo: return memo[key]
            ims = [obs_imgs[list(kept)]] if kept else []
            lbs = [obs_lab[list(kept)]] if kept else []
            if subk: ims.append(sub_imgs[:subk]); lbs.append(sub_lab[:subk])
            IM = np.concatenate(ims, 0); LB = np.concatenate(lbs, 0)
            m = api.adapt(self.bb, IM, LB); mae = eval_mae(m, self.ev_imgs, self.ev_labvec)
            memo[key] = mae; return mae

        allk = list(range(9))
        mae_naive = ae(allk, 0)
        order = list(np.argsort(-r))   # highest residual first
        istrue = lambda i: true_ch[i] != "none"

        def spend_prec(acted):
            return (np.mean([istrue(i) for i in acted]) if acted else np.nan)

        results = {}
        for B in BUDGETS:
            top = order[:B]
            # policy 2 discard
            keep2 = [i for i in allk if i not in top]
            p2 = ae(keep2, 0)
            # policy 3 recapture
            p3 = ae(keep2, len(top))
            row = dict(naive=mae_naive, discard=p2, recapture=p3,
                       prec_discard=spend_prec(top), prec_recapture=spend_prec(top))
            if channel == "two":
                # gates
                res_flag = [i for i in allk if r[i] >= TAU]
                img_flag = [i for i in allk if d[i] >= DET_THR]
                # ---- PARALLEL: rank all flagged by normalised exceedance ----
                cand = []
                for i in allk:
                    if r[i] >= TAU: cand.append((r[i] / TAU, i, "discard"))
                    elif d[i] >= DET_THR: cand.append((d[i] / DET_THR, i, "recapture"))
                cand.sort(reverse=True)
                par = cand[:B]
                dpar = [i for _, i, a in par if a == "discard"]; rpar = [i for _, i, a in par if a == "recapture"]
                keepP = [i for i in allk if i not in dpar and i not in rpar]
                pP = ae(keepP, len(rpar))
                # ---- CASCADE: residual-gate first (discard), then image-gate (recapture) ----
                b = B; dcas = [i for i in order if r[i] >= TAU][:b]; b -= len(dcas)
                rcas = [i for i in np.argsort(-d) if d[i] >= DET_THR and i not in dcas][:max(0, b)]
                keepC = [i for i in allk if i not in dcas and i not in rcas]
                pC = ae(keepC, len(rcas))
                # ---- ORACLE: true channels, within B (by residual priority) ----
                true_cor = [i for i in order if istrue(i)][:B]
                dor = [i for i in true_cor if true_ch[i] == "label"]; ror = [i for i in true_cor if true_ch[i] == "image"]
                keepO = [i for i in allk if i not in dor and i not in ror]
                pO = ae(keepO, len(ror))
                acted_par = dpar + rpar
                row.update(par=pP, cascade=pC, oracle=pO,
                           prec_par=spend_prec(acted_par),
                           # confusion (parallel): image-corrupted DISCARDED (wasteful); label-corrupted RECAPTURED (futile)
                           img_discarded=sum(1 for i in dpar if true_ch[i] == "image"),
                           lab_recaptured=sum(1 for i in rpar if true_ch[i] == "label"),
                           n_acted_par=len(acted_par))
            else:
                # label-only oracle = perfect recapture of true label-corrupted, within B
                true_cor = [i for i in order if istrue(i)][:B]
                pO = ae([i for i in allk if i not in true_cor], len(true_cor))
                row["oracle"] = pO
            results[B] = row
        return results, int(mask.sum()), true_ch


def recovered(naive, policy, oracle):
    den = naive - oracle
    return float((naive - policy) / den) if abs(den) > 0.05 else np.nan


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--benchmark", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(_HERE, "exp_c_records.csv")); a = ap.parse_args()
    device = api.get_device()
    label_cells = [("label", pc, seed) for pc in PCS_LABEL for seed in (0, 1, 2)]
    two_cells = [("two", PC_TWO, seed) for seed in (0, 1, 2)]
    if a.benchmark:
        subj = Subject("p00", device)
        order = [c for pair in zip(label_cells * 3, (two_cells * 5)) for c in pair][:a.benchmark]
        order = order or label_cells
        per = {}
        for (ch, pc, seed) in order[:a.benchmark]:
            t = time.time(); subj.cell(ch, pc, seed); per.setdefault(ch, []).append(time.time() - t)
        allt = [x for v in per.values() for x in v]
        print(f"BENCHMARK: {len(allt)} cells, mean {np.mean(allt):.1f} s/cell (max {np.max(allt):.1f}); load excluded")
        for c in per: print(f"  {c:6s} n={len(per[c])} mean {np.mean(per[c]):.1f} s/cell")
        return
    FIELDS = ["subject", "channel", "pc", "seed", "B", "n_corrupted", "naive", "discard", "recapture",
              "prec_discard", "prec_recapture", "oracle", "par", "cascade", "prec_par",
              "img_discarded", "lab_recaptured", "n_acted_par"]
    rows = []
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS, restval=""); wr.writeheader()
        for s in SUBJ:
            subj = Subject(s, device)
            for (ch, pc, seed) in label_cells + two_cells:
                res, ncor, _ = subj.cell(ch, pc, seed)
                for B, row in res.items():
                    rec = dict(subject=s, channel=ch, pc=pc, seed=seed, B=B, n_corrupted=ncor, **row)
                    rows.append(rec); wr.writerow(rec)
            f.flush(); print(f"  [{s}] done ({len(rows)} rows)", flush=True)
    print(f"wrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
