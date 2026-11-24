# exp_b.py  — Experiment B: two-channel damage decomposition (reduced design)
"""s0 backbone only, seeds 0-2, clean arm cached per subject. Per cell
(subject, channel, severity, p_c, seed): naive / oracle / clean arms ->
post-adaptation MAE on held-out frames -> lost-frame cost, misleading cost, ratio.
Channels: (a) label rotation δ∈{4,6,10}; (b) structured donor-swap;
(c) image corruption via cqs_detection; (d) coordinated next-target label.
Run `python exp_b.py --benchmark 20` first.  Raw per-cell CSV before aggregation.
"""
import argparse, csv, hashlib, json, os, sys, time
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); _AG = os.path.dirname(_HERE)
for p in (_HERE, os.path.join(_AG, "scripts"), _AG):
    if p not in sys.path: sys.path.insert(0, p)
import cv2, torch
import gaze_math as gm, backbone_api as api, corruption as corr
import train_backbone as tb

SUBJ = [f"p{i:02d}" for i in range(15)]
PCS = [0.1, 0.2, 0.4]
LABEL_DELTAS = [4, 6, 10]
CONDS = ([("a_rot", d) for d in LABEL_DELTAS] + [("b_swap", None), ("c_image", None), ("d_coord", None)])
_cqs = None
def cqs():
    global _cqs
    if _cqs is None:
        import cqs_detection as c; _cqs = c
    return _cqs


def _rng(sid, chan, sev, pc, seed):
    key = f"{sid}|{chan}|{sev}|{pc}|{seed}".encode()
    return np.random.RandomState(int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2 ** 32))


def eval_mae(model, images, labvecs, chunk=512):
    out = []
    for i in range(0, len(images), chunk):
        pv = api.predict(model, images[i:i + chunk]); out.append(gm.angular_error_vec_deg(pv, labvecs[i:i + chunk]))
    return float(np.concatenate(out).mean())


class Subject:
    def __init__(self, s, device):
        self.s = s; self.device = device
        self.bb = self._bb()
        imgs, vecs = tb.load_images(s, None, np.random.default_rng(0))
        self.imgs, self.vecs, self.n = imgs, vecs, len(imgs)
        self.sup_ids = np.asarray(api.calibration_split(self.n, 9, seed=42)[0])
        self.sup_imgs = imgs[self.sup_ids]
        self.sup_lab = gm.vector_to_pitchyaw(vecs[self.sup_ids]).reshape(-1, 2)
        em = np.ones(self.n, bool); em[self.sup_ids] = False
        self.ev_imgs = imgs[em]; self.ev_labvec = gm.pitchyaw_to_vector(gm.vector_to_pitchyaw(vecs[em]).reshape(-1, 2))
        self.pool_lab = gm.vector_to_pitchyaw(vecs).reshape(-1, 2)
        self.clean = eval_mae(api.adapt(self.bb, self.sup_imgs, self.sup_lab), self.ev_imgs, self.ev_labvec)  # CACHED
        self._faces = None
        lines = open(os.path.join(tb.ROOT, "Label", f"{s}.label")).read().strip().split("\n")
        oi = lines[0].split().index("Origin")  # Origin col = original dayNN/FFFF.jpg -> day boundary
        self.days = [ln.split()[oi].split("/")[0] for ln in lines[1:]]

    def _bb(self):
        m = tb.build_model(); m.load_state_dict(torch.load(os.path.join(_AG, "models", f"backbone_mpiifacegaze_{self.s}.pt"), map_location="cpu"))
        return m.to(self.device).eval()

    def face_path(self, pos):
        if self._faces is None: self._faces = cqs().face_paths(self.s)
        return self._faces[pos]

    def corrupt_image_112(self, pos, rng):
        gen = np.random.default_rng(int(rng.randint(0, 2 ** 32)))  # cqs.corrupt needs Generator API
        im = cqs().load_img(self.face_path(pos)); pts = cqs().landmarks_xy(im)
        ci, mode = cqs().corrupt(im, pts, gen)
        return cv2.resize(ci, (tb.IMG, tb.IMG), interpolation=cv2.INTER_AREA), mode

    def cell(self, chan, sev, pc, seed):
        rng = _rng(self.s, chan, sev, pc, seed)
        mask = corr.draw_corruption_mask(9, pc, rng)
        cor = np.where(mask)[0]; keep = np.where(~mask)[0]
        n_cor = int(mask.sum())
        naive_imgs = self.sup_imgs.copy(); naive_lab = self.sup_lab.copy()
        med_off = np.nan; offs = []; modes = []; straddle = 0
        if chan == "a_rot":
            for i in cor: naive_lab[i], o = gm.rotate_by_angle(self.sup_lab[i], sev, rng=rng); offs.append(o)
        elif chan == "b_swap":
            nl, offarr = corr.structured_swap(self.sup_lab, mask, self.pool_lab, rng); naive_lab = nl; offs = list(offarr[mask])
        elif chan == "d_coord":
            for i in cor:
                p = int(self.sup_ids[i]); q = p + 1
                if q >= self.n: continue
                naive_lab[i] = self.pool_lab[q]
                offs.append(gm.angular_error_deg(self.sup_lab[i], self.pool_lab[q]))
                if self.days[p] != self.days[q]: straddle += 1
        elif chan == "c_image":
            for i in cor:
                naive_imgs[i], mode = self.corrupt_image_112(int(self.sup_ids[i]), rng)  # label stays correct
                modes.append(mode)
        if offs: med_off = float(np.median(offs))
        naive = eval_mae(api.adapt(self.bb, naive_imgs, naive_lab), self.ev_imgs, self.ev_labvec)
        if len(keep) >= 1:
            oracle = eval_mae(api.adapt(self.bb, self.sup_imgs[keep], self.sup_lab[keep]), self.ev_imgs, self.ev_labvec)
        else:
            oracle = np.nan
        lost = oracle - self.clean; mis = naive - oracle
        return dict(subject=self.s, channel=chan, severity=sev, pc=pc, seed=seed, n_corrupted=n_cor,
                    clean=self.clean, oracle=oracle, naive=naive, lost_cost=lost, misleading_cost=mis,
                    ratio=(mis / lost if lost == lost and abs(lost) > 1e-9 else np.nan),
                    median_offset_deg=med_off, image_modes="|".join(modes), day_straddle=straddle)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--benchmark", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(_HERE, "exp_b_records.csv")); a = ap.parse_args()
    device = api.get_device()
    if a.benchmark:
        subj = Subject("p00", device)  # includes clean-arm + load
        # conditions innermost so the first N cells span ALL channels (esp. c_image/MediaPipe)
        cells = [(c, sev, pc, seed) for pc in PCS for seed in (0, 1, 2) for (c, sev) in CONDS]
        per = {}
        for (c, sev, pc, seed) in cells[:a.benchmark]:
            t = time.time(); subj.cell(c, sev, pc, seed); per.setdefault(c, []).append(time.time() - t)
        allt = [x for v in per.values() for x in v]
        print(f"BENCHMARK: {len(allt)} cells, mean {np.mean(allt):.1f} s/cell "
              f"(max {np.max(allt):.1f}); clean-arm+load excluded")
        for c in per: print(f"  {c:9s} n={len(per[c])}  mean {np.mean(per[c]):.1f} s/cell")
        return
    rows = []
    with open(a.out, "w", newline="") as f:
        w = None
        for s in SUBJ:
            subj = Subject(s, device)
            for (c, sev) in CONDS:
                for pc in PCS:
                    for seed in (0, 1, 2):
                        r = subj.cell(c, sev, pc, seed); rows.append(r)
                        if w is None:
                            w = csv.DictWriter(f, fieldnames=list(r.keys())); w.writeheader()
                        w.writerow(r)
            f.flush(); print(f"  [{s}] done ({len(rows)} cells)", flush=True)
    print(f"wrote {len(rows)} cells -> {a.out}")


if __name__ == "__main__":
    main()
