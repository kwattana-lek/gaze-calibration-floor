#!/usr/bin/env python3
# prepare_rtgene_labels.py -- build an RT-GENE label tree our pipeline can read.
"""WHY THIS EXISTS (and why GazeHub's own scripts are not run as shipped)

GazeHub ships data_processing_rt_train.py / _test.py. Both were read line by line
before this file was written. Neither is usable here, for reasons that are
methodological rather than cosmetic:

 1. _train.py writes TWO rows per label line -- the same gaze label attached once
    to original/face and once to inpainted/face. Every label therefore appears
    twice. Our corruption model draws one corruption decision per frame; a
    duplicated label means a corrupted label is injected twice and the clean/
    corrupt frames are near-duplicates of each other. That silently inflates both
    detectability and the identifiability index. Fatal. We emit one row per frame.

 2. We take the INPAINTED crop, not the original. RT-GENE's glasses subjects wear
    the eyetracking headset in `original/`; the headset's appearance is constant
    within a subject and differs between subjects, i.e. it is a nuisance factor
    perfectly confounded with subject identity. A backbone can read identity off
    it, which is exactly what the identifiability index I is meant to measure.
    _test.py uses original only, so it is unusable for the opposite reason.

 3. Both scripts bin subjects into RT-GENE's three training folds plus a valid
    fold, and s000 appears in NO fold -- it is silently dropped. Our protocol is
    leave-one-person-out over every session, so we emit one label file per
    session, named like MPIIFaceGaze's Label/p00.label, and the LOPO code needs
    no change.

 4. Both scripts pair label_info[num] with sorted(imgs)[num] positionally and
    never check the pairing. We parse the frame id out of the label line and
    assert it equals the frame number in the filename, for every row. A
    mispairing here would be a silent per-frame label error, which is precisely
    the thing this paper is about measuring.

 5. Lines 111-113 of _train.py index the inpainted save-names with imgs_face[num]
    (the ORIGINAL listing) rather than pimgs_face[num]. Harmless in fact -- the
    two filename sets are identical -- but we assert that rather than inherit it.

NO IMAGES ARE COPIED. Image/ is a relative symlink to ../rt_gene, so the 103k
crops stay where they were downloaded.

GazeTo3d is not imported from data_processing_core (not shipped with the two
scripts). It is reimplemented here and VERIFIED against MPIIFaceGaze_Normalized's
own 2DGaze/3DGaze columns, which were produced by the real function, before any
RT-GENE row is written.

Usage:
  python3 prepare_rtgene_labels.py --src ../data/rt_gene --out ../data/RTGENE_Normalized
"""
import argparse, os, re, sys
import numpy as np

MPII_REF = os.path.expanduser("~/Desktop/Documents Since 29-01-2569/Datasets/"
                              "MPIIFaceGaze_Normalized")

# RT-GENE has 17 SESSIONS but only 15 PARTICIPANTS. Two people were recorded a
# second time, on "different days and different camera poses" (the dataset's own
# Zenodo record, doi 10.5281/zenodo.2529036). Session ids are therefore NOT
# participant ids, and nothing in the download says so -- there is no metadata
# file in the corpus at all, and GazeHub's scripts treat all 17 as distinct.
#
# This matters twice over:
#   * pooling. Residuals are pooled across subjects into one ROC per cell, so a
#     duplicated person is weighted twice. With 16 usable sessions that is 4/16
#     of the pool for two people instead of 2/14.
#   * any leave-one-person-out training. Holding out s014 while training on s002
#     is testing on a person the model has already seen. The transport arm uses
#     frozen features and never trains, so it is unaffected -- but a fine-tuned
#     RT-GENE arm would leak, and must use PARTICIPANTS below, not the sessions.
REPEAT_SESSIONS = {"s014": "s002", "s015": "s005"}


def unique_participants(sessions):
    """Sessions with the repeat recordings dropped, keeping the first of each."""
    return [s for s in sessions if s not in REPEAT_SESSIONS]
HEADER = "Face Left Right Origin 3DGaze 3DHead 2DGaze 2DHead"
SPLIT = re.compile(r"\[| |,|\]")
FRAMENUM = re.compile(r"_(\d+)_")


def gaze_to_3d(g):
    """GazeHub data_processing_core.GazeTo3d. Verified in verify_gaze_to_3d()."""
    return np.array([-np.cos(g[1]) * np.sin(g[0]),
                     -np.sin(g[1]),
                     -np.cos(g[1]) * np.cos(g[0])], dtype=np.float64)


def verify_gaze_to_3d(ref_root, n=4000):
    """Refuse to run if our GazeTo3d disagrees with the one that built MPIIFaceGaze."""
    import glob
    labs = sorted(glob.glob(os.path.join(ref_root, "Label", "*.label")))
    if not labs:
        sys.exit(f"ABORT: no reference labels under {ref_root}/Label -- cannot "
                 f"verify GazeTo3d. Pass --ref to point at MPIIFaceGaze_Normalized.")
    worst, seen = 0.0, 0
    for lp in labs:
        lines = open(lp).read().strip().split("\n")
        h = lines[0].split()
        g3, g2 = h.index("3DGaze"), h.index("2DGaze")
        for ln in lines[1:]:
            p = ln.split()
            want = np.array([float(x) for x in p[g3].split(",")])
            got = gaze_to_3d([float(x) for x in p[g2].split(",")])
            worst = max(worst, float(np.abs(want - got).max()))
            seen += 1
            if seen >= n:
                break
        if seen >= n:
            break
    if worst > 1e-5:
        sys.exit(f"ABORT: GazeTo3d disagrees with the reference build by {worst:.3e}. "
                 f"Do not emit labels from an unverified conversion.")
    print(f"[verify] GazeTo3d matches MPIIFaceGaze build on {seen} rows "
          f"(max abs err {worst:.2e})")


def frame_num(fname):
    m = FRAMENUM.search(fname)
    if not m:
        sys.exit(f"ABORT: cannot read a frame number out of {fname!r}")
    return int(m.group(1))


def listing(d):
    files = [f for f in os.listdir(d) if not f.startswith(".")]
    files.sort(key=frame_num)
    return files


def process(src, session, out_label):
    base = os.path.join(src, session)
    orig = {k: listing(os.path.join(base, "original", k)) for k in ("face", "left", "right")}
    inp = {k: listing(os.path.join(base, "inpainted", k)) for k in ("face", "left", "right")}

    for k in ("face", "left", "right"):
        if orig[k] != inp[k]:
            sys.exit(f"ABORT [{session}/{k}]: original and inpainted filename sets "
                     f"differ ({len(orig[k])} vs {len(inp[k])}). GazeHub's script "
                     f"indexes inpainted names with the original listing; that is "
                     f"only safe when the sets are identical.")

    labels = open(os.path.join(base, "label_combined.txt")).read().strip().split("\n")
    n = len(labels)
    for k in ("face", "left", "right"):
        if len(inp[k]) != n:
            sys.exit(f"ABORT [{session}/{k}]: {len(inp[k])} images vs {n} label lines.")

    rows, bad = [], 0
    for i, line in enumerate(labels):
        t = SPLIT.split(line.strip())
        fid = int(t[0])
        # Hard pairing guard: the label's own frame id must equal the image's.
        for k in ("face", "left", "right"):
            if frame_num(inp[k][i]) != fid:
                bad += 1
                sys.exit(f"ABORT [{session}] row {i}: label frame id {fid} but "
                         f"{k} image is {inp[k][i]}. Positional pairing is wrong.")
        head2d = [float(t[3]), float(t[5])]
        gaze2d = [float(t[9]), float(t[11])]
        g3 = gaze_to_3d(gaze2d)
        h3 = gaze_to_3d(head2d)
        rows.append(" ".join([
            f"{session}/inpainted/face/{inp['face'][i]}",
            f"{session}/inpainted/left/{inp['left'][i]}",
            f"{session}/inpainted/right/{inp['right'][i]}",
            session[:4],
            ",".join(g3.astype(str)),
            ",".join(h3.astype(str)),
            f"{gaze2d[0]},{gaze2d[1]}",
            f"{head2d[0]},{head2d[1]}",
        ]))

    with open(out_label, "w") as f:
        f.write(HEADER + "\n")
        f.write("\n".join(rows) + "\n")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir holding sNNN_glasses/")
    ap.add_argument("--out", required=True, help="RTGENE_Normalized root to create")
    ap.add_argument("--ref", default=MPII_REF, help="MPIIFaceGaze_Normalized, for verification")
    a = ap.parse_args()

    src = os.path.abspath(os.path.expanduser(a.src))
    out = os.path.abspath(os.path.expanduser(a.out))
    verify_gaze_to_3d(os.path.expanduser(a.ref))

    sessions = sorted(d for d in os.listdir(src)
                      if d.endswith("_glasses")
                      and os.path.isdir(os.path.join(src, d))
                      and os.path.isfile(os.path.join(src, d, "label_combined.txt")))
    if not sessions:
        sys.exit(f"ABORT: no complete sessions under {src}")
    print(f"[scan] {len(sessions)} complete sessions: {', '.join(s[:4] for s in sessions)}")

    os.makedirs(os.path.join(out, "Label"), exist_ok=True)
    link = os.path.join(out, "Image")
    target = os.path.relpath(src, out)
    if os.path.islink(link):
        if os.readlink(link) != target:
            os.unlink(link)
    if not os.path.exists(link):
        os.symlink(target, link)
    print(f"[link] Image -> {target}")

    total = 0
    for s in sessions:
        n = process(src, s, os.path.join(out, "Label", s[:4] + ".label"))
        total += n
        print(f"  {s[:4]}  {n:>6} frames")
    ids = [s[:4] for s in sessions]
    uniq = unique_participants(ids)
    dropped = [i for i in ids if i in REPEAT_SESSIONS]
    import json
    with open(os.path.join(out, "participants.json"), "w") as fh:
        json.dump({"sessions": ids,
                   "unique_participants": uniq,
                   "repeat_sessions": {k: v for k, v in REPEAT_SESSIONS.items()
                                       if k in ids},
                   "source": "RT-GENE Zenodo record doi 10.5281/zenodo.2529036: "
                             "15 participants across 17 sessions; two were "
                             "recorded a second time on different days and "
                             "different camera poses."}, fh, indent=2)

    print(f"[done] {len(sessions)} sessions, {total} frames -> {out}")
    if dropped:
        print(f"[participants] {len(uniq)} unique people in {len(ids)} sessions. "
              f"{', '.join(f'{k} repeats {REPEAT_SESSIONS[k]}' for k in dropped)} "
              f"(Zenodo doi 10.5281/zenodo.2529036). Session ids are not person ids.")
    print(f"[next] GAZE_DATA_ROOT={out} \\\n"
          f"       GAZE_SUBJECTS={','.join(ids)}          # all sessions\n"
          f"       GAZE_SUBJECTS={','.join(uniq)}   # unique participants")


if __name__ == "__main__":
    sys.exit(main())
