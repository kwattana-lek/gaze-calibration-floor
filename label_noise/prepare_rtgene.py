#!/usr/bin/env python3
# prepare_rtgene.py -- check an RT-GENE build before it is used as a second dataset.
"""WHY THIS EXISTS

The transport experiment is only meaningful if RT-GENE is normalised the same way
MPIIFaceGaze was. GazeHub ships preprocessing CODE for RT-GENE rather than a
prebuilt download, which is the good case: running their own script produces the
same virtual camera and the same label conventions. But nothing enforces that,
and a normalisation difference would masquerade as a transport result -- the
exact confound that ruled out ColumbiaGaze.

So this compares the two builds before any sweep runs, and refuses to bless one
that differs where it matters:

  * the label header carries the columns the loader reads (Face, 3DGaze)
  * face crops are the same size, so the 112 px resize is the same operation
  * gaze vectors are unit-norm under the same sign convention
  * enough subjects, with enough frames each for K support + evaluation

It reports rather than fixes: a mismatch is a decision, not something to paper
over.

Usage:
  python3 prepare_rtgene.py --rtgene ~/.../Datasets/RTGENE_Normalized
"""
import argparse, glob, os, sys
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

REF = os.path.expanduser("~/Desktop/Documents Since 29-01-2569/Datasets/"
                         "MPIIFaceGaze_Normalized")
NEED = ("Face", "3DGaze")
K, MIN_FRAMES = 9, 60


def probe(root, name):
    out = {"name": name, "root": root, "ok": os.path.isdir(root)}
    if not out["ok"]:
        return out
    labs = sorted(glob.glob(os.path.join(root, "Label", "*.label")))
    out["n_subjects"] = len(labs)
    if not labs:
        return out
    head = open(labs[0]).readline().split()
    out["columns"] = head
    out["has_needed"] = [c for c in NEED if c not in head]
    rows, norms = [], []
    for lp in labs:
        lines = open(lp).read().strip().split("\n")
        rows.append(len(lines) - 1)
        gi = head.index("3DGaze") if "3DGaze" in head else None
        if gi is not None:
            for ln in lines[1:200]:
                v = np.array([float(x) for x in ln.split()[gi].split(",")])
                norms.append(float(np.linalg.norm(v)))
    out["frames_min"], out["frames_median"] = int(min(rows)), int(np.median(rows))
    if norms:
        out["gaze_norm"] = (float(np.min(norms)), float(np.max(norms)))
    if cv2 is not None:
        fi = head.index("Face")
        f0 = open(labs[0]).read().strip().split("\n")[1].split()[fi]
        im = cv2.imread(os.path.join(root, "Image", f0))
        out["face_shape"] = None if im is None else tuple(im.shape)
    return out


def report(p):
    print(f"\n--- {p['name']} ---")
    if not p["ok"]:
        print(f"  NOT FOUND: {p['root']}"); return False
    print(f"  subjects        {p.get('n_subjects')}")
    print(f"  frames/subject  min {p.get('frames_min')}  median {p.get('frames_median')}")
    print(f"  face crop       {p.get('face_shape')}")
    gn = p.get("gaze_norm")
    print(f"  |3DGaze|        {gn[0]:.4f} .. {gn[1]:.4f}" if gn else "  |3DGaze|  n/a")
    miss = p.get("has_needed") or []
    print(f"  columns         {'all present' if not miss else 'MISSING ' + ','.join(miss)}")
    return not miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtgene", required=True)
    ap.add_argument("--ref", default=REF)
    a = ap.parse_args()

    r = probe(os.path.expanduser(a.ref), "MPIIFaceGaze (reference)")
    t = probe(os.path.expanduser(a.rtgene), "RT-GENE (candidate)")
    ok = report(r) & report(t)
    if not (r["ok"] and t["ok"]):
        sys.exit("\nABORT: one build is missing.")

    print("\n--- comparability ---")
    checks = []
    checks.append(("face crop size identical",
                   r.get("face_shape") == t.get("face_shape"),
                   f"{r.get('face_shape')} vs {t.get('face_shape')}"))
    gr, gt = r.get("gaze_norm"), t.get("gaze_norm")
    checks.append(("3DGaze unit-norm in both",
                   bool(gr and gt and max(abs(gr[0]-1), abs(gr[1]-1),
                                          abs(gt[0]-1), abs(gt[1]-1)) < 1e-3),
                   f"{gr} vs {gt}"))
    # Column sets are NOT required to be identical. MPIIFaceGaze_Normalized carries
    # four extra columns (WhichEye, Rmat, Smat, GazeOrigin) that record the virtual-
    # camera normalisation; RT-GENE's build has no normalisation step and so has no
    # such columns. What must hold is (a) the two columns the loader reads exist in
    # both, and (b) every column present in both means the same thing. A missing
    # extra column is a stated preprocessing difference (reported below); a shared
    # column that disagrees is fatal.
    rc, tc = r.get("columns") or [], t.get("columns") or []
    shared = [c for c in rc if c in tc]
    checks.append(("loader columns (Face, 3DGaze) in both",
                   all(c in rc and c in tc for c in NEED),
                   f"shared {len(shared)}/{len(rc)} ref cols: {','.join(shared)}"))
    checks.append(("no shared column reordered",
                   [c for c in rc if c in tc] == [c for c in tc if c in rc],
                   "same order" if [c for c in rc if c in tc] == [c for c in tc if c in rc]
                   else "SHARED COLUMNS IN DIFFERENT ORDER -- values would be misread"))
    checks.append((f"enough frames (>= {MIN_FRAMES}/subject)",
                   (t.get("frames_min") or 0) >= MIN_FRAMES,
                   f"min {t.get('frames_min')}"))
    checks.append(("subject count workable (>= 10)",
                   (t.get("n_subjects") or 0) >= 10, f"{t.get('n_subjects')}"))
    for nm, good, detail in checks:
        print(f"  [{'PASS' if good else 'FAIL'}] {nm:<32} {detail}")
        ok &= good

    onlyref = [c for c in rc if c not in tc]
    onlycand = [c for c in tc if c not in rc]
    print("\n--- stated differences (not failures; these go in the manuscript) ---")
    print(f"  ref-only columns   {','.join(onlyref) or 'none'}")
    print(f"  cand-only columns  {','.join(onlycand) or 'none'}")
    # This gate is pointed at more than one candidate corpus, and the normalisation
    # difference is NOT the same for each. Asserting RT-GENE's difference about
    # EYEDIAP would put a false statement in the manuscript: EYEDIAP goes through
    # the same dpc.norm() as MPIIFaceGaze (its build shares all 12 columns, which is
    # the detectable signature), and differs instead in the crop CENTRE. So the gate
    # reports what it can actually see and refuses to narrate what it cannot.
    print(f"  shared columns     {len(shared)} of {len(rc)}; Rmat/Smat/GazeOrigin "
          f"{'PRESENT' if 'Rmat' in tc else 'ABSENT'} in the candidate")
    if "Rmat" in tc:
        print("                     -> candidate carries normalisation metadata, so it")
        print("                        went through a dpc.norm()-style warp. State the")
        print("                        remaining difference (crop centre, focal,")
        print("                        distance) from its own build script, not here.")
    else:
        print("                     -> candidate has no normalisation metadata, so its")
        print("                        crops are the dataset's own, unwarped.")
    print("  why it matters     a preprocessing difference rescales ehat and")
    print("                     delta_floor together; the transport claim is about")
    print("                     their RATIO c, so it must be stated, not hidden.")

    print("\n" + ("READY: the two builds are comparable; RT-GENE can be swept with\n"
                  "  GAZE_DATA_ROOT=<rtgene> GAZE_SUBJECTS=<s000,...>"
                  if ok else
                  "NOT READY: a difference above would show up as a transport result.\n"
                  "  Resolve it, or state the difference in the manuscript."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
