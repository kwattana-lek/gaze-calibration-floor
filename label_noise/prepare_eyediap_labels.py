#!/usr/bin/env python3
# prepare_eyediap_labels.py -- build an EYEDIAP label tree our pipeline can read.
"""WHY THIS EXISTS

EYEDIAP is the second (third, with RT-GENE) corpus for the transport of c. Unlike
RT-GENE it ships raw video plus 3D annotation, so it CAN go through the same
virtual-camera normalisation that built MPIIFaceGaze_Normalized. This script does
that with the very same dpc.norm() class (GazeHub data_processing_core, imported,
not reimplemented), so the MPII/EYEDIAP contrast carries no preprocessing
difference of the kind that had to be stated for RT-GENE.

What is different from GazeHub's data_processing_eyediap.py, and why:

 1. GAZE STATE. EYEDIAP's authors publish frame-level gaze-state annotation
    (OK / BK blink / NL not-looking) and "strongly recommend" it be used. GazeHub
    ignores it. We do not: a frame whose subject is not looking at the target has
    a clean image and a WRONG LABEL, which is the exact object this paper studies.
    Letting such frames into the "clean" pool would contaminate e_hat and every
    clean/corrupt contrast. So:
        Label/<id>.label        OK frames only, every --stride-th frame.
        Label_lapse/<id>.label  every NL and BK frame (stride 1), never used as
                                clean data; this is the measured-lapse material.
    Only CS sessions are annotated (DS is not), so only CS sessions are used.

 2. PAIRING. Annotation files are joined on their own frame index, never
    positionally, and the video's frame count is checked against them.

 3. GEOMETRY IS VERIFIED, NOT ASSUMED. Before anything is written for a session,
    the 3D eyeball centres are pushed through our reading of the VGA calibration
    and compared with the 2D VGA eyeball centres in the same file. If our reading
    of R/T/units were wrong this fails by hundreds of pixels; we abort above
    --reproj-tol px. The normalised frame must also come out upright.

 4. EXTRA COLUMNS for the lapse analysis, appended after the reference columns so
    the existing loader (which indexes by header name) is unaffected:
        Session Frame State Traj SinceJump LapseDeg
    Traj      index of the target trajectory (a jump of the on-screen target
              larger than --jump-px starts a new one)
    SinceJump frames since the current trajectory began
    LapseDeg  angle between the gaze to the CURRENT target and the gaze to the
              PREVIOUS trajectory's final target, from the current face centre.
              For a post-jump NL frame this is the label displacement if the eyes
              are still where they were -- an upper bound later in the window.

Subjects are written as e01..e16 (participants 12 and 13 have no screen sessions
and do not appear). The S and M head-pose sessions of a participant go into the
same label file; the Session column keeps them separable.

Usage:
  python3 prepare_eyediap_labels.py --src ../data/eyediap/Data \
          --state ../data/eyediap/docs/EYEDIAP/Annotations/GazeState/GazeStateExport/Data \
          --out ../data/EYEDIAP_Normalized [--only 1_A_CS_M] [--dry]
"""
import argparse, json, os, sys
import numpy as np
import cv2

DPC_DEFAULT = os.path.expanduser(
    "~/Desktop/Documents Since 29-01-2569/Dynamic Adapter Placement_prev/"
    "Source Codes/data_processing/core")
HEADER = ("Face Left Right Origin WhichEye 3DGaze 3DHead 2DGaze 2DHead Rmat Smat "
          "GazeOrigin Session Frame State Traj SinceJump LapseDeg")
IMSIZE = (224, 224)


def read_table(path):
    """frame -> list of floats. Joined on the file's own frame index."""
    out = {}
    with open(path) as fh:
        header = fh.readline().strip()
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            p = ln.split(";")
            try:
                out[int(float(p[0]))] = [float(v) for v in p[1:]]
            except ValueError:
                continue                     # a row with a blank field = tracking lost
    return header, out


def read_calibration(path):
    cal, key, rows = {}, None, []
    for ln in open(path):
        ln = ln.strip()
        if ln.startswith("["):
            if key:
                cal[key] = np.array(rows, dtype=np.float64)
            key, rows = ln.strip("[]"), []
        elif ln:
            rows.append([float(v) for v in ln.split(";") if v != ""])
    if key:
        cal[key] = np.array(rows, dtype=np.float64)
    K, R, T = cal["intrinsics"], cal["R"], cal["T"].reshape(3)
    assert K.shape == (3, 3) and R.shape == (3, 3), f"bad calibration in {path}"
    return K, R, T


def read_state(path):
    out = {}
    for ln in open(path):
        p = ln.split()
        if len(p) >= 2:
            out[int(p[0])] = p[1].strip(",")
    return out


def to_cam(p_w, R, T):
    """World -> camera. From the dataset's own view_session.project_points:
    points_c = points . R - R^T T   (row vectors)  ==  R^T (p - T)."""
    return R.T @ (np.asarray(p_w, dtype=np.float64) - T)


def target_to_cam(p, R, T, frame):
    """Screen target -> VGA camera frame (metres).

    eye_tracking.txt and head_pose.txt are in the WCS, which the VGA calibration
    defines as p_w = R p_c + T with R = diag(1,-1,-1), T = (0,0,1): an OpenGL-style
    camera frame pushed 1 m down the optical axis. That reading is VERIFIED per
    session by reprojection. screen_coordinates.txt is NOT in that frame: read as
    WCS it puts the screen ~0.5 m BEHIND the participant. Its 3D columns are
    Metadata/ScreenCalibration applied to the pixel position, and that calibration
    is expressed in a camera-centred frame. Which one is settled by evidence
    (check_eyediap_target_frame.py), not by assumption:
      cam  the VGA camera's own frame (x right, y down, z forward)   <- default
      gl   the OpenGL camera frame, i.e. WCS without the 1 m shift
      wcs  the WCS proper (kept only so the failure is reproducible)
    """
    p = np.asarray(p, dtype=np.float64)
    if frame == "cam":
        return p
    if frame == "gl":
        return R.T @ p
    return to_cam(p, R, T)


def angle(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def head_to_2d(rvec):
    """dpc.HeadTo2d without its debugging print."""
    M = cv2.Rodrigues(rvec)[0]
    v = M[:, 2]
    return np.array([np.arctan2(v[0], v[2]), np.arcsin(v[1])])


def process(session, a, dpc, rows_ok, rows_lapse, report):
    sdir = os.path.join(a.src, session)
    need = ["rgb_vga.mov", "rgb_vga_calibration.txt", "head_pose.txt",
            "eye_tracking.txt", "screen_coordinates.txt"]
    miss = [f for f in need if not os.path.isfile(os.path.join(sdir, f))]
    if miss:
        sys.exit(f"ABORT [{session}]: missing {', '.join(miss)}")
    spath = os.path.join(a.state, session, "gaze_state.txt")
    if not os.path.isfile(spath):
        sys.exit(f"ABORT [{session}]: no gaze_state.txt -- refusing to emit frames "
                 f"whose gaze state is unknown.")

    K, R, T = read_calibration(os.path.join(sdir, "rgb_vga_calibration.txt"))
    _, head = read_table(os.path.join(sdir, "head_pose.txt"))
    _, eyes = read_table(os.path.join(sdir, "eye_tracking.txt"))
    shdr, scr = read_table(os.path.join(sdir, "screen_coordinates.txt"))
    state = read_state(spath)
    ncol = {len(v) for v in scr.values()}
    if ncol != {5}:
        sys.exit(f"ABORT [{session}]: screen_coordinates has {ncol} value columns, "
                 f"expected 5 (x_px, y_px, X, Y, Z). Header: {shdr!r}")

    # ---- geometry check: 3D eyeballs -> VGA pixels must match the file's own 2D
    err = []
    for f, e in list(eyes.items())[::25]:
        if len(e) < 18:
            continue
        for p3, p2 in ((e[12:15], e[0:2]), (e[15:18], e[2:4])):
            c = to_cam(p3, R, T)
            if c[2] <= 0:
                continue
            uv = (K @ c)[:2] / c[2]
            err.append(float(np.linalg.norm(uv - np.array(p2))))
    med = float(np.median(err)) if err else float("nan")
    if not err or med > a.reproj_tol:
        sys.exit(f"ABORT [{session}]: 3D eyeballs reproject {med:.1f} px (median) from "
                 f"the file's 2D VGA eyeballs. Our reading of R/T/units is wrong; "
                 f"no label is emitted from unverified geometry.")

    # ---- trajectories from the on-screen target
    frames = sorted(scr)
    traj, since, prev_end = {}, {}, {}
    t, start, last_target_prev = 0, frames[0], None
    for i, f in enumerate(frames):
        if i and np.hypot(scr[f][0] - scr[frames[i - 1]][0],
                          scr[f][1] - scr[frames[i - 1]][1]) > a.jump_px:
            t += 1
            start = f
            last_target_prev = np.array(scr[frames[i - 1]][2:5])
        traj[f], since[f], prev_end[f] = t, f - start, last_target_prev
    n_traj = t + 1

    # ---- video
    cap = cv2.VideoCapture(os.path.join(sdir, "rgb_vga.mov"))
    if not cap.isOpened():
        sys.exit(f"ABORT [{session}]: cannot open rgb_vga.mov")
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if abs(n_video - (max(state) + 1)) > 2:
        sys.exit(f"ABORT [{session}]: video reports {n_video} frames but gaze_state "
                 f"runs to {max(state)}. Frame indices would not line up.")

    pid = f"e{int(session.split('_')[0]):02d}"
    odir = os.path.join(a.out, "Image", pid, "face")
    ldir = os.path.join(a.out, "Image", pid, "lapse")
    if not a.dry:
        os.makedirs(odir, exist_ok=True); os.makedirs(ldir, exist_ok=True)

    cnt = {"OK": 0, "NL": 0, "BK": 0, "untracked": 0, "flipped": 0}
    f = -1
    while True:
        ok, im = cap.read()
        if not ok:
            break
        f += 1
        st = state.get(f)
        if st is None or f not in scr:
            continue
        keep_ok = (st == "OK" and f % a.stride == 0)
        keep_lapse = st in ("NL", "BK")
        if not (keep_ok or keep_lapse):
            continue
        if f not in eyes or f not in head or len(eyes[f]) < 18 or len(head[f]) < 12:
            cnt["untracked"] += 1
            continue
        e, h = eyes[f], head[f]
        eye_l, eye_r = to_cam(e[12:15], R, T), to_cam(e[15:18], R, T)
        if not (np.all(np.isfinite(eye_l)) and eye_l[2] > 0 and eye_r[2] > 0):
            cnt["untracked"] += 1
            continue
        center = 1000.0 * (eye_l + eye_r) / 2.0               # metres -> mm, as MPII
        target = 1000.0 * target_to_cam(scr[f][2:5], R, T, a.target_frame)
        Rh = R.T @ np.array(h[:9]).reshape(3, 3)

        nm = dpc.norm(center=center, gazetarget=target, headrotvec=Rh,
                      imsize=IMSIZE, camparams=K)
        if nm.R_mat[1, 1] <= 0:                                # must be upright
            cnt["flipped"] += 1
            continue
        if nm.GetGaze(scale=True)[2] >= 0:                     # must look TOWARD the camera
            sys.exit(f"ABORT [{session}] frame {f}: normalised gaze points away from "
                     f"the camera under --target-frame {a.target_frame}. The target's "
                     f"coordinate frame is being misread.")
        face = nm.GetImage(im)
        gaze = nm.GetGaze(scale=True)
        hvec = nm.GetHeadRot(vector=True)
        rvec, svec = nm.GetParams()
        g2, h2 = dpc.GazeTo2d(gaze), head_to_2d(hvec)
        origin = nm.GetCoordinate(center)

        lapse = float("nan")
        if prev_end[f] is not None:
            lapse = angle(target - center,
                          1000.0 * target_to_cam(prev_end[f], R, T, a.target_frame) - center)

        sub = "face" if keep_ok else "lapse"
        name = f"{pid}/{sub}/{session}_{f:05d}.jpg"
        if not a.dry:
            cv2.imwrite(os.path.join(a.out, "Image", name), face)
        row = " ".join([
            name, "NA", "NA", f"{session}/{f}", "both",
            ",".join(map(str, gaze)), ",".join(map(str, hvec)),
            ",".join(map(str, g2)), ",".join(map(str, h2)),
            ",".join(map(str, rvec)), ",".join(map(str, svec)),
            ",".join(map(str, origin)),
            session, str(f), st, str(traj[f]), str(since[f]),
            ("nan" if np.isnan(lapse) else f"{lapse:.4f}")])
        (rows_ok if keep_ok else rows_lapse).setdefault(pid, []).append(row)
        cnt[st] += 1
    cap.release()
    if f + 1 < max(state) - 2:
        sys.exit(f"ABORT [{session}]: decoder stopped at frame {f}, annotations run to "
                 f"{max(state)}. Check the video decoder.")
    report[session] = dict(cnt, reproj_px=round(med, 2), trajectories=n_traj,
                           video_frames=f + 1)
    print(f"  {session:<10} reproj {med:4.1f}px  traj {n_traj:3d}  "
          f"OK {cnt['OK']:4d}  NL {cnt['NL']:4d}  BK {cnt['BK']:4d}  "
          f"untracked {cnt['untracked']:3d}  not-upright {cnt['flipped']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="EYEDIAP/Data (session folders)")
    ap.add_argument("--state", required=True, help="GazeStateExport/Data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpc", default=DPC_DEFAULT, help="dir holding data_processing_core.py")
    ap.add_argument("--stride", type=int, default=15,
                    help="OK-frame stride; 15 is the GazeHub convention")
    ap.add_argument("--jump-px", type=float, default=150.0)
    ap.add_argument("--reproj-tol", type=float, default=5.0)
    ap.add_argument("--target-frame", choices=("cam", "gl", "wcs"), default="cam",
                    help="frame of screen_coordinates' 3D columns; see target_to_cam()")
    ap.add_argument("--only", default="", help="comma-separated sessions (testing)")
    ap.add_argument("--dry", action="store_true", help="no images or labels written")
    a = ap.parse_args()
    a.src, a.state, a.out = (os.path.abspath(os.path.expanduser(p))
                             for p in (a.src, a.state, a.out))

    if not os.path.isfile(os.path.join(a.dpc, "data_processing_core.py")):
        sys.exit(f"ABORT: data_processing_core.py not under {a.dpc}. The point of this "
                 f"build is to share MPIIFaceGaze_Normalized's normaliser; pass --dpc.")
    sys.path.insert(0, a.dpc)
    import data_processing_core as dpc

    sessions = sorted((d for d in os.listdir(a.src) if "_CS_" in d
                       and os.path.isdir(os.path.join(a.src, d))),
                      key=lambda s: (int(s.split("_")[0]), s))
    if a.only:
        sessions = [s for s in sessions if s in a.only.split(",")]
    if not sessions:
        sys.exit(f"ABORT: no CS sessions under {a.src}")
    print(f"[scan] {len(sessions)} CS sessions")

    rows_ok, rows_lapse, report = {}, {}, {}
    for s in sessions:
        process(s, a, dpc, rows_ok, rows_lapse, report)

    if a.dry:
        print("[dry] nothing written"); return 0
    for sub, rows in (("Label", rows_ok), ("Label_lapse", rows_lapse)):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
        for pid, rr in rows.items():
            with open(os.path.join(a.out, sub, f"{pid}.label"), "w") as fh:
                fh.write(HEADER + "\n" + "\n".join(rr) + "\n")
    json.dump({"sessions": report, "stride": a.stride, "target_frame": a.target_frame, "jump_px": a.jump_px,
               "subjects": sorted(rows_ok),
               "source": "EYEDIAP, Zenodo record 4467455; CS sessions; gaze state "
                         "from EYEDIAP_GazeStateAnnotations"},
              open(os.path.join(a.out, "build_report.json"), "w"), indent=2)
    ids = sorted(rows_ok)
    print(f"[done] {len(ids)} subjects, "
          f"{sum(len(v) for v in rows_ok.values())} OK rows, "
          f"{sum(len(v) for v in rows_lapse.values())} lapse rows -> {a.out}")
    print(f"[next] GAZE_DATA_ROOT={a.out} GAZE_SUBJECTS={','.join(ids)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
