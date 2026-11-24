#!/usr/bin/env python3
# check_eyediap_target_frame.py -- which frame are EYEDIAP's screen-target 3D columns in?
"""WHY THIS EXISTS

prepare_eyediap_labels.py verifies the eyeball/head geometry by reprojection. It
cannot verify the screen target that way: the target is not visible to the camera.
Read as WCS the target lands behind the participant (rejected outright: the
normalised gaze points away from the camera). Two camera-centred readings remain
and they differ by ~30 deg of pitch:

  cam  VGA camera frame, y down  -> screen ABOVE the camera, gaze roughly level
  gl   OpenGL camera frame, y up -> screen BELOW the camera, gaze steeply down

An independent witness decides: UniGaze h14_cross_X was trained on ETH-XGaze
alone and has never seen EYEDIAP, so its predictions cannot be biased toward either
reading. Whichever build's labels it agrees with is the right one. Its output
convention (pitch, yaw) is not assumed either: it is read off MPIIFaceGaze, where
the labels are known good, before it is used.

(An MPII-trained ResNet-18 was tried first and is NOT a usable witness: on EYEDIAP
its pitch output is nearly constant -- correlation with either labelling ~0 -- so
it cannot discriminate. Recorded so nobody retries it. Geometry agrees with the
UniGaze verdict independently: under `cam` the gaze sits 4 deg from the head
normal on average, under `gl` 31 deg below it for a whole session.)

Reads two builds of the SAME frames (made with --target-frame cam / gl). No
training, no writes.

Usage:
  python3 check_eyediap_target_frame.py --cam <build_cam> --gl <build_gl> [--subject e01]
"""
import argparse, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


def load(root, sid, size=112):
    import cv2
    L = open(os.path.join(root, "Label", f"{sid}.label")).read().strip().split("\n")
    h = L[0].split()
    fi, gi = h.index("Face"), h.index("3DGaze")
    ims, vec, names = [], [], []
    for ln in L[1:]:
        p = ln.split()
        im = cv2.imread(os.path.join(root, "Image", p[fi]))
        ims.append(cv2.cvtColor(cv2.resize(im, (size, size)), cv2.COLOR_BGR2RGB))
        vec.append([float(v) for v in p[gi].split(",")])
        names.append(p[fi])
    return np.stack(ims), np.array(vec), names


def ang(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip((a * b).sum(1), -1, 1)))


def pitchyaw(v):
    return np.degrees(np.c_[np.arcsin(-v[:, 1]), np.arctan2(-v[:, 0], -v[:, 2])])


def lab2d(root, sid):
    L = open(os.path.join(root, "Label", f"{sid}.label")).read().strip().split("\n")
    gi = L[0].split().index("2DGaze")                      # (yaw, pitch), radians
    return np.array([[float(v) for v in l.split()[gi].split(",")] for l in L[1:]])


def yp_to_vec(yp):
    y, p = yp[:, 0], yp[:, 1]
    return np.c_[-np.cos(p) * np.sin(y), -np.sin(p), -np.cos(p) * np.cos(y)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", required=True)
    ap.add_argument("--gl", required=True)
    ap.add_argument("--subject", default="e01")
    ap.add_argument("--ref", default=os.path.expanduser(
        "~/Desktop/Documents Since 29-01-2569/Datasets/MPIIFaceGaze_Normalized"))
    a = ap.parse_args()
    import torch, unigaze, unigaze_arm as U
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = unigaze.load(U.MODEL, device=str(dev)); model.eval()

    def pred(X):                                           # same input path as the arm
        out = []
        for i in range(0, len(X), 8):
            x = torch.from_numpy(X[i:i + 8]).permute(0, 3, 1, 2).float().div(255).to(dev)
            with torch.no_grad():
                out.append(model(x)["pred_gaze"].float().cpu().numpy())
        return np.concatenate(out)

    # 1. learn the witness's column order on labels known to be right
    Xm, _, _ = load(a.ref, "p03", size=224)
    Pm, Lm = pred(Xm[:80]), lab2d(a.ref, "p03")[:80]
    cm = np.array([[np.corrcoef(Lm[:, i], Pm[:, j])[0, 1] for j in range(2)] for i in range(2)])
    order = [int(np.argmax(np.abs(cm[0]))), int(np.argmax(np.abs(cm[1])))]   # pred col for yaw, pitch
    if sorted(order) != [0, 1] or min(abs(cm[0, order[0]]), abs(cm[1, order[1]])) < 0.8:
        sys.exit(f"ABORT: cannot read the witness's output convention off MPII:\n{cm.round(2)}")
    fix = lambda P: P[:, order] * np.sign([cm[0, order[0]], cm[1, order[1]]])
    e_ref = ang(yp_to_vec(fix(Pm)), yp_to_vec(Lm)).mean()
    print(f"witness on MPIIFaceGaze p03: {e_ref:.2f} deg  (pred cols for yaw,pitch = {order})")

    # 2. judge the two EYEDIAP builds
    Xe, _, names = load(a.cam, a.subject, size=224)
    if load(a.gl, a.subject, size=224)[2] != names:
        sys.exit("ABORT: the two builds do not hold the same frames.")
    Pe = fix(pred(Xe))
    res = {}
    for k, root in (("cam", a.cam), ("gl", a.gl)):
        L = lab2d(root, a.subject)
        res[k] = float(ang(yp_to_vec(Pe), yp_to_vec(L)).mean())
        print(f"  {k:3s}: error {res[k]:5.1f} deg | corr yaw {np.corrcoef(L[:,0], Pe[:,0])[0,1]:+.2f} "
              f"pitch {np.corrcoef(L[:,1], Pe[:,1])[0,1]:+.2f} | mean pitch label "
              f"{np.degrees(L[:,1].mean()):+6.1f} witness {np.degrees(Pe[:,1].mean()):+6.1f}")
    c, g = res["cam"], res["gl"]
    if min(c, g) > 15 or abs(c - g) < 8:
        print("VERDICT: UNDECIDED -- neither reading is clearly supported. Do not build.")
        return 1
    print(f"VERDICT: --target-frame {'cam' if c < g else 'gl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
