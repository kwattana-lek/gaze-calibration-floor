"""
RQ1 (de-circularized, full-face-correct): does CQS detect corrupted calibration
frames when the evaluation is honest?

Corruption is defined INDEPENDENTLY by a full-face-aware simulator (known
ground-truth modes); CQS is scored as a detector against those labels.

Integrity guardrails
--------------------
* Sub-score normalizers are calibrated on CLEAN frames only (label-free), never
  tuned against the corruption labels:
    - sigma_max (sharpness) = a low percentile of the clean eye-region Laplacian
      variance, so q_sh spreads across [0,1] instead of saturating at 1.
    - head-pose lambda keeps its calibrated value; pose is estimated FROM THE
      IMAGE (MediaPipe + solvePnP), not from ground-truth annotations.
* Corruptions are geometrically correct for full faces: eye-related degradations
  hit the eye region (from landmarks), not the image bottom.

Run from repo root:
    PYTHONPATH=. python scripts/cqs_detection.py --fast
    PYTHONPATH=. python scripts/cqs_detection.py --n 300
"""
import os, sys, argparse
import numpy as np
import cv2

sys.path.insert(0, '.')
from src.cqs import q_landmark, q_sharpness, q_head_pose, _get_face_mesh
from src.config import CQS_W1, CQS_W2, CQS_W3

ROOT = os.environ.get("GAZE_DATA_ROOT", os.path.join("data", "MPIIFaceGaze_Normalized"))
SUBJECTS = [f'p{i:02d}' for i in range(15)]

MODEL_PTS = np.array([[0,0,0],[0,-63.6,-12.5],[-43.3,32.7,-26.0],
                      [43.3,32.7,-26.0],[-28.9,-28.9,-24.1],[28.9,-28.9,-24.1]], np.float64)
POSE_IDX = [1, 199, 263, 33, 291, 61]
OCULAR   = [33,7,163,144,145,153,362,382,381,380,374,373]


# ── data ──
def face_paths(subj):
    lines = open(os.path.join(ROOT, "Label", subj + ".label")).read().strip().split("\n")
    fi = lines[0].split().index("Face")
    return [ln.split()[fi] for ln in lines[1:]]

def load_img(face_rel):
    im = cv2.imread(os.path.join(ROOT, "Image", face_rel))
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im is not None else None

def landmarks_xy(image_rgb):
    res = _get_face_mesh().process(image_rgb)
    if not res.multi_face_landmarks:
        return None
    h, w = image_rgb.shape[:2]
    return np.array([[lm.x*w, lm.y*h] for lm in res.multi_face_landmarks[0].landmark])

def eye_bbox(pts, shape):
    h, w = shape[:2]
    if pts is None:
        return (w//4, h//6, 3*w//4, h//2)
    oc = np.array([pts[i] for i in OCULAR]); x1,y1 = oc.min(0); x2,y2 = oc.max(0)
    pad = 0.15*(x2-x1)
    return (max(0,int(x1-pad)),max(0,int(y1-pad)),min(w,int(x2+pad)),min(h,int(y2+pad)))

def head_pose(image_rgb, pts):
    if pts is None:
        return 90.,90.,90.
    h,w = image_rgb.shape[:2]
    img_pts = np.array([pts[i] for i in POSE_IDX], np.float64)
    cam = np.array([[w,0,w/2.],[0,w,h/2.],[0,0,1]], np.float64)
    ok,rvec,_ = cv2.solvePnP(MODEL_PTS,img_pts,cam,np.zeros((4,1)),flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return 90.,90.,90.
    R,_ = cv2.Rodrigues(rvec); sy = np.sqrt(R[0,0]**2+R[1,0]**2)
    pitch = np.degrees(np.arctan2(-R[2,0], sy))
    yaw   = np.degrees(np.arctan2(R[1,0], R[0,0]))
    roll  = ((np.degrees(np.arctan2(R[2,1],R[2,2]))+180)%360)-180
    return float(pitch),float(yaw),float(roll)


# ── full-face-aware corruption (independent ground truth) ──
def corrupt(im, pts, rng):
    """Apply one random corruption correctly for a full face. Returns (img, mode)."""
    m = rng.choice(['motion_blur','blink','occlusion','illumination','head_yaw','head_pitch'])
    img = im.copy(); h,w = img.shape[:2]
    ex1,ey1,ex2,ey2 = eye_bbox(pts, im.shape)
    if m == 'motion_blur':
        k = int(rng.integers(9,21))|1
        ker = np.zeros((k,k),np.float32); ker[k//2,:] = 1.0/k
        ang = float(rng.uniform(0,180))
        ker = cv2.warpAffine(ker, cv2.getRotationMatrix2D((k/2,k/2),ang,1), (k,k))
        img = cv2.filter2D(img,-1,ker)
    elif m == 'blink':                       # mask the EYE region (skin tone)
        img[ey1:ey2, ex1:ex2] = np.array([200,180,160],np.uint8)
    elif m == 'occlusion':                   # rectangle over one eye
        cx = ex1 + (0 if rng.random()<0.5 else (ex2-ex1)//2)
        img[ey1:ey2, cx:cx+max(8,(ex2-ex1)//2)] = 128
    elif m == 'illumination':
        g = float(rng.uniform(0.4,2.2))
        lut = ((np.arange(256)/255.)**(1/g)*255).clip(0,255).astype(np.uint8)
        img = lut[img]
    elif m in ('head_yaw','head_pitch'):
        d = int(rng.uniform(0.12,0.22)*(w if m=='head_yaw' else h))*(1 if rng.random()<.5 else -1)
        s = np.float32([[0,0],[w,0],[0,h],[w,h]])
        if m=='head_yaw': t = np.float32([[d,0],[w+d,0],[0,h],[w,h]])
        else:             t = np.float32([[0,d],[w,d],[0,h],[w,h]])
        img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(s,t), (w,h))
    return img, m


def subscores(image_rgb, sigma_max):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    pts  = landmarks_xy(image_rgb)
    qlm  = q_landmark(image_rgb)
    qsh  = q_sharpness(gray, eye_bbox(pts, image_rgb.shape), sigma_max=sigma_max)
    p,y,r = head_pose(image_rgb, pts); qhp = q_head_pose(p,y,r)
    cqs = (qlm**CQS_W1)*(qsh**CQS_W2)*(qhp**CQS_W3)
    return float(cqs),qlm,qsh,qhp


# ── metrics ──
def roc_auc(score,label):
    score,label = np.asarray(score),np.asarray(label)
    order=np.argsort(score); ranks=np.empty_like(order,float); ranks[order]=np.arange(1,len(score)+1)
    npos,nneg=label.sum(),(1-label).sum()
    return float('nan') if npos==0 or nneg==0 else float((ranks[label==1].sum()-npos*(npos+1)/2)/(npos*nneg))

def pr_auc(score,label):
    order=np.argsort(-np.asarray(score)); lab=np.asarray(label)[order]
    tp=np.cumsum(lab); fp=np.cumsum(1-lab); prec=tp/(tp+fp); rec=tp/max(lab.sum(),1)
    ap=0.; prev=0.
    for p,r in zip(prec,rec): ap+=p*(r-prev); prev=r
    return float(ap)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--n',type=int,default=300)
    ap.add_argument('--subjects',nargs='+',default=SUBJECTS)
    ap.add_argument('--sigma_pct',type=float,default=25.0,help='clean-percentile for sigma_max')
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--fast',action='store_true')
    a=ap.parse_args()
    if a.fast: a.n,a.subjects = 60,['p00','p01','p02']
    rng=np.random.default_rng(a.seed)

    # gather clean frames + landmarks
    clean=[]
    for s in a.subjects:
        paths=face_paths(s); idx=rng.choice(len(paths),size=min(a.n,len(paths)),replace=False)
        for j in idx:
            im=load_img(paths[j])
            if im is None: continue
            clean.append((im, landmarks_xy(im)))
        print(f"  loaded {s}: {len(clean)} clean frames total")

    # ── clean-only calibration of sigma_max (label-free) ──
    lap=[]
    for im,pts in clean:
        g=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY); x1,y1,x2,y2=eye_bbox(pts,im.shape)
        p=g[y1:y2,x1:x2]
        if p.size: lap.append(cv2.Laplacian(p,cv2.CV_64F).var())
    sigma_max=float(np.percentile(lap, a.sigma_pct))
    print(f"\nCalibrated sigma_max = {sigma_max:.1f} "
          f"(={a.sigma_pct:.0f}th pct of clean eye-Laplacian; was 40.0)\n")

    # ── build clean/corrupt scored set ──
    rows=[]  # (cqs,qlm,qsh,qhp,label)
    modes=[]
    for im,pts in clean:
        rows.append((*subscores(im,sigma_max),0)); modes.append('clean')
        ci,mode=corrupt(im,pts,rng)
        rows.append((*subscores(ci,sigma_max),1)); modes.append(mode)

    R=np.array([[r[0],r[1],r[2],r[3],r[4]] for r in rows],float)
    cqs,qlm,qsh,qhp,lab=R.T; inv=lambda x:1.0-x

    print("================ CQS DETECTION (non-circular, full-face) ================")
    print(f"n={len(lab)}  ({int((lab==0).sum())} clean, {int((lab==1).sum())} corrupt)")
    print(f"{'signal':16s}{'ROC-AUC':>9s}{'PR-AUC':>9s}{'mean clean':>12s}{'mean corrupt':>14s}")
    for name,sig in [('CQS composite',cqs),('q_landmark',qlm),('q_sharpness',qsh),('q_headpose',qhp)]:
        mc=sig[lab==0].mean(); mk=sig[lab==1].mean()
        print(f"{name:16s}{roc_auc(inv(sig),lab):9.3f}{pr_auc(inv(sig),lab):9.3f}{mc:12.3f}{mk:14.3f}")

    pred=(cqs<0.60).astype(int)
    tp=int(((pred==1)&(lab==1)).sum()); fp=int(((pred==1)&(lab==0)).sum()); fn=int(((pred==0)&(lab==1)).sum())
    prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1); f1=2*prec*rec/max(prec+rec,1e-9)
    print(f"\ntau=0.60:  precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}")
    print("\nPer-mode recall @ tau=0.60:")
    for mm in sorted(set(m for m in modes if m!='clean')):
        mi=[i for i,x in enumerate(modes) if x==mm]
        print(f"  {mm:14s} n={len(mi):4d}  recall={(cqs[mi]<0.60).mean():.3f}")
    print("========================================================================")


if __name__=='__main__':
    main()
