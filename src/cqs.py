"""
Calibration Quality Score (CQS)
Eq. 4 in the paper: CQS = q_lm^w1 * q_sh^w2 * q_hp^w3
"""

import os
os.environ['GLOG_minloglevel'] = '3'    # suppress MediaPipe INFO logs

import cv2
import numpy as np
import mediapipe as mp
from src.config import CQS_LAMBDA, CQS_SIGMA_MAX, CQS_W1, CQS_W2, CQS_W3

# 12 key periocular landmarks from MediaPipe Face Mesh
OCULAR_IDX = [33, 7, 163, 144, 145, 153,   # left eye
              362, 382, 381, 380, 374, 373]  # right eye


def _is_valid_image(image_rgb):
    """Fast pre-check: skip obviously bad frames before MediaPipe."""
    if image_rgb is None or image_rgb.size == 0:
        return False
    gray = image_rgb.mean(axis=2) if image_rgb.ndim == 3 else image_rgb
    std  = float(gray.std())
    mean = float(gray.mean())
    if std < 5.0 or mean < 10.0 or mean > 245.0:
        return False
    return True


_face_mesh = None

def _get_face_mesh():
    global _face_mesh
    if _face_mesh is None:
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
    return _face_mesh


def q_landmark(image_rgb: np.ndarray) -> float:
    """
    q_lm: landmark alignment confidence.
    MediaPipe FaceMesh visibility is always 0.0 for face landmarks.
    Instead we use:
      - Detection success (0.0 if no face found)
      - Landmark coordinate plausibility: check that left/right eye
        centroids are in the expected horizontal halves of the image
      - Inter-ocular distance sanity: eyes should not be too close
        together (blur/occlusion indicator)
    Returns float in [0, 1].
    """
    if not _is_valid_image(image_rgb):
        return 0.0
    if not _is_valid_image(image_rgb):
        return 0.0
    if not _is_valid_image(image_rgb):
        return 0.0
    results = _get_face_mesh().process(image_rgb)
    if not results.multi_face_landmarks:
        return 0.0

    h, w = image_rgb.shape[:2]
    lms  = results.multi_face_landmarks[0].landmark

    # Left eye centre (landmarks 33,133,159,145,153,144)
    LEFT_EYE  = [33, 133, 159, 145, 153, 144]
    RIGHT_EYE = [362, 263, 386, 374, 380, 373]

    lx = np.mean([lms[i].x for i in LEFT_EYE])   # normalised [0,1]
    rx = np.mean([lms[i].x for i in RIGHT_EYE])
    ly = np.mean([lms[i].y for i in LEFT_EYE])
    ry = np.mean([lms[i].y for i in RIGHT_EYE])

    # ── Check 1: left eye should be left of right eye (x < rx) ──
    # (MediaPipe x=0 is image left)
    if lx >= rx:
        lateral_score = 0.3   # flipped / extreme pose
    else:
        lateral_score = 1.0

    # ── Check 2: inter-ocular distance (normalised by image width) ──
    iod = abs(rx - lx)
    # Healthy range: 0.15 (far away) to 0.55 (very close)
    if iod < 0.03:
        iod_score = 0.0    # eyes too close — severe blur or occlusion
    elif iod < 0.08:
        iod_score = iod / 0.08
    elif iod > 0.55:
        iod_score = max(0.0, 1.0 - (iod - 0.55) / 0.2)
    else:
        iod_score = 1.0

    # ── Check 3: eyes should be in upper half of face bounding box ──
    NOSE_TIP = 1
    nose_y = lms[NOSE_TIP].y
    eye_y   = (ly + ry) / 2.0
    if eye_y < nose_y:
        vertical_score = 1.0   # eyes above nose — correct
    else:
        vertical_score = 0.2   # eyes below nose — extreme pitch or error

    q = lateral_score * iod_score * vertical_score
    return float(np.clip(q, 0.0, 1.0))


def q_sharpness(image_gray: np.ndarray, eye_bbox: tuple,
               sigma_max: float = None) -> float:
    """Eq. 2: normalised Laplacian variance of eye patch."""
    x1, y1, x2, y2 = eye_bbox
    patch = image_gray[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    smax    = sigma_max if sigma_max is not None else CQS_SIGMA_MAX
    lap_var = cv2.Laplacian(patch, cv2.CV_64F).var()
    return float(min(lap_var / smax, 1.0))


def q_head_pose(pitch: float, yaw: float, roll: float) -> float:
    """Eq. 3: Gaussian penalty on head pose deviation (degrees)."""
    return float(np.exp(-CQS_LAMBDA * (yaw**2 + pitch**2 + roll**2)))


def compute_cqs(
    image_rgb: np.ndarray,
    image_gray: np.ndarray,
    eye_bbox: tuple,
    pitch: float,
    yaw: float,
    roll: float,
    w1: float = CQS_W1,
    w2: float = CQS_W2,
    w3: float = CQS_W3,
    skip_landmark: bool = False,
    sigma_max: float = None,
) -> tuple:
    """
    Returns (CQS, q_lm, q_sh, q_hp).
    CQS = q_lm^w1 * q_sh^w2 * q_hp^w3

    skip_landmark: set True for eye-patch datasets (MPIIGaze)
                   where no full face is present for MediaPipe.
                   q_lm is set to 1.0 and its weight redistributed.
    """
    ql = 1.0 if skip_landmark else q_landmark(image_rgb)
    qs = q_sharpness(image_gray, eye_bbox, sigma_max=sigma_max if sigma_max is not None else CQS_SIGMA_MAX)
    qh = q_head_pose(pitch, yaw, roll)
    if skip_landmark:
        # Redistribute w1 to w2 and w3 proportionally
        total = w2 + w3
        w2_eff = w2 / total if total > 0 else 0.5
        w3_eff = w3 / total if total > 0 else 0.5
        cqs = (qs ** w2_eff) * (qh ** w3_eff)
    else:
        cqs = (ql ** w1) * (qs ** w2) * (qh ** w3)
    return float(cqs), float(ql), float(qs), float(qh)
