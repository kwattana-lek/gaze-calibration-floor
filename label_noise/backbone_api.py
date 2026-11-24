# backbone_api.py  (agentgaze 112px port)
"""
Phase-1 backbone facade, ported to run INSIDE the ES 112px MPIIFaceGaze pipeline.

Everything is delegated to `scripts/train_backbone.py` (the loader/model/metric
that `run_lopo_backbone.py` reuses) and reproduces `run_lopo_backbone.calibrate`'s
K=9 BatchNorm-frozen calibration recipe. The other Phase-1 modules
(corruption.py, loo_residual.py, run_phase1.py) are unchanged.

Key differences from the earlier ResNet18Gaze port
---------------------------------------------------
* Input: 112x112 full-FACE crop (train_backbone.IMG), ImageNet-normalized.
* Labels: **3DGaze unit vectors** (GazeHub `3DGaze` column, SIGN-corrected and
  L2-normalized in train_backbone.load_rows). The gaze head is `Linear(512, 3)`
  and `predict()` returns 3-D unit vectors directly.
* **No axis bridge.** There is no (pitch,yaw)<->(yaw,pitch) swap. The only
  conversion is angle<->vector via gaze_math: the Phase-1 modules carry labels as
  (pitch, yaw) purely so corruption.rotate_by_angle can express an *exact*
  angular offset; `adapt` maps them straight back to unit vectors with
  gaze_math.pitchyaw_to_vector (an exact round-trip: vec->pitchyaw->vec == vec),
  so the target handed to the net is the original 3DGaze vector, un-permuted.
* Calibration is BN-frozen (`m.eval()`), matching run_lopo_backbone.calibrate;
  this is the K=9 protocol that produced ES Table 4's 3.47 deg (mode "full").
"""
import copy
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# --- resolve paths: label_noise/ lives at agentgaze/label_noise ---------------
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))       # .../agentgaze/label_noise
_AG_ROOT = os.path.dirname(_PKG_DIR)                          # .../agentgaze
_SCRIPTS = os.path.join(_AG_ROOT, "scripts")
for _p in (_AG_ROOT, _SCRIPTS, _PKG_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gaze_math as gm

# `train_backbone` (loader/model/metric, shared with run_lopo_backbone) is
# imported LAZILY: it pulls in cv2/torchvision, which the synthetic path and the
# pure-logic modules (corruption/loo/run_phase1 synthetic mode) must not require.
_TB = None
def _tb():
    global _TB
    if _TB is None:
        import train_backbone as t
        _TB = t
    return _TB

try:
    from src.config import DEVICE          # cv2-free
except Exception:                          # pragma: no cover - fallback
    DEVICE = "mps" if getattr(getattr(torch, "backends", None), "mps", None) \
        and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

CONVENTION = "MPIIFaceGaze/3DGaze-unitvec"
DEFAULT_IMG_SIZE = 112               # == train_backbone.IMG
DEFAULT_SEED = 42

# K=9 BN-frozen calibration hyperparameters (run_lopo_backbone "full" recipe).
DEFAULT_STEPS = 20
DEFAULT_LR = 1e-5
DEFAULT_WD = 1e-4
CALIB_MODE = "full"                  # "full" (all params) or "fc" (head only)
DEFAULT_ACTIVE_LAYERS = None         # unused here (GazeNet has no layered adapters)

_MODELS_DIR = os.path.join(_AG_ROOT, "models")

__all__ = [
    "CONVENTION", "DEFAULT_IMG_SIZE", "DEFAULT_SEED",
    "DEFAULT_STEPS", "DEFAULT_LR", "DEFAULT_WD", "DEFAULT_ACTIVE_LAYERS",
    "get_device", "set_seed", "calibration_split", "calibration_ids",
    "load_backbone", "load_subject_support", "adapt", "predict",
    "make_synthetic_backbone", "SyntheticBackbone",
]


def get_device():
    return DEVICE


def set_seed(seed: int = DEFAULT_SEED):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# K-shot calibration split (same prefix-nested semantics as the DAP split)
# ---------------------------------------------------------------------------
def calibration_split(n_items: int, k_shots: int = 9,
                      seed: int = DEFAULT_SEED) -> Tuple[List[int], List[int]]:
    rng = np.random.RandomState(seed)
    idx = np.arange(n_items)
    rng.shuffle(idx)
    k = min(k_shots, n_items)
    return idx[:k].tolist(), idx[k:].tolist()


def calibration_ids(n_items: int, k_shots: int = 9, seed: int = DEFAULT_SEED) -> List[int]:
    return calibration_split(n_items, k_shots, seed=seed)[0]


# ---------------------------------------------------------------------------
# Backbone loading (per-LOPO-fold cross-subject GazeNet)
# ---------------------------------------------------------------------------
def load_backbone(subject_id: str, seed: int = DEFAULT_SEED,
                  ckpt_path: Optional[str] = None, device=None):
    """
    Load the cross-subject backbone for held-out `subject_id`, i.e.
    models/backbone_mpiifacegaze_<subject_id>.pt (trained by run_lopo_backbone /
    train_backbone on the other 14 subjects). Returns an eval-mode GazeNet.

    We do NOT train on the fly here (that is the slow path and out of scope for
    the port); a missing checkpoint is a clear error.
    """
    device = device or DEVICE
    if ckpt_path is None:
        ckpt_path = os.path.join(_MODELS_DIR, f"backbone_mpiifacegaze_{subject_id}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No backbone checkpoint for {subject_id}: {ckpt_path}. "
            f"Train it with scripts/run_lopo_backbone.py (or train_backbone.py).")
    model = _tb().build_model().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Subject support loading (K=9 frames + full label pool), 3DGaze unit vectors
# ---------------------------------------------------------------------------
def load_subject_support(subject_id: str, K: int = 9, seed: int = DEFAULT_SEED,
                         cgd_root: Optional[str] = None, img_size: int = None,
                         device=None) -> Dict:
    """
    Load one subject's K support frames + full label pool from MPIIFaceGaze
    (train_backbone.load_images -> 112px face crops + 3DGaze unit vectors).

    Returns dict:
        images       : (K, 112, 112, 3) uint8 RGB support frames
        labels       : (K, 2) (pitch, yaw) radians   -- angular chart of the
                       support 3DGaze vectors (see module docstring; NOT an axis
                       swap). corruption/loo operate in this chart.
        pool_labels  : (M, 2) (pitch, yaw) radians    -- all frames (donor pool)
        support_ids  : list[int]

    `cgd_root`/`img_size` are accepted for signature parity with run_phase1's
    generic loader but ignored: the dataset root is train_backbone.ROOT and the
    resolution is fixed at 112.
    """
    rng = np.random.default_rng(seed)
    imgs, vecs = _tb().load_images(subject_id, cap=None, rng=rng)   # (N,112,112,3), (N,3)
    n = len(imgs)
    support_ids, _ = calibration_split(n, K, seed=seed)
    support_ids = np.asarray(support_ids)
    return {
        "images": imgs[support_ids],
        "labels": gm.vector_to_pitchyaw(vecs[support_ids]).reshape(-1, 2),
        "pool_labels": gm.vector_to_pitchyaw(vecs).reshape(-1, 2),
        "support_ids": support_ids.tolist(),
    }


# ---------------------------------------------------------------------------
# predict / adapt  (dispatch: real GazeNet vs SyntheticBackbone)
# ---------------------------------------------------------------------------
def _is_synthetic(model) -> bool:
    return isinstance(model, SyntheticBackbone)


def predict(model, images) -> np.ndarray:
    """
    Predict gaze UNIT VECTORS (N, 3). Real path: GazeNet forward on 112px frames,
    L2-normalized -- no axis permutation. Synthetic path: delegates.
    """
    if _is_synthetic(model):
        return model.predict(images)
    model.eval()
    device = next(model.parameters()).device
    x = _tb().to_batch(np.asarray(images), device)
    with torch.no_grad():
        v = F.normalize(model(x), dim=1).cpu().numpy()
    return v


def adapt(model, images, labels, steps: int = DEFAULT_STEPS, lr: float = DEFAULT_LR,
          active_layers=DEFAULT_ACTIVE_LAYERS, wd: float = DEFAULT_WD,
          mode: str = CALIB_MODE, freeze_bn: bool = True):
    """
    K=9 BN-frozen per-subject calibration, returning an ADAPTED COPY (so LOO folds
    stay independent). Reproduces run_lopo_backbone.calibrate exactly, except it
    returns the model instead of the MAE.

    labels : (N, 2) (pitch, yaw) -> mapped straight to 3DGaze unit-vector targets
             via gaze_math.pitchyaw_to_vector (exact round-trip; no axis bridge).
    `active_layers` is ignored (GazeNet has no layered adapters); `mode` selects
    "full" (all params, the Table-4 recipe) or "fc" (head only).
    """
    labels = np.asarray(labels, dtype=np.float64).reshape(-1, 2)
    if _is_synthetic(model):
        return model.adapt(images, labels, steps=steps, lr=lr)

    device = next(model.parameters()).device
    m = copy.deepcopy(model)
    m.eval()                                   # BN frozen (run_lopo_backbone recipe)
    if mode == "fc":
        for p in m.parameters(): p.requires_grad = False
        for p in m.fc.parameters(): p.requires_grad = True
    else:
        for p in m.parameters(): p.requires_grad = True
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)

    tb = _tb()
    xc = tb.to_batch(np.asarray(images), device)
    yc = torch.from_numpy(gm.pitchyaw_to_vector(labels).astype(np.float32)).to(device)
    for _ in range(int(steps)):
        opt.zero_grad(); tb.angular_loss(m(xc), yc).backward(); opt.step()
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Synthetic backbone -- lets run_phase1 / the smoke test run with no data or
# checkpoint. Representation-agnostic (unchanged from the DAP port).
# ---------------------------------------------------------------------------
class SyntheticBackbone:
    """"Images" are per-frame TRUE gaze unit vectors (N,3); predicts truth + an
    additive (pitch,yaw) bias fit on the support labels, so support contamination
    inflates clean-frame residuals (the §4.4 confound), exercising the plumbing."""

    def __init__(self, bias=None):
        self.bias = np.zeros(2, dtype=np.float64) if bias is None else np.asarray(bias, float)

    @staticmethod
    def _truth_pitchyaw(images) -> np.ndarray:
        v = images.numpy() if torch.is_tensor(images) else np.asarray(images, float)
        return gm.vector_to_pitchyaw(v.reshape(-1, 3)).reshape(-1, 2)

    def predict(self, images) -> np.ndarray:
        truth = self._truth_pitchyaw(images)
        return gm.pitchyaw_to_vector(truth + self.bias[None, :])

    def adapt(self, images, labels, steps: int = DEFAULT_STEPS, lr: float = DEFAULT_LR):
        truth = self._truth_pitchyaw(images)
        labels = np.asarray(labels, dtype=np.float64).reshape(-1, 2)
        return SyntheticBackbone(bias=(labels - truth).mean(axis=0))


def make_synthetic_backbone(bias=None) -> SyntheticBackbone:
    return SyntheticBackbone(bias=bias)
