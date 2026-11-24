# gaze_math.py
"""
Gaze geometry for the label-noise Phase-1 study. Pure numpy, no repo dependency.

NOTE (provenance): HANDOFF.md §5 lists this file as "DONE — written, verified".
No such file was present in the repo when Phase 1 was (re)built, so this module
was authored from the spec in §6 and §4.2. If a canonical gaze_math.py exists
elsewhere, prefer it — the public surface used downstream is:
    pitchyaw_to_vector, vector_to_pitchyaw, angular_error_deg, rotate_by_angle

Convention (HANDOFF §6, Eq. 2 of the DAP manuscript):

    v(alpha, beta) = (cos a * sin b,  sin a,  cos a * cos b)

with alpha = pitch, beta = yaw, both in RADIANS. Labels are carried as
(pitch, yaw) pairs in radians (the "pitchyaw" representation). Angular errors are
returned in DEGREES (Phase 1's severity axis is in degrees; convert to radians
only at the manuscript-reporting boundary).
"""
import numpy as np

RAD2DEG = 180.0 / np.pi
DEG2RAD = np.pi / 180.0

__all__ = [
    "RAD2DEG",
    "DEG2RAD",
    "pitchyaw_to_vector",
    "vector_to_pitchyaw",
    "angular_error_deg",
    "angular_error_vec_deg",
    "rotate_by_angle",
    "random_unit_pitchyaw",
]


def _as2(a):
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(-1, 2) if a.ndim == 1 else a


def pitchyaw_to_vector(pitchyaw):
    """
    (pitch, yaw) radians -> 3D unit gaze vector(s).

    Accepts shape (2,) or (N, 2); returns (3,) or (N, 3) accordingly.
        v = (cos(pitch) sin(yaw), sin(pitch), cos(pitch) cos(yaw))
    """
    a = np.asarray(pitchyaw, dtype=np.float64)
    single = a.ndim == 1
    a = _as2(a)
    pitch, yaw = a[:, 0], a[:, 1]
    cp = np.cos(pitch)
    v = np.stack([cp * np.sin(yaw), np.sin(pitch), cp * np.cos(yaw)], axis=1)
    return v[0] if single else v


def vector_to_pitchyaw(v):
    """
    3D gaze vector(s) -> (pitch, yaw) radians. Inverse of pitchyaw_to_vector.

    pitch = arcsin(v_y);  yaw = arctan2(v_x, v_z).
    """
    a = np.asarray(v, dtype=np.float64)
    single = a.ndim == 1
    a = a.reshape(-1, 3) if single else a
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    pitch = np.arcsin(np.clip(a[:, 1], -1.0, 1.0))
    yaw = np.arctan2(a[:, 0], a[:, 2])
    out = np.stack([pitch, yaw], axis=1)
    return out[0] if single else out


def angular_error_vec_deg(v1, v2):
    """Angle between gaze vectors, in degrees. Broadcasts over leading dim."""
    a = np.asarray(v1, dtype=np.float64)
    b = np.asarray(v2, dtype=np.float64)
    single = a.ndim == 1 and b.ndim == 1
    a = a.reshape(-1, 3)
    b = b.reshape(-1, 3)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    dot = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
    deg = np.arccos(dot) * RAD2DEG
    return float(deg[0]) if single else deg


def angular_error_deg(pitchyaw_a, pitchyaw_b):
    """Angle between two (pitch, yaw) labels, in degrees."""
    return angular_error_vec_deg(
        pitchyaw_to_vector(pitchyaw_a), pitchyaw_to_vector(pitchyaw_b)
    )


def _orthonormal_basis(v):
    """
    Two unit vectors e1, e2 spanning the plane orthogonal to unit vector v.
    Chooses a helper axis not parallel to v for numerical stability.
    """
    v = v / (np.linalg.norm(v) + 1e-12)
    helper = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(helper, v)) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(helper, v)
    e1 = e1 / (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(v, e1)          # already unit (v, e1 orthonormal)
    return e1, e2


def rotate_by_angle(pitchyaw, delta_deg, azimuth_rad=None, rng=None):
    """
    Rotate a (pitch, yaw) label by EXACTLY delta_deg, at a given (or random)
    azimuth within the plane orthogonal to the label vector.

    This yields an exact angular offset by construction:
        v' = cos(d) * v + sin(d) * (cos(az) * e1 + sin(az) * e2)
    where {e1, e2} is an orthonormal basis of the plane orthogonal to v, so
    angle(v, v') == d exactly (unlike naive per-component jitter, whose angular
    magnitude depends on the operating point).

    Parameters
    ----------
    pitchyaw : (2,) label in radians.
    delta_deg : rotation magnitude in degrees.
    azimuth_rad : fixed azimuth; if None, drawn uniformly from [0, 2pi) using rng.
    rng : numpy RandomState (required if azimuth_rad is None).

    Returns
    -------
    (new_pitchyaw, realised_offset_deg)
        realised_offset_deg == delta_deg up to float error; returned explicitly so
        callers log the *achieved* severity rather than the requested one.
    """
    v = pitchyaw_to_vector(pitchyaw)
    if azimuth_rad is None:
        if rng is None:
            raise ValueError("rotate_by_angle needs azimuth_rad or an rng.")
        azimuth_rad = rng.uniform(0.0, 2.0 * np.pi)
    e1, e2 = _orthonormal_basis(v)
    d = delta_deg * DEG2RAD
    axis = np.cos(azimuth_rad) * e1 + np.sin(azimuth_rad) * e2
    vp = np.cos(d) * v + np.sin(d) * axis
    vp = vp / (np.linalg.norm(vp) + 1e-12)
    new_py = vector_to_pitchyaw(vp)
    realised = angular_error_vec_deg(v, vp)
    return new_py, realised


def random_unit_pitchyaw(rng, pitch_range_deg=(-25, 25), yaw_range_deg=(-35, 35)):
    """Sample a plausible (pitch, yaw) label in radians (for synthetic subjects)."""
    pitch = rng.uniform(*pitch_range_deg) * DEG2RAD
    yaw = rng.uniform(*yaw_range_deg) * DEG2RAD
    return np.array([pitch, yaw], dtype=np.float64)
