"""Direction cosine matrices, ``v_I = C @ v_B`` (BODY -> inertial)."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from orbital.attitude.quaternion import FloatArray, canonical, normalize


def rot_x(angle_rad: float) -> FloatArray:
    """Active right-handed rotation about +x by ``angle_rad``."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle_rad: float) -> FloatArray:
    """Active right-handed rotation about +y by ``angle_rad``."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle_rad: float) -> FloatArray:
    """Active right-handed rotation about +z by ``angle_rad``."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def is_rotation_matrix(c: ArrayLike, tol: float = 1e-9) -> bool:
    """True if ``c`` is orthogonal with determinant +1 to within ``tol``."""
    m = np.asarray(c, dtype=float)
    return (
        m.shape == (3, 3)
        and bool(np.allclose(m.T @ m, np.eye(3), atol=tol))
        and abs(float(np.linalg.det(m)) - 1.0) < tol
    )


def to_dcm(q: ArrayLike) -> FloatArray:
    """DCM (BODY -> inertial) from a quaternion. ``q`` is normalised first."""
    w, x, y, z = normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def from_dcm(c: ArrayLike) -> FloatArray:
    """Quaternion from a DCM (BODY -> inertial), canonical sign ``w >= 0``.

    Shepperd's method: use the extraction with the largest pivot, which stays
    accurate near 180 degrees.

    Raises
    ------
    ValueError
        If ``c`` is not a proper rotation matrix.
    """
    m = np.asarray(c, dtype=float)
    if not is_rotation_matrix(m, tol=1e-6):
        raise ValueError("input is not a proper rotation matrix")
    tr = np.trace(m)
    candidates = (tr, m[0, 0], m[1, 1], m[2, 2])
    k = int(np.argmax(candidates))
    if k == 0:
        w = 0.5 * np.sqrt(1.0 + tr)
        f = 0.25 / w
        q = [w, (m[2, 1] - m[1, 2]) * f, (m[0, 2] - m[2, 0]) * f, (m[1, 0] - m[0, 1]) * f]
    elif k == 1:
        x = 0.5 * np.sqrt(1.0 + 2.0 * m[0, 0] - tr)
        f = 0.25 / x
        q = [(m[2, 1] - m[1, 2]) * f, x, (m[1, 0] + m[0, 1]) * f, (m[0, 2] + m[2, 0]) * f]
    elif k == 2:
        y = 0.5 * np.sqrt(1.0 + 2.0 * m[1, 1] - tr)
        f = 0.25 / y
        q = [(m[0, 2] - m[2, 0]) * f, (m[1, 0] + m[0, 1]) * f, y, (m[2, 1] + m[1, 2]) * f]
    else:
        z = 0.5 * np.sqrt(1.0 + 2.0 * m[2, 2] - tr)
        f = 0.25 / z
        q = [(m[1, 0] - m[0, 1]) * f, (m[0, 2] + m[2, 0]) * f, (m[2, 1] + m[1, 2]) * f, z]
    return canonical(normalize(q))
