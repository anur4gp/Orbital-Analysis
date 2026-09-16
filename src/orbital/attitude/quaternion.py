"""Quaternion algebra and attitude kinematics.

A unit quaternion ``q = [w, x, y, z]`` (scalar first, Hamilton product)
represents the rotation that takes a vector expressed in BODY axes to the same
vector expressed in the inertial frame:

    v_I = q ⊗ [0, v_B] ⊗ q*

With the body angular velocity ``omega_B`` (rad/s, BODY axes, relative to
inertial), the kinematics are

    dq/dt = 1/2 q ⊗ [0, omega_B]

That equation preserves ``|q|`` exactly, but a numerical integrator does not:
each step leaks a truncation error into the norm. The drift is handled by the
integrators' constraint projection (see :mod:`orbital.integrators.base`), not
hidden here -- :func:`kinematics` integrates whatever it is given, so the drift
stays measurable.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

IDENTITY: FloatArray = np.array([1.0, 0.0, 0.0, 0.0])


def as_quaternion(q: ArrayLike) -> FloatArray:
    """Validate shape and return ``q`` as a float array of length 4."""
    arr = np.asarray(q, dtype=float)
    if arr.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {arr.shape}")
    return arr


def norm_error(q: ArrayLike) -> float:
    """Unit-norm constraint violation ``| |q| - 1 |``, dimensionless."""
    return abs(float(np.linalg.norm(as_quaternion(q))) - 1.0)


def normalize(q: ArrayLike) -> FloatArray:
    """Project ``q`` onto the unit sphere.

    Raises
    ------
    ValueError
        If ``q`` is (numerically) zero, which has no meaningful direction.
    """
    arr = as_quaternion(q)
    n = float(np.linalg.norm(arr))
    if n < 1e-12:
        raise ValueError("cannot normalise a zero quaternion")
    return arr / n


def canonical(q: ArrayLike) -> FloatArray:
    """Return the representative of ``{q, -q}`` with ``w >= 0``.

    ``q`` and ``-q`` are the same rotation; comparisons should use this form.
    """
    arr = as_quaternion(q)
    return -arr if arr[0] < 0.0 else arr


def conjugate(q: ArrayLike) -> FloatArray:
    """Quaternion conjugate ``[w, -x, -y, -z]`` (the inverse, for unit ``q``)."""
    w, x, y, z = as_quaternion(q)
    return np.array([w, -x, -y, -z])


def multiply(p: ArrayLike, q: ArrayLike) -> FloatArray:
    """Hamilton product ``p ⊗ q``.

    Composition order: ``p ⊗ q`` applies ``q`` first, then ``p``.
    """
    pw, px, py, pz = as_quaternion(p)
    qw, qx, qy, qz = as_quaternion(q)
    return np.array([
        pw * qw - px * qx - py * qy - pz * qz,
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
    ])


def from_axis_angle(axis: ArrayLike, angle_rad: float) -> FloatArray:
    """Unit quaternion for a right-handed rotation of ``angle_rad`` about ``axis``.

    Parameters
    ----------
    axis
        Rotation axis, any non-zero length (normalised here).
    angle_rad
        Rotation angle, rad.
    """
    a = np.asarray(axis, dtype=float)
    n = float(np.linalg.norm(a))
    if a.shape != (3,) or n < 1e-12:
        raise ValueError("axis must be a non-zero 3-vector")
    half = 0.5 * angle_rad
    return np.concatenate([[np.cos(half)], np.sin(half) * a / n])


def rotation_angle(q: ArrayLike) -> float:
    """Magnitude of the rotation represented by ``q``, rad, in ``[0, pi]``.

    Uses ``atan2`` rather than ``acos(w)``, which loses precision near zero
    -- exactly where attitude-error comparisons live.
    """
    c = canonical(normalize(q))
    return 2.0 * float(np.arctan2(np.linalg.norm(c[1:]), c[0]))


def rotate(q: ArrayLike, v_body: ArrayLike) -> FloatArray:
    """Express a BODY-frame vector in the inertial frame: ``q ⊗ v ⊗ q*``.

    ``q`` must be unit norm; units of the result are those of ``v_body``.
    """
    v = np.asarray(v_body, dtype=float)
    qv = np.concatenate([[0.0], v])
    return multiply(multiply(q, qv), conjugate(q))[1:]


def kinematics(q: ArrayLike, omega_body_rad_s: ArrayLike) -> FloatArray:
    """Quaternion time derivative ``dq/dt = 1/2 q ⊗ [0, omega]``, 1/s.

    Parameters
    ----------
    q
        Attitude, BODY -> inertial. Deliberately not normalised here.
    omega_body_rad_s
        Angular velocity of BODY relative to inertial, in BODY axes, rad/s.
    """
    w = np.asarray(omega_body_rad_s, dtype=float)
    return 0.5 * multiply(q, np.concatenate([[0.0], w]))
