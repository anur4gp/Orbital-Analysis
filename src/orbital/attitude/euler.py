"""Euler-angle conversions for the two sequences this project uses.

Both sequences are **intrinsic** (each rotation about the axis as already
moved) and describe BODY -> inertial:

``3-2-1`` (yaw, pitch, roll)
    ``C = Rz(yaw) Ry(pitch) Rx(roll)``. The aerospace attitude standard.
    Singular at ``pitch = +-pi/2``.
``3-1-3`` (precession phi, nutation theta, spin psi)
    ``C = Rz(phi) Rx(theta) Rz(psi)``. The classical sequence for spinning
    bodies, used for the torque-free precession analysis. Singular at
    ``theta = 0`` or ``pi``.

At a singularity only the sum (or difference) of the first and third angles
is defined. The convention here sets the **third** angle to zero and puts the
whole rotation in the first, so the conversion still round-trips the rotation
exactly even though it cannot round-trip the angles.

Only two sequences are provided on purpose: each is written out and tested
explicitly rather than generated for all twelve.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from orbital.attitude.dcm import from_dcm, to_dcm
from orbital.attitude.quaternion import FloatArray, from_axis_angle, multiply

#: Below this |cos(pitch)| (3-2-1) or |sin(theta)| (3-1-3) the sequence is
#: treated as singular. 1e-9 rad of the singular point.
GIMBAL_LOCK_TOL = 1e-9

_X, _Y, _Z = np.eye(3)


def from_euler_321(yaw_rad: float, pitch_rad: float, roll_rad: float) -> FloatArray:
    """Quaternion (BODY -> inertial) from 3-2-1 angles, rad."""
    return multiply(
        multiply(from_axis_angle(_Z, yaw_rad), from_axis_angle(_Y, pitch_rad)),
        from_axis_angle(_X, roll_rad),
    )


def to_euler_321(q: ArrayLike) -> FloatArray:
    """3-2-1 angles ``[yaw, pitch, roll]``, rad, from a quaternion.

    Ranges: yaw and roll in ``(-pi, pi]``, pitch in ``[-pi/2, pi/2]``.
    At gimbal lock roll is set to zero (see module docstring).
    """
    c = to_dcm(q)
    pitch = float(np.arcsin(np.clip(-c[2, 0], -1.0, 1.0)))
    if abs(np.cos(pitch)) < GIMBAL_LOCK_TOL:
        yaw = float(np.arctan2(-c[0, 1], c[1, 1]))
        return np.array([yaw, pitch, 0.0])
    roll = float(np.arctan2(c[2, 1], c[2, 2]))
    yaw = float(np.arctan2(c[1, 0], c[0, 0]))
    return np.array([yaw, pitch, roll])


def from_euler_313(phi_rad: float, theta_rad: float, psi_rad: float) -> FloatArray:
    """Quaternion (BODY -> inertial) from 3-1-3 angles, rad."""
    return multiply(
        multiply(from_axis_angle(_Z, phi_rad), from_axis_angle(_X, theta_rad)),
        from_axis_angle(_Z, psi_rad),
    )


def to_euler_313(q: ArrayLike) -> FloatArray:
    """3-1-3 angles ``[phi, theta, psi]``, rad, from a quaternion.

    Ranges: phi and psi in ``(-pi, pi]``, theta in ``[0, pi]``.
    At a singularity psi is set to zero (see module docstring).
    """
    c = to_dcm(q)
    theta = float(np.arccos(np.clip(c[2, 2], -1.0, 1.0)))
    if abs(np.sin(theta)) < GIMBAL_LOCK_TOL:
        phi = float(np.arctan2(c[1, 0], c[0, 0]))
        return np.array([phi, theta, 0.0])
    psi = float(np.arctan2(c[2, 0], c[2, 1]))
    phi = float(np.arctan2(c[0, 2], -c[1, 2]))
    return np.array([phi, theta, psi])


def dcm_from_euler_321(yaw_rad: float, pitch_rad: float, roll_rad: float) -> FloatArray:
    """DCM (BODY -> inertial) from 3-2-1 angles, rad. Convenience wrapper."""
    return to_dcm(from_euler_321(yaw_rad, pitch_rad, roll_rad))


def euler_321_from_dcm(c: ArrayLike) -> FloatArray:
    """3-2-1 angles, rad, from a DCM (BODY -> inertial). Convenience wrapper."""
    return to_euler_321(from_dcm(c))
