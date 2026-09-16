"""Conserved quantities, used to validate the propagator.

Each function takes trajectory arrays and returns one value per sample, so
drift is a one-liner: ``x - x[0]``.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from orbital.attitude.dcm import to_dcm
from orbital.attitude.quaternion import FloatArray
from orbital.dynamics.forces import ConservativeForce
from orbital.dynamics.inertia import InertiaTensor


def rotational_kinetic_energy(omega_rad_s: FloatArray, inertia: InertiaTensor) -> FloatArray:
    """``T = 1/2 omega . I omega``, J, for ``omega`` of shape ``(N, 3)``."""
    w = np.atleast_2d(omega_rad_s)
    return 0.5 * np.einsum("ni,ij,nj->n", w, inertia.matrix, w)


def angular_momentum_inertial(
    q: FloatArray, omega_rad_s: FloatArray, inertia: InertiaTensor
) -> FloatArray:
    """Body angular momentum in inertial axes, ``H_I = C(q) I omega``, kg m^2/s.

    Returns shape ``(N, 3)``. Conserved as a *vector* when torque-free, which
    tests the attitude history, not just the body rates.
    """
    qs = np.atleast_2d(q)
    ws = np.atleast_2d(omega_rad_s)
    return np.array([to_dcm(qi) @ (inertia.matrix @ wi) for qi, wi in zip(qs, ws, strict=True)])


def specific_orbital_energy(
    r_km: FloatArray, v_km_s: FloatArray, gravity: Sequence[ConservativeForce]
) -> FloatArray:
    """``E = |v|^2 / 2 + sum V(r)``, km^2/s^2, shape ``(N,)``."""
    rs = np.atleast_2d(r_km)
    vs = np.atleast_2d(v_km_s)
    kinetic = 0.5 * np.einsum("ni,ni->n", vs, vs)
    potential = np.array([sum(g.potential(ri) for g in gravity) for ri in rs])
    return kinetic + potential


def specific_angular_momentum(r_km: FloatArray, v_km_s: FloatArray) -> FloatArray:
    """Orbital ``h = r x v``, km^2/s, shape ``(N, 3)``."""
    return np.cross(np.atleast_2d(r_km), np.atleast_2d(v_km_s))
