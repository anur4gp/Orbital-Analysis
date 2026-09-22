"""Conserved quantities for propagator validation, one value per sample."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from orbital.attitude.dcm import to_dcm
from orbital.attitude.quaternion import FloatArray
from orbital.dynamics.forces import ConservativeForce
from orbital.dynamics.inertia import InertiaTensor


def rotational_kinetic_energy(omega_rad_s: FloatArray, inertia: InertiaTensor) -> FloatArray:
    """Rotational kinetic energy ``1/2 omega . I omega``, J; omega (N, 3) rad/s."""
    w = np.atleast_2d(omega_rad_s)
    return 0.5 * np.einsum("ni,ij,nj->n", w, inertia.matrix, w)


def angular_momentum_inertial(
    q: FloatArray, omega_rad_s: FloatArray, inertia: InertiaTensor
) -> FloatArray:
    """Angular momentum in inertial axes, ``C(q) I omega``, kg m^2/s, shape (N, 3)."""
    qs = np.atleast_2d(q)
    ws = np.atleast_2d(omega_rad_s)
    return np.array([to_dcm(qi) @ (inertia.matrix @ wi) for qi, wi in zip(qs, ws, strict=True)])


def specific_orbital_energy(
    r_km: FloatArray, v_km_s: FloatArray, gravity: Sequence[ConservativeForce]
) -> FloatArray:
    """Specific energy ``|v|^2 / 2 + sum V(r)``, km^2/s^2, shape (N,)."""
    rs = np.atleast_2d(r_km)
    vs = np.atleast_2d(v_km_s)
    kinetic = 0.5 * np.einsum("ni,ni->n", vs, vs)
    potential = np.array([sum(g.potential(ri) for g in gravity) for ri in rs])
    return kinetic + potential


def specific_angular_momentum(r_km: FloatArray, v_km_s: FloatArray) -> FloatArray:
    """Specific angular momentum ``r x v``, km^2/s, shape (N, 3)."""
    return np.cross(np.atleast_2d(r_km), np.atleast_2d(v_km_s))
