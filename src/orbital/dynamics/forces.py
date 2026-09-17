"""Force models, expressed as accelerations in ECI_J2000.

A force model returns acceleration (km/s^2), not force: gravity is
mass-independent, and non-gravitational models can divide by
``mass_properties.mass_kg`` themselves. Gravity models also expose their
potential so the total mechanical energy can be checked for conservation.

Modelling assumption for J2: the ECI_J2000 z-axis is taken as Earth's spin
axis. Precession and nutation move the true pole by about 0.3 degrees since
J2000; that is neglected here and would enter through a proper
ECI -> ECEF transformation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from orbital.attitude.quaternion import FloatArray
from orbital.core.constants import J2_EARTH, MU_EARTH_KM3_S2, R_EARTH_KM
from orbital.dynamics.inertia import MassProperties
from orbital.dynamics.state import RigidBodyState


class ForceModel(Protocol):
    """Acceleration on the centre of mass."""

    def acceleration(
        self, state: RigidBodyState, mass_properties: MassProperties
    ) -> FloatArray:
        """Acceleration, km/s^2, ECI_J2000."""
        ...


@runtime_checkable
class ConservativeForce(ForceModel, Protocol):
    """A force derivable from a potential: ``a = -grad V``."""

    def potential(self, r_km: FloatArray) -> float:
        """Potential energy per unit mass, km^2/s^2."""
        ...


@runtime_checkable
class PositionOnlyForce(ForceModel, Protocol):
    """A force that depends only on position, evaluable for many points at once.

    Orbit determination propagates many states together (sigma points,
    finite-difference stencils); the vectorised form avoids a Python-level
    loop per state.
    """

    def acceleration_many(self, r_km: FloatArray) -> FloatArray:
        """Accelerations, km/s^2, for positions of shape ``(k, 3)``."""
        ...


@dataclass(frozen=True)
class TwoBodyGravity:
    """Point-mass gravity ``a = -mu r / |r|^3``.

    Parameters
    ----------
    mu_km3_s2
        Gravitational parameter, km^3/s^2.
    """

    mu_km3_s2: float = MU_EARTH_KM3_S2

    def acceleration(
        self, state: RigidBodyState, mass_properties: MassProperties
    ) -> FloatArray:
        r = state.r_km
        rn = float(np.linalg.norm(r))
        return -self.mu_km3_s2 * r / rn**3

    def acceleration_many(self, r_km: FloatArray) -> FloatArray:
        rn = np.sqrt(np.einsum("ki,ki->k", r_km, r_km))
        return -self.mu_km3_s2 * r_km / rn[:, None] ** 3

    def potential(self, r_km: FloatArray) -> float:
        return -self.mu_km3_s2 / float(np.linalg.norm(r_km))


@dataclass(frozen=True)
class J2Gravity:
    """Perturbing acceleration of Earth's oblateness (J2 term only).

    Add alongside :class:`TwoBodyGravity`; this model is the perturbation,
    not the total field. Potential per unit mass:

        V_J2 = (mu J2 R^2 / (2 r^3)) (3 z^2 / r^2 - 1)

    Parameters
    ----------
    mu_km3_s2
        Gravitational parameter, km^3/s^2.
    j2
        Unnormalised J2 coefficient, dimensionless.
    r_ref_km
        Reference radius of the harmonic expansion, km.
    """

    mu_km3_s2: float = MU_EARTH_KM3_S2
    j2: float = J2_EARTH
    r_ref_km: float = R_EARTH_KM

    def acceleration(
        self, state: RigidBodyState, mass_properties: MassProperties
    ) -> FloatArray:
        return self.acceleration_many(state.r_km[None, :])[0]

    def acceleration_many(self, r_km: FloatArray) -> FloatArray:
        x, y, z = r_km[:, 0], r_km[:, 1], r_km[:, 2]
        r2 = x * x + y * y + z * z
        k = -1.5 * self.j2 * self.mu_km3_s2 * self.r_ref_km**2 / r2**2.5
        zr = 5.0 * z * z / r2
        return np.column_stack([k * x * (1.0 - zr), k * y * (1.0 - zr), k * z * (3.0 - zr)])

    def potential(self, r_km: FloatArray) -> float:
        r = float(np.linalg.norm(r_km))
        z = float(r_km[2])
        return (
            0.5 * self.mu_km3_s2 * self.j2 * self.r_ref_km**2 / r**3
            * (3.0 * z * z / (r * r) - 1.0)
        )
