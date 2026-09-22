"""Torque models, expressed in BODY axes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from orbital.attitude.dcm import to_dcm
from orbital.attitude.quaternion import FloatArray
from orbital.core.constants import MU_EARTH_KM3_S2
from orbital.dynamics.inertia import MassProperties
from orbital.dynamics.state import RigidBodyState


class TorqueModel(Protocol):
    """External torque about the centre of mass."""

    def torque(self, state: RigidBodyState, mass_properties: MassProperties) -> FloatArray:
        """Torque, N m, BODY axes."""
        ...


@dataclass(frozen=True)
class GravityGradientTorque:
    """Gravity-gradient torque of a point-mass central body.

        tau_B = (3 mu / |r|^3) (u x I u),    u = C^T r / |r|

    ``u`` is the BODY-axis unit vector from Earth's centre. ``mu / r^3`` is
    1/s^2 in any length unit, so the torque is in N m.

    Parameters
    ----------
    mu_km3_s2
        Gravitational parameter, km^3/s^2.
    """

    mu_km3_s2: float = MU_EARTH_KM3_S2

    def torque(self, state: RigidBodyState, mass_properties: MassProperties) -> FloatArray:
        """Gravity-gradient torque, N m, BODY axes."""
        r = state.r_km
        rn = float(np.linalg.norm(r))
        u = to_dcm(state.q).T @ (r / rn)
        inertia = mass_properties.inertia.matrix
        return 3.0 * self.mu_km3_s2 / rn**3 * np.cross(u, inertia @ u)
