"""13-element rigid-body state.

Packed layout::

    y[0:3]    r      km, ECI_J2000
    y[3:6]    v      km/s, ECI_J2000
    y[6:10]   q      BODY -> ECI_J2000, scalar first
    y[10:13]  omega  rad/s, BODY axes
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from orbital.attitude.quaternion import FloatArray, as_quaternion, normalize
from orbital.conventions import Frame

STATE_SIZE = 13
R, V, Q, W = slice(0, 3), slice(3, 6), slice(6, 10), slice(10, 13)


def _vec3(x: ArrayLike, name: str) -> FloatArray:
    arr = np.asarray(x, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    return arr


@dataclass(frozen=True)
class RigidBodyState:
    """Full 6-DOF state at time ``t_s``.

    Attributes
    ----------
    r_km
        Centre-of-mass position, km.
    v_km_s
        Centre-of-mass velocity, km/s.
    q
        Attitude quaternion, BODY -> ``frame``. Normalised on construction.
    omega_rad_s
        Angular velocity of BODY w.r.t. inertial, BODY axes, rad/s.
    t_s
        Time since the reference epoch, s.
    frame
        Frame of ``r``, ``v`` and the target frame of ``q``.
    """

    r_km: FloatArray
    v_km_s: FloatArray
    q: FloatArray
    omega_rad_s: FloatArray
    t_s: float = 0.0
    frame: Frame = field(default=Frame.ECI_J2000)

    def __post_init__(self) -> None:
        object.__setattr__(self, "r_km", _vec3(self.r_km, "r_km"))
        object.__setattr__(self, "v_km_s", _vec3(self.v_km_s, "v_km_s"))
        object.__setattr__(self, "omega_rad_s", _vec3(self.omega_rad_s, "omega_rad_s"))
        object.__setattr__(self, "q", normalize(as_quaternion(self.q)))

    def to_vector(self) -> FloatArray:
        """Pack into the 13-element integrator vector."""
        return np.concatenate([self.r_km, self.v_km_s, self.q, self.omega_rad_s])

    @classmethod
    def from_vector(
        cls, y: ArrayLike, t_s: float = 0.0, frame: Frame = Frame.ECI_J2000
    ) -> RigidBodyState:
        """Unpack a 13-element vector; the quaternion is normalised."""
        arr = np.asarray(y, dtype=float)
        if arr.shape != (STATE_SIZE,):
            raise ValueError(f"state vector must have shape ({STATE_SIZE},), got {arr.shape}")
        return cls(arr[R], arr[V], arr[Q], arr[W], t_s, frame)
