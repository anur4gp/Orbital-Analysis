"""Direct position fix (e.g. an onboard GNSS solution), ECI_J2000, km.

Linear in the state, so the EKF and UKF must agree exactly on it -- which is
what makes it useful as a reference case, and it shows that adding a sensor
needs no filter changes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbital.attitude.quaternion import FloatArray


@dataclass(frozen=True)
class PositionFix:
    """Position measurement ``z = r + noise``.

    Parameters
    ----------
    sigma_km
        1-sigma noise per axis, km.
    """

    sigma_km: float = 0.010

    @property
    def name(self) -> str:
        return "position fix"

    @property
    def dim(self) -> int:
        return 3

    @property
    def noise_covariance(self) -> FloatArray:
        return self.sigma_km**2 * np.eye(3)

    def predict(self, t_s: float, x: FloatArray) -> FloatArray:
        return np.array(x[:3], dtype=float)

    def jacobian(self, t_s: float, x: FloatArray) -> FloatArray:
        return np.hstack([np.eye(3), np.zeros((3, 3))])

    def is_available(self, t_s: float, x: FloatArray) -> bool:
        return True

    def residual(self, z: FloatArray, z_pred: FloatArray) -> FloatArray:
        return np.asarray(z, dtype=float) - z_pred
