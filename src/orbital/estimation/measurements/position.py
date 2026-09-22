"""Direct position fix (e.g. onboard GNSS); linear, so EKF and UKF agree exactly."""
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
        """Label used in histories and plots."""
        return "position fix"

    @property
    def dim(self) -> int:
        """Three components: the position vector."""
        return 3

    @property
    def noise_covariance(self) -> FloatArray:
        """Isotropic noise covariance, km^2."""
        return self.sigma_km**2 * np.eye(3)

    def predict(self, t_s: float, x: FloatArray) -> FloatArray:
        """Predicted position, km: the state's own position block."""
        return np.array(x[:3], dtype=float)

    def jacobian(self, t_s: float, x: FloatArray) -> FloatArray:
        """Constant Jacobian ``[I, 0]``: the model is exactly linear."""
        return np.hstack([np.eye(3), np.zeros((3, 3))])

    def is_available(self, t_s: float, x: FloatArray) -> bool:
        """Always true: an onboard fix has no viewing geometry."""
        return True

    def residual(self, z: FloatArray, z_pred: FloatArray) -> FloatArray:
        """Innovation ``z - z_pred``, km."""
        return np.asarray(z, dtype=float) - z_pred
