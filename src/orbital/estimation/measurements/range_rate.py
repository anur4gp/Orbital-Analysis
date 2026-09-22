"""Two-way range and range-rate from a ground station.

    rho = |r - r_s|,  rhodot = rho_hat . (v - v_s)
    d rhodot / d r = (v_rel - rhodot rho_hat)^T / rho

Light time and atmospheric delay are not modelled.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbital.attitude.quaternion import FloatArray
from orbital.estimation.measurements.ground_station import GroundStation


@dataclass(frozen=True)
class RangeRangeRate:
    """Range (km) and range-rate (km/s) from ``station``.

    Parameters
    ----------
    station
        The tracking site.
    sigma_range_km
        1-sigma range noise, km.
    sigma_range_rate_km_s
        1-sigma range-rate noise, km/s.
    """

    station: GroundStation
    sigma_range_km: float = 0.010
    sigma_range_rate_km_s: float = 1.0e-5

    @property
    def name(self) -> str:
        """Label used in histories and plots, naming the site."""
        return f"range/range-rate @ {self.station.name}"

    @property
    def dim(self) -> int:
        """Two components: range and range-rate."""
        return 2

    @property
    def noise_covariance(self) -> FloatArray:
        """Diagonal noise covariance, km^2 and (km/s)^2."""
        return np.diag([self.sigma_range_km**2, self.sigma_range_rate_km_s**2])

    def _geometry(self, t_s: float, x: FloatArray) -> tuple[FloatArray, FloatArray, float]:
        rho_vec = x[:3] - self.station.position_eci(t_s)
        v_rel = x[3:6] - self.station.velocity_eci(t_s)
        return rho_vec, v_rel, float(np.linalg.norm(rho_vec))

    def predict(self, t_s: float, x: FloatArray) -> FloatArray:
        """Predicted ``[range (km), range-rate (km/s)]``."""
        rho_vec, v_rel, rho = self._geometry(t_s, x)
        return np.array([rho, float(rho_vec @ v_rel) / rho])

    def jacobian(self, t_s: float, x: FloatArray) -> FloatArray:
        """Analytic dh/dx, shape (2, 6)."""
        rho_vec, v_rel, rho = self._geometry(t_s, x)
        u = rho_vec / rho
        rho_dot = float(u @ v_rel)
        h = np.zeros((2, 6))
        h[0, :3] = u
        h[1, :3] = (v_rel - rho_dot * u) / rho
        h[1, 3:] = u
        return h

    def is_available(self, t_s: float, x: FloatArray) -> bool:
        """Whether the object is above the site's elevation mask."""
        return self.station.is_visible(t_s, x[:3])

    def residual(self, z: FloatArray, z_pred: FloatArray) -> FloatArray:
        """Innovation ``z - z_pred``; neither component wraps."""
        return np.asarray(z, dtype=float) - z_pred
