"""Unscented Kalman filter (additive noise, scaled sigma points).

    lambda = alpha^2 (n + kappa) - n
    Wm_0 = lambda / (n + lambda),  Wc_0 = Wm_0 + 1 - alpha^2 + beta
    Wm_i = Wc_i = 1 / (2 (n + lambda))

Defaults ``alpha=1, beta=2, kappa=0`` keep all weights non-negative; the common
``alpha=1e-3`` is numerically fragile in 6-D. Sigma points are redrawn after
prediction so Q enters the measurement update.
"""
from __future__ import annotations

import numpy as np

from orbital.attitude.quaternion import FloatArray
from orbital.estimation.base import Observation, SequentialFilter, UpdateInfo, symmetrize
from orbital.estimation.orbit_model import OrbitModel


class UKF(SequentialFilter):
    """Unscented Kalman filter for the 6-state orbit.

    Parameters
    ----------
    model, process_noise_psd
        As for :class:`~orbital.estimation.base.SequentialFilter`.
    alpha, beta, kappa
        Sigma-point spread and weighting.
    """

    name = "UKF"

    def __init__(
        self,
        model: OrbitModel,
        process_noise_psd: float = 0.0,
        alpha: float = 1.0,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        super().__init__(model, process_noise_psd)
        n = 6
        self.lam = alpha**2 * (n + kappa) - n
        if n + self.lam <= 0.0:
            raise ValueError("alpha and kappa must give n + lambda > 0")
        self.wm = np.full(2 * n + 1, 0.5 / (n + self.lam))
        self.wc = self.wm.copy()
        self.wm[0] = self.lam / (n + self.lam)
        self.wc[0] = self.wm[0] + (1.0 - alpha**2 + beta)

    def sigma_points(self, x: FloatArray, p: FloatArray) -> FloatArray:
        """The ``2n + 1`` sigma points, shape ``(13, 6)``; raises if P is not PD."""
        n = len(x)
        root = np.linalg.cholesky((n + self.lam) * p)
        return np.vstack([x, x + root.T, x - root.T])

    def _moments(self, points: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        mean = self.wm @ points
        dev = points - mean
        cov = (self.wc[:, None] * dev).T @ dev
        return mean, dev, cov

    def predict(
        self, x: FloatArray, p: FloatArray, t0_s: float, t1_s: float
    ) -> tuple[FloatArray, FloatArray]:
        """Propagate every sigma point, then re-estimate mean and covariance."""
        images = self.model.propagate_many(self.sigma_points(x, p), t0_s, t1_s)
        mean, _, cov = self._moments(images)
        return mean, symmetrize(cov + self.process_noise(t1_s - t0_s))

    def update(
        self, x: FloatArray, p: FloatArray, obs: Observation
    ) -> tuple[FloatArray, FloatArray, UpdateInfo]:
        """Condition on one observation using sigma points through h."""
        m = obs.model
        chi = self.sigma_points(x, p)
        zs = np.array([m.predict(obs.t_s, c) for c in chi])
        z_mean, z_dev, s = self._moments(zs)
        s = symmetrize(s + m.noise_covariance)
        x_dev = chi - self.wm @ chi
        p_xz = (self.wc[:, None] * x_dev).T @ z_dev
        k = np.linalg.solve(s, p_xz.T).T
        y = m.residual(obs.z, z_mean)
        return x + k @ y, symmetrize(p - k @ s @ k.T), UpdateInfo(y, s)

