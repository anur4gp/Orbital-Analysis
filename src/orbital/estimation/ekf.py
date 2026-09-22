"""Extended Kalman filter with Joseph-form update.

Equations::

    x- = phi(x),  P- = Phi P Phi^T + Q
    K  = P- H^T S^-1,  S = H P- H^T + R
    P+ = (I - K H) P- (I - K H)^T + K R K^T
"""
from __future__ import annotations

import numpy as np

from orbital.attitude.quaternion import FloatArray
from orbital.estimation.base import Observation, SequentialFilter, UpdateInfo, symmetrize


class EKF(SequentialFilter):
    """Extended Kalman filter for the 6-state orbit."""

    name = "EKF"

    def predict(
        self, x: FloatArray, p: FloatArray, t0_s: float, t1_s: float
    ) -> tuple[FloatArray, FloatArray]:
        """Propagate the state nonlinearly and the covariance through the STM."""
        x1, phi = self.model.propagate_with_stm(x, t0_s, t1_s)
        p1 = phi @ p @ phi.T + self.process_noise(t1_s - t0_s)
        return x1, symmetrize(p1)

    def update(
        self, x: FloatArray, p: FloatArray, obs: Observation
    ) -> tuple[FloatArray, FloatArray, UpdateInfo]:
        """Condition on one observation, linearising h about the prior mean."""
        m = obs.model
        h = m.jacobian(obs.t_s, x)
        r = m.noise_covariance
        y = m.residual(obs.z, m.predict(obs.t_s, x))
        s = symmetrize(h @ p @ h.T + r)
        k = np.linalg.solve(s, h @ p).T  # P H^T S^-1, with S and P symmetric
        i_kh = np.eye(len(x)) - k @ h
        p_post = i_kh @ p @ i_kh.T + k @ r @ k.T
        return x + k @ y, symmetrize(p_post), UpdateInfo(y, s)
