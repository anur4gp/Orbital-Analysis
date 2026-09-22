"""NEES / NIS consistency tests and accuracy statistics over Monte Carlo runs.

NEES ``e^T P^-1 e`` (n dof) and NIS ``y^T S^-1 y`` (m dof) averaged over N
runs are chi-square with N*dof / N; above the band means overconfident.
Reference: Bar-Shalom, Li & Kirubarajan, ch. 5.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from orbital.attitude.quaternion import FloatArray
from orbital.estimation.measurements.base import MeasurementModel


def nees(errors: FloatArray, covariances: FloatArray) -> FloatArray:
    """NEES for errors ``(..., n)`` and covariances ``(..., n, n)``."""
    sol = np.linalg.solve(covariances, errors[..., None])[..., 0]
    return np.einsum("...i,...i->...", errors, sol)


def average_bounds(dof: int, n_runs: int, confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided interval for a chi-square(``dof``) statistic averaged over ``n_runs``."""
    tail = 0.5 * (1.0 - confidence)
    lo, hi = chi2.ppf([tail, 1.0 - tail], dof * n_runs)
    return float(lo / n_runs), float(hi / n_runs)


def fraction_inside(values: FloatArray, bounds: tuple[float, float]) -> float:
    """Fraction of ``values`` inside ``bounds`` (NaNs ignored)."""
    v = values[np.isfinite(values)]
    return float(np.mean((v >= bounds[0]) & (v <= bounds[1])))


def rms_over_runs(errors: FloatArray) -> FloatArray:
    """RMS vector-norm error per time, ``(N, K, d) -> (K,)``."""
    return np.sqrt(np.mean(np.sum(errors**2, axis=-1), axis=0))


def measurement_nonlinearity(
    model: MeasurementModel, t_s: float, x: FloatArray, p: FloatArray,
    steps: FloatArray | None = None,
) -> FloatArray:
    """Second-order measurement spread over noise sigma, per component.

    ``sqrt(tr(G P G P) / 2) / sigma`` with ``G`` the Hessian of ``h``; values
    near or above 1 mean the EKF linearisation is unreliable. Default
    finite-difference steps are 1 m and 1 mm/s.
    """
    h = np.array([1e-3] * 3 + [1e-6] * 3) if steps is None else np.asarray(steps)
    n = x.size
    hess = np.empty((model.dim, n, n))
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = h[i]
            ej[j] = h[j]
            d = (model.predict(t_s, x + ei + ej) - model.predict(t_s, x + ei - ej)
                 - model.predict(t_s, x - ei + ej) + model.predict(t_s, x - ei - ej))
            hess[:, i, j] = hess[:, j, i] = d / (4.0 * h[i] * h[j])
    gp = hess @ p
    spread = np.sqrt(0.5 * np.einsum("kij,kji->k", gp, gp))
    return spread / np.sqrt(np.diag(model.noise_covariance))
