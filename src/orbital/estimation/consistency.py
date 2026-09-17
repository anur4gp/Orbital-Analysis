"""Filter consistency and accuracy statistics over Monte Carlo runs.

A filter is *consistent* when its covariance describes its actual errors.
Two chi-square tests check this (Bar-Shalom, Li & Kirubarajan, ch. 5):

NEES (needs truth)
    ``eps_k = e_k^T P_k^-1 e_k`` with ``e = x_true - x_hat``. For a consistent
    filter it is chi-square with n = 6 dof. Averaged over N independent
    runs, ``N * mean(eps_k)`` is chi-square with N n dof, giving a two-sided
    acceptance interval for the average.
NIS (needs only measurements)
    ``nu_k = y_k^T S_k^-1 y_k``, chi-square with m dof, averaged the same way.
    This is the one that can be run on real data.

An average above the interval means the filter is *overconfident*
(P too small for its errors) -- the dangerous direction, because the filter
then discounts measurements that would correct it.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from orbital.attitude.quaternion import FloatArray
from orbital.estimation.measurements.base import MeasurementModel


def nees(errors: FloatArray, covariances: FloatArray) -> FloatArray:
    """Normalised estimation error squared.

    Parameters
    ----------
    errors
        ``x_true - x_hat``, shape ``(..., n)``.
    covariances
        Matching covariances, shape ``(..., n, n)``.

    Returns
    -------
    ndarray
        NEES, shape ``(...)``.
    """
    sol = np.linalg.solve(covariances, errors[..., None])[..., 0]
    return np.einsum("...i,...i->...", errors, sol)


def average_bounds(dof: int, n_runs: int, confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided acceptance interval for a chi-square statistic averaged over runs.

    Parameters
    ----------
    dof
        Degrees of freedom of one sample (6 for NEES, m for NIS).
    n_runs
        Number of independent runs averaged.
    confidence
        Probability mass inside the interval.
    """
    tail = 0.5 * (1.0 - confidence)
    lo, hi = chi2.ppf([tail, 1.0 - tail], dof * n_runs)
    return float(lo / n_runs), float(hi / n_runs)


def fraction_inside(values: FloatArray, bounds: tuple[float, float]) -> float:
    """Fraction of ``values`` inside ``bounds`` (NaNs ignored)."""
    v = values[np.isfinite(values)]
    return float(np.mean((v >= bounds[0]) & (v <= bounds[1])))


def rms_over_runs(errors: FloatArray) -> FloatArray:
    """RMS vector-norm error at each time.

    Parameters
    ----------
    errors
        Shape ``(N, K, d)``, e.g. the position block of the state error.

    Returns
    -------
    ndarray
        ``sqrt(mean_runs |e|^2)``, shape ``(K,)``, in the units of ``errors``.
    """
    return np.sqrt(np.mean(np.sum(errors**2, axis=-1), axis=0))


def measurement_nonlinearity(
    model: MeasurementModel, t_s: float, x: FloatArray, p: FloatArray,
    steps: FloatArray | None = None,
) -> FloatArray:
    """Second-order measurement spread relative to the noise, per component.

    For ``h_i`` with Hessian ``G_i``, the second-order term of ``h_i(x + dx)``
    with ``dx ~ N(0, P)`` has mean ``tr(G_i P) / 2`` and standard deviation
    ``sqrt(tr(G_i P G_i P) / 2)``. The EKF drops both. This returns that
    standard deviation divided by the measurement sigma: well below 1 the
    linearisation is harmless; near or above 1 the EKF treats as information
    an error it does not model.

    Parameters
    ----------
    model
        Measurement model.
    t_s, x, p
        Time, linearisation point and prior covariance.
    steps
        Finite-difference steps per state component (default 1 m, 1 mm/s).
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
