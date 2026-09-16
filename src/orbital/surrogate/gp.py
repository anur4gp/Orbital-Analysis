"""Gaussian-process surrogate for log10(Pc) over the encounter-parameter box.

Anisotropic (ARD) squared-exponential kernel, hyperparameters by maximizing
the log marginal likelihood. The GP itself is plain numpy -- a Cholesky
factorization and two triangular solves -- so the only dependency is scipy's
optimizer, which CLAUDE.md already anticipates for later phases.

Training targets are log10(Pc) rather than Pc: Pc spans ~13 orders of
magnitude over the box, so absolute error is meaningless and a GP on the raw
value would be dominated entirely by the few large-Pc points.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize


def ard_sqexp(xa: np.ndarray, xb: np.ndarray, log_theta: np.ndarray) -> np.ndarray:
    """Squared-exponential kernel with one length scale per dimension.

    log_theta packs [log amplitude, log length scales...]; working in logs
    keeps the optimizer unconstrained and the parameters positive.
    """
    # Clipped so the optimizer cannot walk into overflow or a zero length
    # scale; the marginal likelihood is flat well before these limits.
    amp = np.exp(2.0 * np.clip(log_theta[0], -10.0, 10.0))
    ls = np.exp(np.clip(log_theta[1:], -8.0, 8.0))
    diff = (xa[:, None, :] - xb[None, :, :]) / ls
    return amp * np.exp(-0.5 * np.sum(diff ** 2, axis=2))


@dataclass
class GP:
    """Fitted GP. Inputs are assumed already scaled to the unit cube."""

    x: np.ndarray
    y_mean: float
    y_std: float
    log_theta: np.ndarray
    log_noise: float
    _lower: np.ndarray = field(repr=False)
    _alpha: np.ndarray = field(repr=False)

    def predict(self, x_new: np.ndarray, return_std: bool = False):
        """Posterior mean (and optionally standard deviation) at new points."""
        x_new = np.atleast_2d(x_new)
        k_star = ard_sqexp(x_new, self.x, self.log_theta)
        mean = k_star @ self._alpha * self.y_std + self.y_mean
        if not return_std:
            return mean
        v = np.linalg.solve(self._lower, k_star.T)
        k_ss = np.exp(2.0 * self.log_theta[0])
        var = np.clip(k_ss - np.sum(v ** 2, axis=0), 0.0, None)
        return mean, np.sqrt(var) * self.y_std


def _neg_log_marginal(params: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    log_theta, log_noise = params[:-1], params[-1]
    n = x.shape[0]
    k = ard_sqexp(x, x, log_theta) + (np.exp(2.0 * log_noise) + 1e-10) * np.eye(n)
    try:
        lower = np.linalg.cholesky(k)
    except np.linalg.LinAlgError:
        return 1e12
    alpha = np.linalg.solve(lower.T, np.linalg.solve(lower, y))
    return float(0.5 * y @ alpha + np.sum(np.log(np.diag(lower))) + 0.5 * n * np.log(2 * np.pi))


def fit_gp(x: np.ndarray, y: np.ndarray, n_restarts: int = 4,
           seed: int = 0) -> GP:
    """Fit a GP to (x in the unit cube, y). Targets are standardized internally."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.asarray(y, dtype=float)
    y_mean, y_std = float(y.mean()), float(y.std())
    y_std = y_std if y_std > 0 else 1.0
    y_s = (y - y_mean) / y_std

    dim = x.shape[1]
    rng = np.random.default_rng(seed)
    best: np.ndarray | None = None
    best_val = np.inf
    # Start from a unit-amplitude, mid-range-length-scale, low-noise guess and
    # perturb; the marginal likelihood is multimodal in the length scales.
    start = np.concatenate([[0.0], np.full(dim, np.log(0.5)), [np.log(1e-3)]])
    for i in range(n_restarts):
        p0 = start if i == 0 else start + rng.normal(0, 0.7, size=start.size)
        bounds = [(-10.0, 10.0)] + [(-8.0, 8.0)] * dim + [(-14.0, 2.0)]
        res = minimize(_neg_log_marginal, np.clip(p0, [b[0] for b in bounds],
                                                 [b[1] for b in bounds]),
                       args=(x, y_s), method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 400})
        if res.fun < best_val:
            best_val, best = float(res.fun), res.x

    assert best is not None, "at least one restart always runs"
    log_theta, log_noise = best[:-1], best[-1]
    k = ard_sqexp(x, x, log_theta) + (np.exp(2.0 * log_noise) + 1e-10) * np.eye(x.shape[0])
    lower = np.linalg.cholesky(k)
    alpha = np.linalg.solve(lower.T, np.linalg.solve(lower, y_s))
    return GP(x=x, y_mean=y_mean, y_std=y_std, log_theta=log_theta,
              log_noise=float(log_noise), _lower=lower, _alpha=alpha)
