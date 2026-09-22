"""Synthetic per-object RTN covariance.

    C_rtn = diag(sigma_R^2, (k_T sigma_R)^2, (k_N sigma_R)^2)

Ratios k_T, k_N are fixed; only sigma_R is calibrated (one scalar observable
per event cannot identify more).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_K_T = 10.0
DEFAULT_K_N = 1.5


def rtn_basis(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """RTN basis with columns R, T, N, so ``x_eci = A @ x_rtn``."""
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)

    r_hat = r / np.linalg.norm(r)
    h = np.cross(r, v)
    n_hat = h / np.linalg.norm(h)
    t_hat = np.cross(n_hat, r_hat)
    return np.column_stack([r_hat, t_hat, n_hat])


@dataclass(frozen=True)
class RTNCovariance:
    """Diagonal RTN position covariance, parameterized by one scale."""

    sigma_r_km: float
    k_t: float = DEFAULT_K_T
    k_n: float = DEFAULT_K_N

    @property
    def sigma_t_km(self) -> float:
        """In-track 1-sigma, km: the radial scale times ``k_t``."""
        return self.k_t * self.sigma_r_km

    @property
    def sigma_n_km(self) -> float:
        """Cross-track 1-sigma, km: the radial scale times ``k_n``."""
        return self.k_n * self.sigma_r_km

    def matrix_rtn(self) -> np.ndarray:
        """3x3 covariance in the object's own RTN frame."""
        return np.diag(
            np.array([self.sigma_r_km, self.sigma_t_km, self.sigma_n_km]) ** 2
        )

    def matrix_eci(self, r: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Covariance rotated into the inertial frame, ``A C A^T``, km^2."""
        a = rtn_basis(r, v)
        return a @ self.matrix_rtn() @ a.T


def combined_covariance(
    r1: np.ndarray, v1: np.ndarray, r2: np.ndarray, v2: np.ndarray,
    cov1: RTNCovariance, cov2: RTNCovariance,
) -> np.ndarray:
    """Relative-position covariance C1 + C2, inertial frame, km^2.

    Each object has its own RTN frame, so both are rotated to inertial before
    summing. Errors are assumed independent.
    """
    return cov1.matrix_eci(r1, v1) + cov2.matrix_eci(r2, v2)


def sample_relative_offsets(
    cov: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw ``n`` offsets from ``N(0, cov)``, km, shape (n, 3)."""
    try:
        factor = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(cov)
        factor = vecs @ np.diag(np.sqrt(np.clip(vals, 0.0, None)))
    return rng.standard_normal((n, 3)) @ factor.T


# Fitted by scripts/calibrate.py on 37 cdm_public events (k_T, k_N fixed).
CALIBRATED_SIGMA_R_KM = 0.0953
