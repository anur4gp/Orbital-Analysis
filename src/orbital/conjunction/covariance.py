"""Option B covariance model: diagonal RTN, per object.

cdm_public carries no covariance, so uncertainty is synthesized. Each object
gets a diagonal covariance in its own RTN frame

    C_rtn = diag(sigma_R^2, (k_T sigma_R)^2, (k_N sigma_R)^2)

with the anisotropy ratios k_T, k_N FIXED to literature-typical values and
only the overall scale sigma_R calibrated. Twelve free parameters cannot be
identified from one scalar observable per event; one or two can.

This is a stated modeling assumption, not a data product -- see CLAUDE.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# In-track error dominates because a small period/energy error integrates into
# a growing along-track lag. Cross-track stays comparable to radial.
DEFAULT_K_T = 10.0
DEFAULT_K_N = 1.5


def rtn_basis(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Orthonormal RTN basis as a 3x3 matrix whose COLUMNS are R, T, N.

    R = radial (along position), N = orbit normal (along r x v),
    T = N x R, which completes the right-handed triad and lies along the
    velocity for a circular orbit. Returned matrix A maps RTN -> inertial:
    x_eci = A @ x_rtn.
    """
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
        return self.k_t * self.sigma_r_km

    @property
    def sigma_n_km(self) -> float:
        return self.k_n * self.sigma_r_km

    def matrix_rtn(self) -> np.ndarray:
        """3x3 covariance in the object's own RTN frame."""
        return np.diag(
            np.array([self.sigma_r_km, self.sigma_t_km, self.sigma_n_km]) ** 2
        )

    def matrix_eci(self, r: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Same covariance rotated into the inertial frame: A C A^T."""
        a = rtn_basis(r, v)
        return a @ self.matrix_rtn() @ a.T


def combined_covariance(
    r1: np.ndarray, v1: np.ndarray, r2: np.ndarray, v2: np.ndarray,
    cov1: RTNCovariance, cov2: RTNCovariance,
) -> np.ndarray:
    """Relative-position covariance C1 + C2, in the inertial frame.

    The two objects have DIFFERENT RTN frames, so each covariance must be
    rotated into a common frame before summing -- adding them componentwise
    in RTN would be wrong. Assumes the two error sets are independent, which
    is standard for objects tracked separately.
    """
    return cov1.matrix_eci(r1, v1) + cov2.matrix_eci(r2, v2)


def sample_relative_offsets(
    cov: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw n relative-position offsets from N(0, cov). Returns (n, 3).

    Uses the Cholesky factor; falls back to an eigendecomposition if the
    matrix is numerically non-positive-definite.
    """
    try:
        factor = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(cov)
        factor = vecs @ np.diag(np.sqrt(np.clip(vals, 0.0, None)))
    return rng.standard_normal((n, 3)) @ factor.T


# Fitted by src/calibrate.py against 37 deduplicated cdm_public events, holding
# k_T and k_N fixed. See CLAUDE.md for the caveat: this reproduces the overall
# scale of TLE error, not which individual events are worst.
CALIBRATED_SIGMA_R_KM = 0.0953
