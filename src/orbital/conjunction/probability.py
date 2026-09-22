"""Collision probability in the short-term encounter model.

Pc is the projected 2-D Gaussian integrated over the hard-body disk in the
encounter plane (perpendicular to relative velocity).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Equivalent-sphere radii per Space-Track RCS class (assumed). EXCL_VOL is a
# km-scale screening volume, not a physical size, so it is not used here.
HBR_BY_RCS_M = {"SMALL": 0.5, "MEDIUM": 1.0, "LARGE": 3.0}
DEFAULT_HBR_M = 1.0


def hard_body_radius_km(rcs1: str, rcs2: str) -> float:
    """Combined hard-body radius, km, from two RCS classes (unknown -> default)."""
    r1 = HBR_BY_RCS_M.get((rcs1 or "").upper(), DEFAULT_HBR_M)
    r2 = HBR_BY_RCS_M.get((rcs2 or "").upper(), DEFAULT_HBR_M)
    return (r1 + r2) / 1000.0


def encounter_plane_basis(v_rel: np.ndarray) -> np.ndarray:
    """Orthonormal basis (3, 2) of the plane perpendicular to ``v_rel``."""
    n = np.asarray(v_rel, dtype=float)
    n = n / np.linalg.norm(n)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(n @ seed) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e1 = seed - (seed @ n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return np.column_stack([e1, e2])


def project_encounter(
    dr: np.ndarray, v_rel: np.ndarray, cov: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project relative position (km) and covariance (km^2) onto the encounter plane."""
    b = encounter_plane_basis(v_rel)
    return b.T @ np.asarray(dr, dtype=float), b.T @ cov @ b


@dataclass
class PcResult:
    """A Pc estimate with its Monte Carlo uncertainty."""

    pc: float
    stderr: float
    n_draws: int
    n_hits: int

    @property
    def relative_error(self) -> float:
        """Standard error divided by the estimate; NaN when no hits were drawn."""
        return float("inf") if self.pc == 0 else self.stderr / self.pc

    def __str__(self) -> str:
        return f"{self.pc:.4e} +/- {self.stderr:.1e} ({self.n_hits} hits / {self.n_draws:,})"


def pc_monte_carlo(
    mu_2d: np.ndarray,
    cov_2d: np.ndarray,
    hbr_km: float,
    n_draws: int,
    rng: np.random.Generator,
    batch: int = 2_000_000,
) -> PcResult:
    """Brute-force Pc: fraction of samples inside the hard-body disk (batched)."""
    try:
        factor = np.linalg.cholesky(cov_2d)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(cov_2d)
        factor = vecs @ np.diag(np.sqrt(np.clip(vals, 0.0, None)))

    hits, done = 0, 0
    r2 = hbr_km * hbr_km
    while done < n_draws:
        m = min(batch, n_draws - done)
        pts = mu_2d + rng.standard_normal((m, 2)) @ factor.T
        hits += int(np.count_nonzero((pts * pts).sum(axis=1) < r2))
        done += m

    pc = hits / n_draws
    # Binomial stderr, floored at the 1-hit scale when there are no hits.
    stderr = np.sqrt(max(pc, 1.0 / n_draws) * (1 - pc) / n_draws)
    return PcResult(pc=pc, stderr=float(stderr), n_draws=n_draws, n_hits=hits)


def pc_analytic(mu_2d: np.ndarray, cov_2d: np.ndarray, hbr_km: float,
                n_r: int = 400, n_theta: int = 400) -> float:
    """Pc by midpoint polar quadrature of the 2-D Gaussian over the disk.

    ``n_r`` is a floor; it is raised so the radial step resolves the
    smallest covariance sigma.
    """
    inv = np.linalg.inv(cov_2d)
    norm = 1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(cov_2d)))

    # ~200 steps per sigma: midpoint error ~(dr/sigma)^2 / 24 ~ 1e-6.
    sigma_min = float(np.sqrt(np.clip(np.linalg.eigvalsh(cov_2d).min(), 1e-300, None)))
    n_r = int(np.clip(np.ceil(hbr_km / (sigma_min / 200.0)), n_r, 4_000_000))

    dr = hbr_km / n_r
    dtheta = 2.0 * np.pi / n_theta
    theta = (np.arange(n_theta) + 0.5) * dtheta
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    total = 0.0
    chunk = max(1, 4_000_000 // n_theta)
    for start in range(0, n_r, chunk):
        r = (np.arange(start, min(start + chunk, n_r)) + 0.5) * dr
        x = r[:, None] * cos_t[None, :] - mu_2d[0]
        y = r[:, None] * sin_t[None, :] - mu_2d[1]
        quad = inv[0, 0] * x * x + 2 * inv[0, 1] * x * y + inv[1, 1] * y * y
        total += float((np.exp(-0.5 * quad) * r[:, None]).sum())

    return float(norm * total * dr * dtheta)


def pc_small_disk(mu_2d: np.ndarray, cov_2d: np.ndarray, hbr_km: float) -> float:
    """Closed form for HBR << sigma: density at the miss point times disk area."""
    inv = np.linalg.inv(cov_2d)
    norm = 1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(cov_2d)))
    quad = mu_2d @ inv @ mu_2d
    return float(norm * np.exp(-0.5 * quad) * np.pi * hbr_km * hbr_km)


def log10_pc_small_disk(mu_2d: np.ndarray, cov_2d: np.ndarray, hbr_km: float) -> float:
    """log10 of :func:`pc_small_disk`, computed in logs so it never underflows."""
    inv = np.linalg.inv(cov_2d)
    quad = float(mu_2d @ inv @ mu_2d)
    log_norm = -np.log(2.0 * np.pi * np.sqrt(np.linalg.det(cov_2d)))
    log_area = np.log(np.pi * hbr_km * hbr_km)
    return float((log_norm - 0.5 * quad + log_area) / np.log(10.0))
