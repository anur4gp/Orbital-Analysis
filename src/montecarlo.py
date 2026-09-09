"""Phase 2: brute-force Monte Carlo collision probability.

Uses the short-term encounter model. At high relative speed the encounter
lasts milliseconds, relative motion is effectively rectilinear, and the 3D
problem collapses onto the 2D "encounter plane" perpendicular to the
relative velocity. Pc is then the integral of the projected Gaussian over a
disk of the combined hard-body radius.

This module is the expensive ground truth Phase 3's surrogate is benchmarked
against, so it is deliberately unoptimized in method -- only vectorized.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# EXCL_VOL in cdm_public is a km-scale SCREENING volume keyed to object class
# (debris 1, rocket body 3, payload 5) -- not a physical size. Hard-body radii
# come from the RCS size class instead. Space-Track bins RCS as
# SMALL < 0.1 m^2, MEDIUM 0.1-1.0 m^2, LARGE > 1.0 m^2; the radii below are
# conservative equivalent-sphere values and are a stated assumption.
HBR_BY_RCS_M = {"SMALL": 0.5, "MEDIUM": 1.0, "LARGE": 3.0}
DEFAULT_HBR_M = 1.0


def hard_body_radius_km(rcs1: str, rcs2: str) -> float:
    """Combined hard-body radius in km, from the two RCS size classes."""
    r1 = HBR_BY_RCS_M.get((rcs1 or "").upper(), DEFAULT_HBR_M)
    r2 = HBR_BY_RCS_M.get((rcs2 or "").upper(), DEFAULT_HBR_M)
    return (r1 + r2) / 1000.0


def encounter_plane_basis(v_rel: np.ndarray) -> np.ndarray:
    """Orthonormal 3x2 basis for the plane perpendicular to relative velocity."""
    n = np.asarray(v_rel, dtype=float)
    n = n / np.linalg.norm(n)
    # Any vector not parallel to n seeds the first in-plane direction.
    seed = np.array([1.0, 0.0, 0.0])
    if abs(n @ seed) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e1 = seed - (seed @ n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return np.column_stack([e1, e2])


def project_encounter(dr: np.ndarray, v_rel: np.ndarray, cov: np.ndarray):
    """Project relative position and covariance onto the encounter plane.

    Returns (mu_2d, cov_2d): the miss vector and combined covariance as seen
    looking down the relative-velocity axis.
    """
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
    """Brute-force Pc: fraction of sampled encounters inside the hard-body disk.

    Draws are batched so the memory footprint stays flat as n_draws grows into
    the tens of millions, which is where a Pc of ~1e-4 needs to be resolved.
    """
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
    # Binomial standard error; falls back to the 1-hit scale when hits == 0.
    stderr = np.sqrt(max(pc, 1.0 / n_draws) * (1 - pc) / n_draws)
    return PcResult(pc=pc, stderr=float(stderr), n_draws=n_draws, n_hits=hits)


def pc_analytic(mu_2d: np.ndarray, cov_2d: np.ndarray, hbr_km: float,
                n_r: int = 400, n_theta: int = 400) -> float:
    """Pc by direct polar quadrature of the 2D Gaussian over the disk.

    Independent of the Monte Carlo, so disagreement between the two points at
    a bug in one of them rather than at sampling noise.

    The radial resolution adapts to the covariance: a fixed grid whose step
    exceeds the smallest standard deviation would step straight over the
    probability mass and silently return a wrong answer. Integration is
    chunked radially so memory stays flat however fine the grid becomes.
    """
    inv = np.linalg.inv(cov_2d)
    norm = 1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(cov_2d)))

    # Resolve the tightest direction of the Gaussian finely: the midpoint rule
    # carries a relative error of about (dr/sigma)^2 / 24, so ~200 steps per
    # sigma buys roughly 1e-6. Costs nothing in the HBR << sigma regime that
    # real conjunctions live in, where the floor of n_r binds instead.
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
    """Closed form for HBR << sigma: density at the miss point times disk area.

    Here HBR is metres and the covariance is kilometres, so the Gaussian is
    essentially flat across the disk and this is accurate to many digits.
    """
    inv = np.linalg.inv(cov_2d)
    norm = 1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(cov_2d)))
    quad = mu_2d @ inv @ mu_2d
    return float(norm * np.exp(-0.5 * quad) * np.pi * hbr_km * hbr_km)


def log10_pc_small_disk(mu_2d: np.ndarray, cov_2d: np.ndarray, hbr_km: float) -> float:
    """log10(Pc) computed directly, without ever forming Pc.

    Screened conjunctions routinely miss by tens of sigma, where Pc underflows
    to exactly zero in double precision -- a 40-sigma encounter is nominally
    ~1e-350. Working in logs keeps those labels finite and ordered, which
    matters because they are the negative class of the Phase 4 dataset.
    Valid in the HBR << sigma regime, which every real conjunction satisfies.
    """
    inv = np.linalg.inv(cov_2d)
    quad = float(mu_2d @ inv @ mu_2d)
    log_norm = -np.log(2.0 * np.pi * np.sqrt(np.linalg.det(cov_2d)))
    log_area = np.log(np.pi * hbr_km * hbr_km)
    return float((log_norm - 0.5 * quad + log_area) / np.log(10.0))
