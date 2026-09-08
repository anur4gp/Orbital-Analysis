"""Encounter-parameter spaces the surrogate is fitted over.

**4-D is primary.** Pc is determined by exactly four parameters, because its
defining integral has a rotation-symmetric domain (the hard-body disk) and is
covariant under uniform scaling of all lengths. That is an exact identity,
not an approximation: `tests/test_paramspace.py` verifies both invariances
and the reduction itself to ~1e-12 relative error over hundreds of random
encounters spanning the full box. At equal fill distance the reduction cuts
required design points roughly 8x, and every design point costs one expensive
Monte Carlo run.

The identity holds *given the model* -- circular hard body, Gaussian
uncertainty, short-term encounter. Slow encounters (vrel < 1 km/s, currently
filtered out) fall outside it. The raw 6-D space is retained as an ablation
to demonstrate the saving empirically rather than assert it.

## 4-D space (primary)

    0  log10 s1     sigma_1 / HBR      uncertainty size vs object size
    1  log10 r      sigma_2 / sigma_1  anisotropy, <= 1 by construction
    2  d1           mu_1 / sigma_1     miss along the major axis, in sigmas
    3  d2           mu_2 / sigma_2     miss along the minor axis, in sigmas

Lengths are in units of the hard-body radius, so HBR = 1 by construction.
d1, d2 >= 0: Pc is even in each component separately in the eigenbasis
(also verified in the tests).

## 6-D space (ablation)

Parameters, all mapped from the unit cube:

    0  mu_1        km      miss-vector component in the encounter plane
    1  mu_2        km      miss-vector component
    2  log10 s1    km      larger covariance axis
    3  log10 s2    km      smaller covariance axis
    4  theta       rad     orientation of the covariance eigenbasis, [0, pi)
    5  log10 hbr   km      combined hard-body radius

Ranges are taken from 40 real cdm_public conjunctions rebuilt in Phase 2,
widened where the real sample is thin. The HBR range deliberately spans
0.5-20 m: the RCS-derived 1-4 m is probably below operational values, and
Pc scales as HBR^2, so the surrogate must cover that uncertainty.
"""
from __future__ import annotations

import numpy as np

# --- 4-D primary space -------------------------------------------------
# Ranges from 40 real cdm_public conjunctions rebuilt in Phase 2:
# sigma_1/HBR spanned 69-1332, sigma_2/sigma_1 spanned 0.10-0.55. Widened.
# d1, d2 are capped at 6 sigma, beyond which Pc is negligible against any
# operational decision threshold.
BOUNDS_4D = np.array([
    [1.5, 3.5],     # log10(sigma_1 / HBR)   ->    32 - 3162
    [-1.5, 0.0],    # log10(sigma_2/sigma_1) -> 0.032 - 1
    [0.0, 6.0],     # mu_1 / sigma_1
    [0.0, 6.0],     # mu_2 / sigma_2
])
DIM_4D = BOUNDS_4D.shape[0]
NAMES_4D = ["log10_s1_over_hbr", "log10_aniso", "d1", "d2"]


def from_unit_cube_4d(u: np.ndarray) -> np.ndarray:
    """Map [0,1]^4 onto the reduced parameter box."""
    u = np.atleast_2d(np.asarray(u, dtype=float))
    if u.shape[1] != DIM_4D:
        raise ValueError(f"expected {DIM_4D} columns, got {u.shape[1]}")
    return BOUNDS_4D[:, 0] + u * (BOUNDS_4D[:, 1] - BOUNDS_4D[:, 0])


def unpack_4d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """One reduced parameter vector -> (mu_2d, cov_2d, hbr).

    Lengths are in hard-body radii, so hbr is exactly 1. The covariance is
    diagonal because the reduced coordinates are already the eigenbasis, and
    the anisotropy is parameterized as a ratio so sigma_2 <= sigma_1 holds by
    construction rather than needing a rejection step.
    """
    s1 = 10.0 ** x[0]
    s2 = s1 * 10.0 ** x[1]
    mu = np.array([x[2] * s1, x[3] * s2])
    return mu, np.diag([s1 ** 2, s2 ** 2]), 1.0


# --- 6-D raw space (ablation) ------------------------------------------
# (low, high) per dimension, in the units above.
BOUNDS = np.array([
    [-3.0, 3.0],                       # mu_1, km
    [-3.0, 3.0],                       # mu_2, km
    [np.log10(0.10), np.log10(1.50)],  # log10 sigma_1, km
    [np.log10(0.10), np.log10(1.50)],  # log10 sigma_2, km
    [0.0, np.pi],                      # theta, rad (eigenbasis has period pi)
    [np.log10(0.0005), np.log10(0.02)],  # log10 HBR, km  (0.5 m - 20 m)
])

DIM = BOUNDS.shape[0]
NAMES = ["mu_1", "mu_2", "log10_sigma_1", "log10_sigma_2", "theta", "log10_hbr"]


def from_unit_cube(u: np.ndarray) -> np.ndarray:
    """Map points in [0,1]^6 onto the physical parameter box."""
    u = np.atleast_2d(np.asarray(u, dtype=float))
    if u.shape[1] != DIM:
        raise ValueError(f"expected {DIM} columns, got {u.shape[1]}")
    return BOUNDS[:, 0] + u * (BOUNDS[:, 1] - BOUNDS[:, 0])


def to_unit_cube(x: np.ndarray) -> np.ndarray:
    """Inverse of `from_unit_cube`."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    return (x - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])


def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """One parameter vector -> (mu_2d, cov_2d, hbr_km).

    The covariance is rebuilt as Q diag(s1^2, s2^2) Q^T, so it is symmetric
    positive definite for every point in the box -- a box in the raw entries
    (c11, c12, c22) would not be.
    """
    mu = np.array([x[0], x[1]])
    s1, s2 = 10.0 ** x[2], 10.0 ** x[3]
    theta = x[4]
    c, s = np.cos(theta), np.sin(theta)
    q = np.array([[c, -s], [s, c]])
    cov = q @ np.diag([s1 ** 2, s2 ** 2]) @ q.T
    return mu, cov, float(10.0 ** x[5])


def reduce_to_4d(mu: np.ndarray, cov: np.ndarray, hbr: float) -> np.ndarray:
    """Collapse an encounter to the 4 parameters Pc actually depends on.

    Rotate into the covariance eigenbasis (the hard-body disk is rotation
    invariant, so the absolute orientation cannot matter), then divide every
    length by the hard-body radius (Pc is scale invariant). What survives is
    (mu_1/R, mu_2/R, sigma_1/R, sigma_2/R).
    """
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    mu_rot = vecs.T @ mu
    return np.array([mu_rot[0] / hbr, mu_rot[1] / hbr,
                     np.sqrt(vals[0]) / hbr, np.sqrt(vals[1]) / hbr])
