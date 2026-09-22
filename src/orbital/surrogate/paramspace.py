"""Encounter-parameter spaces for the surrogate.

Pc depends on only four parameters (rotation invariance of the disk and
scale covariance of all lengths), verified in ``tests/test_paramspace.py``.
The raw 6-D space is kept as an ablation.

4-D (lengths in hard-body radii, so HBR = 1)::

    0  log10 s1   sigma_1 / HBR
    1  log10 r    sigma_2 / sigma_1   (<= 1)
    2  d1         mu_1 / sigma_1      (Pc is even, so >= 0)
    3  d2         mu_2 / sigma_2

6-D::

    0, 1  mu_1, mu_2   km
    2, 3  log10 s1, s2 km
    4     theta        rad, covariance orientation in [0, pi)
    5     log10 hbr    km
"""
from __future__ import annotations

import numpy as np

# Widened from 40 rebuilt cdm_public events (sigma_1/HBR 69-1332, ratio 0.10-0.55).
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
    """One reduced parameter vector -> (mu_2d, diagonal cov_2d, hbr = 1)."""
    s1 = 10.0 ** x[0]
    s2 = s1 * 10.0 ** x[1]
    mu = np.array([x[2] * s1, x[3] * s2])
    return mu, np.diag([s1 ** 2, s2 ** 2]), 1.0


BOUNDS = np.array([
    [-3.0, 3.0],                       # mu_1, km
    [-3.0, 3.0],                       # mu_2, km
    [np.log10(0.10), np.log10(1.50)],  # log10 sigma_1, km
    [np.log10(0.10), np.log10(1.50)],  # log10 sigma_2, km
    [0.0, np.pi],                      # theta, rad (eigenbasis has period pi)
    [np.log10(0.0005), np.log10(0.02)],  # log10 HBR, km (0.5 - 20 m)
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
    """One parameter vector -> (mu_2d, cov_2d = Q diag(s1^2, s2^2) Q^T, hbr_km)."""
    mu = np.array([x[0], x[1]])
    s1, s2 = 10.0 ** x[2], 10.0 ** x[3]
    theta = x[4]
    c, s = np.cos(theta), np.sin(theta)
    q = np.array([[c, -s], [s, c]])
    cov = q @ np.diag([s1 ** 2, s2 ** 2]) @ q.T
    return mu, cov, float(10.0 ** x[5])


def reduce_to_4d(mu: np.ndarray, cov: np.ndarray, hbr: float) -> np.ndarray:
    """Rotate to the covariance eigenbasis and scale by HBR.

    Returns ``(mu_1/R, mu_2/R, sigma_1/R, sigma_2/R)``.
    """
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    mu_rot = vecs.T @ mu
    return np.array([mu_rot[0] / hbr, mu_rot[1] / hbr,
                     np.sqrt(vals[0]) / hbr, np.sqrt(vals[1]) / hbr])
