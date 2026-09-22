"""Triage dataset: pre-Pc features and log10(Pc) labels.

Features use only screening outputs; anything derived from Pc would be circular.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from orbital.conjunction.covariance import (
    CALIBRATED_SIGMA_R_KM,
    RTNCovariance,
    combined_covariance,
)
from orbital.conjunction.probability import log10_pc_small_disk, project_encounter
from orbital.conjunction.screening import Conjunction
from orbital.sgp4tools.tle import TLE

# Assumed debris HBR (CelesTrak TLEs carry no RCS).
DEFAULT_HBR_KM = 0.002

FEATURES = [
    "miss_km",          # closest approach distance
    "vrel_km_s",        # relative speed at TCA
    "alt_km",           # mean altitude at TCA
    "rel_incl_deg",     # angle between the two orbit planes
    "tle_age_max_d",    # staleness of the worse element set
    "period_diff_min",  # difference in orbital period
    "ecc_max",          # larger of the two eccentricities
    "bstar_max",        # larger drag term, a proxy for propagation error
]


def orbit_plane_angle(
    r1: ArrayLike, v1: ArrayLike, r2: ArrayLike, v2: ArrayLike
) -> float:
    """Angle between the two orbit normals, degrees."""
    h1 = np.cross(np.asarray(r1, dtype=float), np.asarray(v1, dtype=float))
    h2 = np.cross(np.asarray(r2, dtype=float), np.asarray(v2, dtype=float))
    c = float(h1 @ h2 / (np.linalg.norm(h1) * np.linalg.norm(h2)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def features_for(c: Conjunction, t1: TLE, t2: TLE) -> dict[str, float]:
    """Feature values for one conjunction, keyed by :data:`FEATURES`."""
    return {
        "miss_km": c.miss_km,
        "vrel_km_s": c.vrel_km_s,
        "alt_km": float(0.5 * (np.linalg.norm(c.r1) + np.linalg.norm(c.r2)) - 6378.137),
        "rel_incl_deg": orbit_plane_angle(c.r1, c.v1, c.r2, c.v2),
        "tle_age_max_d": float(max(abs(t1.age_days(c.tca)), abs(t2.age_days(c.tca)))),
        "period_diff_min": abs(t1.period_minutes - t2.period_minutes),
        "ecc_max": max(t1.eccentricity, t2.eccentricity),
        "bstar_max": max(abs(t1.bstar), abs(t2.bstar)),
    }


def log10_pc_for(c: Conjunction, hbr_km: float = DEFAULT_HBR_KM,
                 sigma_r_km: float = CALIBRATED_SIGMA_R_KM) -> float:
    """log10(Pc) under the RTN covariance model (logs avoid underflow)."""
    cov_model = RTNCovariance(sigma_r_km)
    cov = combined_covariance(c.r1, c.v1, c.r2, c.v2, cov_model, cov_model)
    mu_2d, cov_2d = project_encounter(c.r1 - c.r2, c.v1 - c.v2, cov)
    return log10_pc_small_disk(mu_2d, cov_2d, hbr_km)


def build(
    conjunctions: list[Conjunction], tles: list[TLE], **kwargs: float
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Feature matrix, log10(Pc) labels and column names; skips pairs missing a TLE."""
    rows, pcs = [], []
    by_norad = {t.catalog_number: t for t in tles}
    for c in conjunctions:
        t1, t2 = by_norad.get(c.norad_i), by_norad.get(c.norad_j)
        if t1 is None or t2 is None:
            continue
        rows.append(features_for(c, t1, t2))
        pcs.append(log10_pc_for(c, **kwargs))
    x = np.array([[r[f] for f in FEATURES] for r in rows])
    return x, np.array(pcs), FEATURES
