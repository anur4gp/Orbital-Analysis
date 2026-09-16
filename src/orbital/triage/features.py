"""Turn screened conjunctions into a labelled triage dataset.

Features are deliberately restricted to quantities available BEFORE any Pc
computation -- geometry and orbit descriptors that fall out of screening.
Including Pc or anything derived from it would make the classifier circular:
the whole point is to decide which pairs deserve the expensive treatment
without doing the expensive computation.

The label comes from Pc, computed with the Phase 2 covariance model and the
exact quadrature.
"""
from __future__ import annotations

import numpy as np

from orbital.conjunction.covariance import (
    CALIBRATED_SIGMA_R_KM,
    RTNCovariance,
    combined_covariance,
)
from orbital.conjunction.probability import log10_pc_small_disk, project_encounter
from orbital.conjunction.screening import Conjunction

# CelesTrak TLE pulls carry no RCS, so a single conservative hard-body radius
# is assumed for debris. Pc scales as HBR^2, so this shifts the whole label
# distribution together and is a stated assumption, not a fitted value.
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


def orbit_plane_angle(r1, v1, r2, v2) -> float:
    """Angle between the two orbital planes, in degrees."""
    h1, h2 = np.cross(r1, v1), np.cross(r2, v2)
    c = float(h1 @ h2 / (np.linalg.norm(h1) * np.linalg.norm(h2)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def features_for(c: Conjunction, t1, t2) -> dict:
    """Cheap descriptors of one conjunction. No Pc, and nothing derived from it."""
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
    """log10(Pc) for one conjunction under the Phase 2 covariance model.

    Returned in logs because screened conjunctions routinely miss by tens of
    sigma, where Pc underflows to zero and the negative class loses all its
    ordering.
    """
    cov_model = RTNCovariance(sigma_r_km)
    cov = combined_covariance(c.r1, c.v1, c.r2, c.v2, cov_model, cov_model)
    mu_2d, cov_2d = project_encounter(c.r1 - c.r2, c.v1 - c.v2, cov)
    return log10_pc_small_disk(mu_2d, cov_2d, hbr_km)


def build(conjunctions: list[Conjunction], tles: list, **kwargs):
    """Assemble (feature matrix, log10(Pc) vector, feature names)."""
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
