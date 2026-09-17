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
from numpy.typing import ArrayLike

from orbital.conjunction.covariance import (
    CALIBRATED_SIGMA_R_KM,
    RTNCovariance,
    combined_covariance,
)
from orbital.conjunction.probability import log10_pc_small_disk, project_encounter
from orbital.conjunction.screening import Conjunction
from orbital.sgp4tools.tle import TLE

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


def orbit_plane_angle(
    r1: ArrayLike, v1: ArrayLike, r2: ArrayLike, v2: ArrayLike
) -> float:
    """Angle between the two orbital planes.

    Parameters
    ----------
    r1, v1
        First object's position (km) and velocity (km/s).
    r2, v2
        Second object's position and velocity, same units.

    Returns
    -------
    float
        Angle between the orbit normals, degrees, in [0, 180].
    """
    h1 = np.cross(np.asarray(r1, dtype=float), np.asarray(v1, dtype=float))
    h2 = np.cross(np.asarray(r2, dtype=float), np.asarray(v2, dtype=float))
    c = float(h1 @ h2 / (np.linalg.norm(h1) * np.linalg.norm(h2)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def features_for(c: Conjunction, t1: TLE, t2: TLE) -> dict[str, float]:
    """Cheap descriptors of one conjunction.

    No Pc, and nothing derived from it: triage has to run before the
    expensive computation to be worth anything.

    Parameters
    ----------
    c
        The screened conjunction.
    t1, t2
        Element sets of the two objects, for staleness and drag terms.

    Returns
    -------
    dict
        One value per name in :data:`FEATURES`.
    """
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

    Parameters
    ----------
    c
        The screened conjunction.
    hbr_km
        Combined hard-body radius, km.
    sigma_r_km
        Radial 1-sigma of the covariance model, km.

    Returns
    -------
    float
        Base-10 logarithm of the collision probability.
    """
    cov_model = RTNCovariance(sigma_r_km)
    cov = combined_covariance(c.r1, c.v1, c.r2, c.v2, cov_model, cov_model)
    mu_2d, cov_2d = project_encounter(c.r1 - c.r2, c.v1 - c.v2, cov)
    return log10_pc_small_disk(mu_2d, cov_2d, hbr_km)


def build(
    conjunctions: list[Conjunction], tles: list[TLE], **kwargs: float
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Assemble the design matrix and labels for a set of conjunctions.

    Parameters
    ----------
    conjunctions
        Screened conjunctions.
    tles
        Element sets; conjunctions whose objects are missing are skipped.
    **kwargs
        Passed to :func:`log10_pc_for`.

    Returns
    -------
    x : numpy.ndarray
        Features, shape (n, len(FEATURES)).
    y : numpy.ndarray
        log10(Pc) labels, shape (n,).
    names : list of str
        Column names, in matrix order.
    """
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
