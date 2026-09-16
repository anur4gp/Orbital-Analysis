"""Reduction of catalog conjunctions to the 2-D encounter-plane problem.

Extracted from the Phase 2 baseline script so that both the benchmark and
the figure generator can build the same cases. It previously lived in
``run_montecarlo.py`` and was imported script-to-script, which is fine until
either script grows a ``__main__`` side effect.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbital.conjunction.covariance import (
    CALIBRATED_SIGMA_R_KM,
    RTNCovariance,
    combined_covariance,
)
from orbital.conjunction.events import (
    build_geometry,
    deduplicate,
    fetch_tles_for,
    load_events,
    tractable,
)
from orbital.conjunction.probability import hard_body_radius_km, project_encounter
from orbital.sgp4tools.spacetrack import SpaceTrack

#: Below this relative speed the short-term encounter assumption fails.
MIN_VREL_KM_S = 1.0


@dataclass
class Case:
    """One conjunction reduced to the 2-D encounter-plane problem.

    Attributes
    ----------
    event : Event
        The originating screening record.
    mu_2d : numpy.ndarray, shape (2,)
        Projected miss vector in the encounter plane, km.
    cov_2d : numpy.ndarray, shape (2, 2)
        Projected combined relative covariance, km^2.
    hbr_km : float
        Combined hard-body radius, km.
    vrel : float
        Relative speed at closest approach, km/s.
    """

    event: object
    mu_2d: np.ndarray
    cov_2d: np.ndarray
    hbr_km: float
    vrel: float


def build_cases(limit: int, sigma_r_km: float = CALIBRATED_SIGMA_R_KM,
                min_vrel_km_s: float = MIN_VREL_KM_S) -> list[Case]:
    """Fetch, rebuild and project conjunctions into encounter-plane cases.

    Parameters
    ----------
    limit : int
        Maximum number of cases to return.
    sigma_r_km : float, optional
        Radial covariance scale, km. Defaults to the calibrated value.
    min_vrel_km_s : float, optional
        Relative-speed floor, km/s. Slower encounters violate the short-term
        encounter assumption and are skipped.

    Returns
    -------
    list of Case
    """
    with SpaceTrack() as st:
        events = deduplicate(tractable(load_events(st, limit=500)))
        ids = {i for e in events[: limit * 3] for i in e.object_ids}
        tles = fetch_tles_for(st, ids)

    cov_model = RTNCovariance(sigma_r_km)
    cases: list[Case] = []
    for event in events:
        g = build_geometry(event, tles)
        if g is None or g.relative_velocity_km_s < min_vrel_km_s:
            continue
        cov = combined_covariance(g.r1, g.v1, g.r2, g.v2, cov_model, cov_model)
        mu_2d, cov_2d = project_encounter(g.r1 - g.r2, g.v1 - g.v2, cov)
        cases.append(Case(
            event=event,
            mu_2d=mu_2d,
            cov_2d=cov_2d,
            hbr_km=hard_body_radius_km(event.sat1_rcs, event.sat2_rcs),
            vrel=g.relative_velocity_km_s,
        ))
        if len(cases) >= limit:
            break
    return cases
