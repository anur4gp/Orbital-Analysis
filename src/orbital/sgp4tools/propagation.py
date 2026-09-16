"""SGP4 propagation.

TLE mean elements are defined by the SGP4 theory itself, so they must be
propagated with SGP4 -- a two-body Kepler propagator will not reproduce them.
All state vectors below are TEME frame, km and km/s.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
from sgp4.api import Satrec, jday

from orbital.sgp4tools.tle import TLE

MU_EARTH = 398600.4418      # km^3/s^2
R_EARTH_EQ = 6378.137       # km, WGS-72/84 equatorial radius


class PropagationError(RuntimeError):
    """SGP4 returned a nonzero error code (decayed orbit, bad elements, ...)."""


def satrec_from_tle(t: TLE) -> Satrec:
    return Satrec.twoline2rv(t.line1, t.line2)


def _to_jd(when: datetime) -> tuple[float, float]:
    """UTC datetime -> (jd, fr). Naive datetimes are assumed to be UTC."""
    if when.tzinfo is not None:
        when = when.astimezone(UTC).replace(tzinfo=None)
    seconds = when.second + when.microsecond * 1e-6
    return jday(when.year, when.month, when.day, when.hour, when.minute, seconds)


def propagate(sat: Satrec, when: datetime) -> tuple[np.ndarray, np.ndarray]:
    """State vector at a single time. Raises PropagationError if error != 0."""
    jd, fr = _to_jd(when)
    error, r, v = sat.sgp4(jd, fr)
    if error != 0:
        raise PropagationError(f"sgp4 error code {error} at {when.isoformat()}")
    return np.array(r), np.array(v)


def propagate_series(
    sat: Satrec,
    start: datetime,
    duration_minutes: float,
    step_minutes: float = 1.0,
    strict: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Propagate over a time grid.

    Returns (times, positions, velocities, errors) with positions/velocities
    shaped (n, 3). With strict=True any nonzero error code raises; otherwise
    the error codes are returned for the caller to mask on.
    """
    offsets = np.arange(0.0, duration_minutes + 0.5 * step_minutes, step_minutes)
    times = np.array([start + timedelta(minutes=float(m)) for m in offsets])

    jd_fr = np.array([_to_jd(t) for t in times])
    errors, r, v = sat.sgp4_array(np.ascontiguousarray(jd_fr[:, 0]),
                                  np.ascontiguousarray(jd_fr[:, 1]))
    if strict and np.any(errors != 0):
        bad = int(np.flatnonzero(errors != 0)[0])
        raise PropagationError(
            f"sgp4 error code {int(errors[bad])} at {times[bad].isoformat()}"
        )
    return times, r, v, errors


def radius_km(r: np.ndarray) -> np.ndarray:
    """Geocentric distance from a position vector (or an (n, 3) stack)."""
    return np.linalg.norm(np.atleast_2d(r), axis=1)


def altitude_km(r: np.ndarray) -> np.ndarray:
    """Altitude above a *spherical* Earth. Approximate by up to ~21 km at the
    poles because it ignores oblateness -- fine for sanity checks, not for
    geolocation."""
    return radius_km(r) - R_EARTH_EQ


def period_minutes(sat: Satrec) -> float:
    """Orbital period from the Kozai mean motion (rad/min)."""
    return 2.0 * np.pi / sat.no_kozai


def semi_major_axis_km(sat: Satrec) -> float:
    """a = (mu / n^2)^(1/3), with n converted from rad/min to rad/s."""
    n = sat.no_kozai / 60.0
    return (MU_EARTH / n**2) ** (1.0 / 3.0)


def apsides_km(sat: Satrec) -> tuple[float, float]:
    """(perigee, apogee) altitudes above a spherical Earth."""
    a = semi_major_axis_km(sat)
    return a * (1 - sat.ecco) - R_EARTH_EQ, a * (1 + sat.ecco) - R_EARTH_EQ
