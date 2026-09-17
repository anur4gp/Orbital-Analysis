"""Time and Earth-orientation utilities, checked against published values."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pytest
from sgp4.api import jday

from orbital.core.constants import OMEGA_EARTH_RAD_S, WGS84_A_KM, WGS84_F
from orbital.core.frames import EarthRotation, geodetic_to_ecef, gmst_rad
from orbital.core.timescales import julian_date


def test_gmst_matches_vallado_example():
    """Vallado, Fundamentals of Astrodynamics, Example 3-5: 152.578787810 deg."""
    gmst = np.degrees(gmst_rad(datetime(1992, 8, 20, 12, 14, tzinfo=UTC)))
    assert gmst == pytest.approx(152.578787810, abs=1e-6)


@pytest.mark.parametrize(
    "epoch",
    [datetime(2000, 1, 1, 12, tzinfo=UTC), datetime(1996, 10, 26, 14, 20, tzinfo=UTC),
     datetime(2026, 9, 16, 23, 59, 59, tzinfo=UTC)],
)
def test_julian_date_matches_sgp4(epoch):
    ours = julian_date(epoch)
    theirs = jday(epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute, epoch.second)
    assert ours[0] == theirs[0]
    assert ours[1] == pytest.approx(theirs[1], abs=1e-12)


def test_j2000_epoch():
    assert sum(julian_date(datetime(2000, 1, 1, 12, tzinfo=UTC))) == 2451545.0


def test_naive_and_non_utc_epochs_are_rejected():
    with pytest.raises(ValueError):
        julian_date(datetime(2020, 1, 1))
    with pytest.raises(ValueError):
        julian_date(datetime(2020, 1, 1, tzinfo=timezone(timedelta(hours=1))))


def test_gmst_advances_at_sidereal_rate():
    earth = EarthRotation(datetime(2026, 9, 16, tzinfo=UTC))
    one_day_later = gmst_rad(datetime(2026, 9, 17, tzinfo=UTC))
    predicted = np.mod(earth.angle_rad(86400.0), 2 * np.pi)
    # IAU-82 rate vs the constant omega_E differ by ~1e-7 rad per day.
    assert predicted == pytest.approx(one_day_later, abs=1e-6)
    assert OMEGA_EARTH_RAD_S * 86400.0 > 2 * np.pi  # a sidereal day is shorter


class TestGeodetic:
    def test_equator_prime_meridian(self):
        assert np.allclose(geodetic_to_ecef(0.0, 0.0, 0.0), [WGS84_A_KM, 0, 0])

    def test_north_pole_is_at_the_polar_radius(self):
        b = WGS84_A_KM * (1 - WGS84_F)
        assert np.allclose(geodetic_to_ecef(90.0, 0.0, 0.0), [0, 0, b], atol=1e-9)

    def test_altitude_is_along_the_normal(self):
        low = geodetic_to_ecef(35.0, -117.0, 0.0)
        high = geodetic_to_ecef(35.0, -117.0, 1.0)
        assert np.linalg.norm(high - low) == pytest.approx(1.0)

    def test_rotation_is_proper(self):
        c = EarthRotation(datetime(2026, 1, 1, tzinfo=UTC)).ecef_to_eci(1234.0)
        assert np.allclose(c.T @ c, np.eye(3))
        assert np.allclose(c[:, 2], [0, 0, 1])
