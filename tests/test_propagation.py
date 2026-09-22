"""SGP4 wrappers, checked against orbital mechanics rather than stored output."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from orbital.sgp4tools.propagation import (
    MU_EARTH,
    R_EARTH_EQ,
    PropagationError,
    _to_jd,
    altitude_km,
    apsides_km,
    period_minutes,
    propagate,
    propagate_series,
    radius_km,
    satrec_from_tle,
    semi_major_axis_km,
)
from orbital.sgp4tools.tle import checksum, parse_tle


def with_checksum(line: str) -> str:
    """Line with a recomputed check digit, so the parser accepts it."""
    body = line[:68]
    return body + str(checksum(body))


# 20 rev/day puts perigee below the surface; SGP4 must return an error.
DECAYED_L1 = with_checksum(
    "1 00900U 64063C   24117.50000000  .00000000  00000-0  00000-0 0  999"
)
DECAYED_L2 = with_checksum(
    "2 00900  90.0000   0.0000 0000000   0.0000   0.0000 20.00000000    0"
)


@pytest.fixture
def iss_sat(iss_tle):
    return satrec_from_tle(iss_tle)


class TestTimeConversion:
    def test_naive_and_aware_agree(self):
        naive = datetime(2024, 4, 26, 12, 30, 15)
        aware = naive.replace(tzinfo=UTC)
        assert _to_jd(naive) == _to_jd(aware)

    def test_offset_is_converted_not_ignored(self):
        from datetime import timezone
        plus_two = datetime(2024, 4, 26, 14, 30, 15, tzinfo=timezone(timedelta(hours=2)))
        assert _to_jd(plus_two) == _to_jd(datetime(2024, 4, 26, 12, 30, 15))

    def test_one_hour_is_one_twentyfourth_of_a_day(self):
        a = sum(_to_jd(datetime(2024, 4, 26, 0)))
        b = sum(_to_jd(datetime(2024, 4, 26, 1)))
        assert b - a == pytest.approx(1 / 24)


class TestDerivedElements:
    def test_period_matches_the_iss(self, iss_sat):
        assert 92.0 < period_minutes(iss_sat) < 93.0

    def test_semi_major_axis_satisfies_keplers_third_law(self, iss_sat):
        a = semi_major_axis_km(iss_sat)
        t_s = period_minutes(iss_sat) * 60.0
        assert t_s == pytest.approx(2 * np.pi * np.sqrt(a**3 / MU_EARTH), rel=1e-12)

    def test_apsides_bracket_the_mean_altitude(self, iss_sat):
        perigee, apogee = apsides_km(iss_sat)
        mean_alt = semi_major_axis_km(iss_sat) - R_EARTH_EQ
        assert perigee < mean_alt < apogee
        assert 380.0 < perigee < 430.0
        assert 390.0 < apogee < 440.0

    def test_apsides_reduce_to_the_circular_case(self, iss_lines):
        """With e = 0 both apsides equal a - R."""
        name, line1, line2 = iss_lines
        t = parse_tle(line1, with_checksum(line2[:26] + "0000000" + line2[33:]), name)
        sat = satrec_from_tle(t)
        perigee, apogee = apsides_km(sat)
        assert perigee == pytest.approx(apogee)
        assert perigee == pytest.approx(semi_major_axis_km(sat) - R_EARTH_EQ)


class TestPropagation:
    def test_state_is_physically_consistent(self, iss_sat, epoch):
        r, v = propagate(iss_sat, epoch)
        assert 6700.0 < np.linalg.norm(r) < 6850.0
        assert 7.5 < np.linalg.norm(v) < 7.8

    def test_energy_is_conserved_over_three_days(self, iss_sat, epoch):
        """Vis-viva; loose bound since SGP4 includes drag and J2."""
        times, r, v, _ = propagate_series(iss_sat, epoch, 3 * 1440.0, 10.0)
        energy = 0.5 * np.einsum("ij,ij->i", v, v) - MU_EARTH / radius_km(r)
        assert np.ptp(energy) / abs(energy.mean()) < 0.01

    def test_altitude_stays_in_the_iss_band(self, iss_sat, epoch):
        _, r, _, _ = propagate_series(iss_sat, epoch, 1440.0, 5.0)
        alt = altitude_km(r)
        assert 380.0 < alt.min() and alt.max() < 450.0

    def test_period_measured_from_radius_minima(self, iss_sat, epoch):
        """The sampled trajectory's own period must match the element set's."""
        times, r, _, _ = propagate_series(iss_sat, epoch, 600.0, 0.5)
        rad = radius_km(r)
        minima = np.flatnonzero((rad[1:-1] < rad[:-2]) & (rad[1:-1] <= rad[2:])) + 1
        measured = np.mean(np.diff(minima)) * 0.5
        assert measured == pytest.approx(period_minutes(iss_sat), abs=0.3)

    def test_inclination_has_no_secular_drift(self, iss_sat, epoch):
        """J2 makes inclination oscillate (~0.02 deg) but not drift."""
        _, r, v, _ = propagate_series(iss_sat, epoch, 1440.0, 5.0)
        h = np.cross(r, v)
        incl = np.degrees(np.arccos(h[:, 2] / np.linalg.norm(h, axis=1)))
        half = incl.size // 2
        assert np.ptp(incl) < 0.05
        assert incl[:half].mean() == pytest.approx(incl[half:].mean(), abs=0.005)

    def test_series_grid_is_inclusive_and_regular(self, iss_sat, epoch):
        times, r, v, errors = propagate_series(iss_sat, epoch, 10.0, 2.0)
        assert len(times) == 6
        assert times[0] == epoch
        assert times[-1] == epoch + timedelta(minutes=10)
        assert r.shape == v.shape == (6, 3)
        assert not errors.any()

    def test_series_matches_single_calls(self, iss_sat, epoch):
        times, r, v, _ = propagate_series(iss_sat, epoch, 4.0, 2.0)
        for t, ri, vi in zip(times, r, v, strict=True):
            r_one, v_one = propagate(iss_sat, t)
            assert np.allclose(ri, r_one)
            assert np.allclose(vi, v_one)


class TestErrorHandling:
    def test_decayed_orbit_raises(self):
        t = parse_tle(DECAYED_L1, DECAYED_L2, "DECAYED")
        with pytest.raises(PropagationError, match="error code"):
            propagate(satrec_from_tle(t), datetime(2024, 4, 26, tzinfo=UTC))

    def test_series_can_report_instead_of_raising(self):
        t = parse_tle(DECAYED_L1, DECAYED_L2, "DECAYED")
        sat = satrec_from_tle(t)
        with pytest.raises(PropagationError):
            propagate_series(sat, datetime(2024, 4, 26, tzinfo=UTC), 10.0, 5.0)
        _, _, _, errors = propagate_series(
            sat, datetime(2024, 4, 26, tzinfo=UTC), 10.0, 5.0, strict=False
        )
        assert np.any(errors != 0)


class TestVectorHelpers:
    def test_radius_accepts_one_vector_or_a_stack(self):
        assert radius_km([3.0, 4.0, 0.0]) == pytest.approx(5.0)
        assert np.allclose(radius_km([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]]), [5.0, 2.0])

    def test_altitude_is_radius_minus_earth_radius(self):
        assert altitude_km([R_EARTH_EQ + 400.0, 0.0, 0.0]) == pytest.approx(400.0)
