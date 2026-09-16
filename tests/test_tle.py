"""TLE parsing: column positions, implied decimals, and epoch handling.

Assertions are against the format specification rather than golden output,
so a regression points at the rule that broke.
"""
from __future__ import annotations

from datetime import UTC

import pytest

from orbital.sgp4tools.tle import (
    _decimal_point_assumed,
    check_line,
    checksum,
    parse_tle,
    parse_tle_file,
)


class TestChecksum:
    def test_fixture_lines_validate(self, iss_lines):
        _, l1, l2 = iss_lines
        assert check_line(l1)
        assert check_line(l2)

    def test_corrupted_line_rejected(self, iss_lines):
        _, l1, _ = iss_lines
        assert not check_line(l1[:20] + "9" + l1[21:])

    def test_checksum_is_a_single_digit(self, iss_lines):
        _, l1, _ = iss_lines
        assert 0 <= checksum(l1) <= 9

    def test_wrong_length_rejected(self, iss_lines):
        _, l1, _ = iss_lines
        assert not check_line(l1[:-1])


@pytest.mark.parametrize(
    ("field", "expected"),
    [("30074-3", 0.30074e-3), ("00000-0", 0.0), ("-11606-4", -0.11606e-4)],
)
def test_assumed_decimal_exponential(field, expected):
    """TLEs omit both the leading decimal point and the exponent marker."""
    assert _decimal_point_assumed(field) == pytest.approx(expected)


class TestParsedFields:
    def test_identity(self, iss_tle):
        assert iss_tle.name == "ISS (ZARYA)"
        assert iss_tle.catalog_number == 25544
        assert iss_tle.classification == "U"
        assert iss_tle.international_designator == "98067A"

    def test_line1_drag_terms(self, iss_tle):
        assert iss_tle.bstar == pytest.approx(0.30074e-3)
        assert iss_tle.mean_motion_dot == pytest.approx(0.00016717)
        assert iss_tle.element_set_number == 999

    @pytest.mark.parametrize(
        ("attr", "expected"),
        [
            ("inclination", 51.6416),
            ("raan", 247.4627),
            ("eccentricity", 0.0006703),
            ("arg_perigee", 130.5360),
            ("mean_anomaly", 325.0288),
            ("mean_motion", 15.49309239),
        ],
    )
    def test_orbital_elements(self, iss_tle, attr, expected):
        assert getattr(iss_tle, attr) == pytest.approx(expected)

    def test_revolution_number(self, iss_tle):
        assert iss_tle.revolution_number == 44847

    def test_period_is_consistent_with_mean_motion(self, iss_tle):
        """Period and mean motion are the same quantity in different units."""
        assert iss_tle.period_minutes == pytest.approx(1440.0 / iss_tle.mean_motion)

    def test_period_is_physical_for_the_iss(self, iss_tle):
        assert 92.0 < iss_tle.period_minutes < 93.0


class TestEpoch:
    def test_epoch_decodes_to_the_right_day(self, iss_tle):
        """24117.51782528 is 2024, day 117 = 26 April, at 0.51782528 of a day."""
        assert (iss_tle.epoch.year, iss_tle.epoch.month, iss_tle.epoch.day) == (2024, 4, 26)

    def test_epoch_fraction_decodes_to_the_right_time(self, iss_tle):
        assert (iss_tle.epoch.hour, iss_tle.epoch.minute) == (12, 25)

    def test_epoch_is_timezone_aware_utc(self, iss_tle):
        assert iss_tle.epoch.tzinfo == UTC

    def test_age_is_zero_at_its_own_epoch(self, iss_tle):
        assert iss_tle.age_days(iss_tle.epoch) == pytest.approx(0.0, abs=1e-9)


class TestFileParsing:
    def test_with_name_line(self, iss_lines):
        name, l1, l2 = iss_lines
        assert len(parse_tle_file(f"{name}\n{l1}\n{l2}\n")) == 1

    def test_without_name_line(self, iss_lines):
        _, l1, l2 = iss_lines
        assert len(parse_tle_file(f"{l1}\n{l2}\n")) == 1

    def test_multiple_records(self, iss_lines):
        name, l1, l2 = iss_lines
        assert len(parse_tle_file(f"{name}\n{l1}\n{l2}\n{name}\n{l1}\n{l2}\n")) == 2


class TestRejectsMalformedInput:
    def test_mismatched_catalog_numbers(self, iss_lines):
        """Line 1 and line 2 must agree on which object they describe.

        The altered line has its check digit recomputed, so this exercises the
        catalog-number comparison rather than tripping the checksum first --
        which is what an earlier version of this test did, passing for the
        wrong reason.
        """
        _, l1, l2 = iss_lines
        altered = l2.replace("2 25544", "2 25545")[:68]
        altered += str(checksum(altered))
        assert check_line(altered), "the altered line must itself be well formed"
        with pytest.raises(ValueError, match="catalog numbers"):
            parse_tle(l1, altered)

    def test_swapped_lines(self, iss_lines):
        _, l1, l2 = iss_lines
        with pytest.raises(ValueError):
            parse_tle(l2, l1)
