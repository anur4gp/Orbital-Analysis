"""cdm_public record parsing, deduplication, and rebuilt geometry.

No network: rows are dictionaries shaped like Space-Track's JSON, and the
geometry is rebuilt from the frozen ISS element set.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from orbital.conjunction.events import (
    Event,
    build_geometry,
    deduplicate,
    parse_event,
    tractable,
)


def row(cdm_id=1, tca="2026-09-04T12:00:00.000000", pc="1.5e-4", min_rng="250.0",
        sat1=25544, sat2=48274, type1="PAYLOAD", type2="DEBRIS"):
    return {
        "CDM_ID": str(cdm_id), "TCA": tca, "PC": pc, "MIN_RNG": min_rng,
        "SAT_1_ID": str(sat1), "SAT_2_ID": str(sat2),
        "SAT_1_NAME": " ISS (ZARYA) ", "SAT_2_NAME": "FENGYUN 1C DEB",
        "SAT1_OBJECT_TYPE": type1, "SAT2_OBJECT_TYPE": type2,
        "SAT1_RCS": "LARGE", "SAT2_RCS": "SMALL",
        "SAT_1_EXCL_VOL": "5.0", "SAT_2_EXCL_VOL": "1.0",
    }


def event(**kwargs) -> Event:
    return parse_event(row(**kwargs))


class TestParsing:
    def test_fields_and_types(self):
        e = event()
        assert e.cdm_id == 1
        assert e.tca == datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
        assert e.pc == pytest.approx(1.5e-4)
        assert e.min_rng_m == pytest.approx(250.0)
        assert e.object_ids == (25544, 48274)
        assert e.sat1_name == "ISS (ZARYA)"          # stripped

    def test_trailing_z_is_accepted(self):
        assert event(tca="2026-09-04T12:00:00Z").tca.tzinfo is UTC

    def test_epochs_are_timezone_aware(self):
        """Naive epochs silently compare wrong against SGP4 times."""
        assert event().tca.utcoffset() == timedelta(0)

    @pytest.mark.parametrize("bad", [None, "", "n/a"])
    def test_unparseable_numbers_become_none(self, bad):
        e = parse_event(row(pc=bad, min_rng=bad))
        assert e.pc is None and e.min_rng_m is None

    def test_missing_optional_keys_are_tolerated(self):
        minimal = {"CDM_ID": "7", "TCA": "2026-09-04T00:00:00",
                   "SAT_1_ID": "1", "SAT_2_ID": "2"}
        e = parse_event(minimal)
        assert (e.pc, e.min_rng_m, e.sat1_name, e.sat1_excl_vol) == (None, None, "", None)

    def test_combined_exclusion_volume(self):
        assert event().combined_excl_vol_km == pytest.approx(6.0)
        assert parse_event(row() | {"SAT_2_EXCL_VOL": None}).combined_excl_vol_km is None


class TestTractable:
    def test_drops_unknown_object_types(self):
        assert tractable([event(type2="UNKNOWN")]) == []

    def test_drops_analyst_objects(self):
        """Six-digit ids are analyst objects, absent from the public catalog."""
        assert tractable([event(sat2=270123)]) == []

    def test_keeps_ordinary_events(self):
        assert len(tractable([event()])) == 1


class TestDeduplicate:
    def test_collapses_the_mirrored_filing(self):
        """Every conjunction is filed twice, once with each object primary."""
        a = event(cdm_id=1, sat1=25544, sat2=48274)
        b = event(cdm_id=2, sat1=48274, sat2=25544)
        assert len(deduplicate([a, b])) == 1

    def test_collapses_refilings_within_the_tolerance(self):
        a = event(cdm_id=1, tca="2026-09-04T12:00:00")
        b = event(cdm_id=2, tca="2026-09-04T12:10:00")   # 600 s later
        assert len(deduplicate([a, b], tca_tolerance_s=900.0)) == 1

    def test_keeps_genuinely_separate_approaches(self):
        a = event(cdm_id=1, tca="2026-09-04T12:00:00")
        b = event(cdm_id=2, tca="2026-09-04T18:00:00")
        assert len(deduplicate([a, b])) == 2

    def test_keeps_different_pairs_at_the_same_time(self):
        a = event(cdm_id=1, sat1=25544, sat2=48274)
        b = event(cdm_id=2, sat1=25544, sat2=12345)
        assert len(deduplicate([a, b])) == 2

    def test_output_is_sorted_by_tca(self):
        late = event(cdm_id=1, tca="2026-09-05T00:00:00", sat2=1)
        early = event(cdm_id=2, tca="2026-09-04T00:00:00", sat2=2)
        assert [e.cdm_id for e in deduplicate([late, early])] == [2, 1]

    def test_empty_input(self):
        assert deduplicate([]) == []


class TestGeometry:
    def test_missing_tle_returns_none(self):
        assert build_geometry(event(), {}) is None

    def test_rebuilds_from_element_sets(self, iss_tle, epoch):
        """Both objects set to the ISS: the rebuilt separation is exactly zero,
        so the geometry wiring itself is what is being checked.
        """
        e = event(tca=epoch.replace(tzinfo=None).isoformat(), sat1=25544, sat2=25544)
        geometry = build_geometry(e, {25544: iss_tle})
        assert geometry is not None
        assert geometry.miss_km == pytest.approx(0.0)
        assert geometry.relative_velocity_km_s == pytest.approx(0.0)
        assert np.allclose(geometry.r1, geometry.r2)
        assert geometry.tle_age1_days == pytest.approx(iss_tle.age_days(e.tca))

    def test_reported_miss_is_converted_to_km(self, iss_tle, epoch):
        e = event(tca=epoch.replace(tzinfo=None).isoformat(), sat1=25544, sat2=25544,
                  min_rng="250.0")
        geometry = build_geometry(e, {25544: iss_tle})
        assert geometry.reported_miss_km == pytest.approx(0.25)
        assert geometry.miss_error_km == pytest.approx(geometry.miss_km - 0.25)

    def test_missing_reported_miss_propagates_as_none(self, iss_tle, epoch):
        e = event(tca=epoch.replace(tzinfo=None).isoformat(), sat1=25544, sat2=25544,
                  min_rng=None)
        geometry = build_geometry(e, {25544: iss_tle})
        assert geometry.reported_miss_km is None
        assert geometry.miss_error_km is None
