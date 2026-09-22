"""Building encounter-plane cases from screening records."""
from __future__ import annotations

import numpy as np
import pytest

from orbital.conjunction import cases as cases_module
from orbital.conjunction.cases import MIN_VREL_KM_S, Case, build_cases
from orbital.conjunction.events import parse_event
from orbital.sgp4tools.tle import checksum, parse_tle

EPOCH_TCA = "2024-04-26T12:25:00.000000"


def event_row(cdm_id, sat1, sat2, rcs1="LARGE", rcs2="SMALL", type1="PAYLOAD",
              type2="DEBRIS", tca=EPOCH_TCA):
    return {
        "CDM_ID": str(cdm_id), "TCA": tca, "PC": "1e-4", "MIN_RNG": "300.0",
        "SAT_1_ID": str(sat1), "SAT_2_ID": str(sat2),
        "SAT_1_NAME": "A", "SAT_2_NAME": "B",
        "SAT1_OBJECT_TYPE": type1, "SAT2_OBJECT_TYPE": type2,
        "SAT1_RCS": rcs1, "SAT2_RCS": rcs2,
        "SAT_1_EXCL_VOL": "5.0", "SAT_2_EXCL_VOL": "1.0",
    }


def retrograde_of(tle, catalog_number: int):
    """Same orbit, opposite RAAN: a fast head-on encounter geometry."""
    raan = (float(tle.line2[17:25]) + 180.0) % 360.0
    line1 = f"1 {catalog_number:05d}" + tle.line1[7:68]
    line2 = f"2 {catalog_number:05d} " + tle.line2[8:17] + f"{raan:8.4f}" + tle.line2[25:68]
    return parse_tle(line1 + str(checksum(line1)), line2 + str(checksum(line2)), "RETRO")


class StubSpaceTrack:
    """Stands in for the authenticated client; records what was asked for."""

    def __init__(self, rows, tles):
        self.rows = rows
        self.tles = tles
        self.queries = []
        self.entered = self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.exited = True

    def query(self, request_class, **kwargs):
        self.queries.append((request_class, kwargs))
        return self.rows


@pytest.fixture
def stub(monkeypatch, iss_tle):
    """Two objects on retrograde orbits: a real, fast conjunction geometry."""
    other = retrograde_of(iss_tle, 90100)
    tles = {iss_tle.catalog_number: iss_tle, other.catalog_number: other}
    rows = [event_row(1, iss_tle.catalog_number, other.catalog_number)]
    client = StubSpaceTrack(rows, tles)
    monkeypatch.setattr(cases_module, "SpaceTrack", lambda *a, **k: client)
    monkeypatch.setattr(cases_module, "load_events",
                        lambda st, limit=500, **kw: [parse_event(r) for r in st.rows])
    monkeypatch.setattr(cases_module, "fetch_tles_for", lambda st, ids, **kw: st.tles)
    return client


class TestBuildCases:
    def test_builds_a_projected_case(self, stub):
        built = build_cases(limit=5)
        assert len(built) == 1
        case = built[0]
        assert isinstance(case, Case)
        assert case.mu_2d.shape == (2,)
        assert case.cov_2d.shape == (2, 2)
        assert np.allclose(case.cov_2d, case.cov_2d.T)
        assert np.linalg.eigvalsh(case.cov_2d).min() > 0.0
        assert case.vrel > MIN_VREL_KM_S
        assert case.hbr_km > 0.0

    def test_uses_the_client_as_a_context_manager(self, stub):
        build_cases(limit=1)
        assert stub.entered and stub.exited

    def test_respects_the_limit(self, stub, iss_tle):
        # Hours apart, so not merged by deduplication.
        stub.rows = [
            event_row(i, iss_tle.catalog_number, 90100,
                      tca=f"2024-04-26T{12 + i:02d}:25:00.000000")
            for i in range(5)
        ]
        assert len(build_cases(limit=2)) == 2

    def test_skips_slow_encounters(self, stub):
        assert build_cases(limit=5, min_vrel_km_s=100.0) == []

    def test_skips_events_whose_element_sets_are_missing(self, stub, iss_tle):
        stub.rows = [event_row(9, iss_tle.catalog_number, 99999)]
        stub.tles = {iss_tle.catalog_number: iss_tle}
        assert build_cases(limit=5) == []

    def test_covariance_scale_propagates(self, stub):
        """cov_2d is quadratic in sigma_r, so a 10x scale is a 100x covariance."""
        small = build_cases(limit=1, sigma_r_km=0.1)[0]
        large = build_cases(limit=1, sigma_r_km=1.0)[0]
        # atol covers the off-diagonals, which are numerical zeros.
        assert np.allclose(large.cov_2d, 100.0 * small.cov_2d, rtol=1e-9, atol=1e-12)
        assert np.allclose(large.mu_2d, small.mu_2d)

    def test_hard_body_radius_follows_the_rcs_classes(self, stub, iss_tle):
        stub.rows = [event_row(1, iss_tle.catalog_number, 90100, rcs1="SMALL",
                               rcs2="SMALL")]
        small = build_cases(limit=1)[0].hbr_km
        stub.rows = [event_row(1, iss_tle.catalog_number, 90100, rcs1="LARGE",
                               rcs2="LARGE")]
        assert build_cases(limit=1)[0].hbr_km > small
