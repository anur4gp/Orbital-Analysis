"""Catalog screening: the geometry of the sieve, not stored counts.

``find_candidates`` is driven with synthetic straight-line motion, where the
closest approach is known in closed form, so the analytic filter can be
checked against the exact answer. The SGP4 stages are exercised on the frozen
ISS element set plus shifted copies of it, which manufactures conjunctions
with a known miss distance without any network access.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest

from orbital.conjunction.screening import (
    MAX_VREL_KM_S,
    coarse_gate_km,
    find_candidates,
    propagate_catalog,
    refine,
    screen,
)
from orbital.sgp4tools.propagation import satrec_from_tle
from orbital.sgp4tools.tle import checksum, parse_tle


def linear_pair(miss_km: float, approach_speed_km_s: float, n_times: int = 21,
                step_s: float = 60.0):
    """Two objects on straight lines whose closest approach is ``miss_km``.

    Object 0 sits still; object 1 passes it along +x, offset in z. Closest
    approach falls exactly on the middle sample.
    """
    t = (np.arange(n_times) - n_times // 2) * step_s
    r = np.zeros((2, n_times, 3))
    v = np.zeros((2, n_times, 3))
    r[1, :, 0] = approach_speed_km_s * t
    r[1, :, 2] = miss_km
    v[1, :, 0] = approach_speed_km_s
    return r, v


class TestCoarseGate:
    def test_gate_covers_the_worst_case_closing_distance(self):
        """A pair closes by at most MAX_VREL * step between samples, so half
        that must be added to the target threshold.
        """
        assert coarse_gate_km(60.0, 50.0) == pytest.approx(50.0 + 0.5 * MAX_VREL_KM_S * 60)

    def test_gate_grows_with_the_step(self):
        assert coarse_gate_km(120.0, 50.0) > coarse_gate_km(60.0, 50.0)

    def test_documented_width(self):
        """CLAUDE.md quotes 530 km at 60 s; the rule must still give that."""
        assert coarse_gate_km(60.0, 50.0) == pytest.approx(530.0)


class TestFindCandidates:
    def test_finds_a_close_approach_the_grid_never_samples(self):
        """The whole point of the analytic filter: at 10 km/s and a 60 s grid
        the pair is never sampled closer than 300 km, yet it passes within
        1 km.
        """
        r, v = linear_pair(miss_km=1.0, approach_speed_km_s=10.0, step_s=60.0)
        separations = np.linalg.norm(r[0] - r[1], axis=1)
        assert separations.min() == pytest.approx(1.0)  # midpoint sample
        hits = find_candidates(r, v, coarse_gate_km(60.0, 50.0), 50.0, 60.0)
        assert [(i, j) for i, j, _ in hits] == [(0, 1)]

    def test_rejects_a_pair_that_stays_far_apart(self):
        r, v = linear_pair(miss_km=400.0, approach_speed_km_s=10.0)
        assert find_candidates(r, v, coarse_gate_km(60.0, 50.0), 50.0, 60.0) == []

    def test_candidate_time_index_is_the_local_minimum(self):
        r, v = linear_pair(miss_km=5.0, approach_speed_km_s=8.0, n_times=21)
        hits = find_candidates(r, v, coarse_gate_km(60.0, 50.0), 50.0, 60.0)
        assert [t for _, _, t in hits] == [10]

    def test_keeps_both_of_two_separate_approaches(self):
        """A pair can have several conjunctions in one window; every local
        minimum must survive, not just the deepest.
        """
        n = 41
        r = np.zeros((2, n, 3))
        v = np.zeros((2, n, 3))
        t = np.arange(n) * 60.0
        # Separation with minima near indices 10 and 30.
        r[1, :, 0] = 20.0 * np.cos(2 * np.pi * t / (20 * 60.0))
        v[1, :, 0] = -20.0 * (2 * np.pi / (20 * 60.0)) * np.sin(2 * np.pi * t / (20 * 60.0))
        hits = find_candidates(r, v, 500.0, 50.0, 60.0)
        assert len(hits) >= 2

    def test_zero_relative_velocity_does_not_divide_by_zero(self):
        """Co-moving objects: t* is undefined, and the code must not crash."""
        n = 5
        r = np.zeros((2, n, 3))
        r[1, :, 2] = 10.0
        v = np.zeros((2, n, 3))
        hits = find_candidates(r, v, 500.0, 50.0, 60.0)
        assert hits == []

    def test_chunking_does_not_change_the_result(self):
        r, v = linear_pair(miss_km=2.0, approach_speed_km_s=9.0)
        whole = find_candidates(r, v, 530.0, 50.0, 60.0)
        chunked = find_candidates(r, v, 530.0, 50.0, 60.0, pair_chunk=1)
        assert whole == chunked


def shifted_copy(tle, seconds: float, catalog_number: int):
    """The same orbit with its mean anomaly advanced by ``seconds``.

    Two objects on one orbit separated in phase give a known, controllable
    along-track separation -- a conjunction with an answer to check against.
    """
    mean_motion = float(tle.line2[52:63])           # rev/day
    mean_anomaly = float(tle.line2[43:51])          # deg
    advanced = (mean_anomaly + 360.0 * mean_motion * seconds / 86400.0) % 360.0
    line2 = (f"2 {catalog_number:05d} " + tle.line2[8:43]
             + f"{advanced:8.4f}" + tle.line2[51:68])
    line1 = f"1 {catalog_number:05d}" + tle.line1[7:68]
    return parse_tle(line1 + str(checksum(line1)), line2 + str(checksum(line2)), "COPY")


class TestSGP4Stages:
    def test_propagate_catalog_shapes_and_mask(self, iss_tle, epoch):
        tles = [iss_tle, shifted_copy(iss_tle, 30.0, 90001)]
        offsets, r, v, ok = propagate_catalog(tles, epoch, 10.0, 60.0)
        assert offsets[0] == 0.0 and offsets[-1] == pytest.approx(600.0)
        assert r.shape == v.shape == (2, offsets.size, 3)
        assert ok.all()

    def test_refine_recovers_the_true_minimum(self, iss_tle, epoch):
        """Two objects 2 s apart in phase: they never collide, and the refined
        miss distance must match their along-track separation (~15 km).
        """
        other = shifted_copy(iss_tle, 2.0, 90002)
        sat_a, sat_b = satrec_from_tle(iss_tle), satrec_from_tle(other)
        result = refine(sat_a, sat_b, epoch, 60.0)
        assert result is not None
        tca, miss, vrel, r1, v1, r2, v2 = result
        assert np.linalg.norm(r1 - r2) == pytest.approx(miss)
        assert 10.0 < miss < 20.0
        assert vrel < 0.1                      # same orbit: nearly co-moving

    def test_refine_finds_a_lower_separation_than_its_starting_point(self,
                                                                     iss_tle, epoch):
        other = shifted_copy(iss_tle, 45.0, 90003)
        sat_a, sat_b = satrec_from_tle(iss_tle), satrec_from_tle(other)
        start = epoch + timedelta(seconds=25.0)
        _, miss, _, _, _, _, _ = refine(sat_a, sat_b, start, 60.0)
        jd_pair = refine(sat_a, sat_b, start, 0.0, iterations=1)
        assert miss <= jd_pair[1] + 1e-9

    def test_refine_reports_failure_on_a_decayed_object(self, iss_tle, epoch):
        bad1 = "1 00900U 64063C   24117.50000000  .00000000  00000-0  00000-0 0  999"
        bad2 = "2 00900  90.0000   0.0000 0000000   0.0000   0.0000 20.00000000    0"
        decayed = parse_tle(bad1 + str(checksum(bad1)), bad2 + str(checksum(bad2)), "X")
        result = refine(satrec_from_tle(iss_tle), satrec_from_tle(decayed), epoch, 60.0)
        assert result is None


@pytest.mark.slow
class TestScreenEndToEnd:
    def test_finds_the_planted_conjunction(self, iss_tle, epoch, capsys):
        """A copy of the ISS 3 s behind it is a 22 km 'conjunction'; the full
        sieve must report it, with consistent geometry.
        """
        tles = [iss_tle, shifted_copy(iss_tle, 3.0, 90010)]
        found = screen(tles, epoch, minutes=95.0, step_seconds=60.0,
                       refine_threshold_km=50.0, verbose=True)
        assert found, "planted conjunction was missed"
        c = found[0]
        assert {c.norad_i, c.norad_j} == {25544, 90010}
        assert c.miss_km == pytest.approx(float(np.linalg.norm(c.r1 - c.r2)), rel=1e-9)
        assert c.miss_km < 50.0
        assert epoch <= c.tca <= epoch + timedelta(minutes=95)
        assert "coarse gate" in capsys.readouterr().out

    def test_widely_separated_orbits_produce_nothing(self, iss_tle, epoch):
        """Same plane, half an orbit apart: no approach within 50 km."""
        tles = [iss_tle, shifted_copy(iss_tle, 2790.0, 90011)]
        assert screen(tles, epoch, minutes=95.0, step_seconds=60.0,
                      refine_threshold_km=50.0, verbose=False) == []
