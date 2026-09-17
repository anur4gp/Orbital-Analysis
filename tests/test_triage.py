"""Triage features, labels, and full-recall evaluation.

The operating point is checked on hand-built score arrays where the right
answer is countable by hand, and the features on synthetic conjunctions whose
geometry is known (coplanar, perpendicular, head-on).
"""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from orbital.conjunction.screening import Conjunction
from orbital.triage.evaluation import (
    TriageResult,
    full_recall_operating_point,
    grouped_split,
)
from orbital.triage.features import (
    DEFAULT_HBR_KM,
    FEATURES,
    build,
    features_for,
    log10_pc_for,
    orbit_plane_angle,
)

TCA = datetime(2026, 9, 4, 12, tzinfo=UTC)


def conjunction(miss_km=1.0, r1=None, v1=None, r2=None, v2=None,
                norad_i=25544, norad_j=48274) -> Conjunction:
    r1 = np.array([7000.0, 0.0, 0.0]) if r1 is None else np.asarray(r1, float)
    v1 = np.array([0.0, 7.5, 0.0]) if v1 is None else np.asarray(v1, float)
    r2 = r1 + np.array([0.0, 0.0, miss_km]) if r2 is None else np.asarray(r2, float)
    v2 = np.array([0.0, -7.5, 0.0]) if v2 is None else np.asarray(v2, float)
    return Conjunction(
        i=0, j=1, norad_i=norad_i, norad_j=norad_j, tca=TCA,
        miss_km=float(np.linalg.norm(r1 - r2)),
        vrel_km_s=float(np.linalg.norm(v1 - v2)),
        r1=r1, v1=v1, r2=r2, v2=v2,
    )


class TestOrbitPlaneAngle:
    def test_coplanar_same_direction_is_zero(self):
        r, v = [7000.0, 0, 0], [0, 7.5, 0]
        assert orbit_plane_angle(r, v, r, v) == pytest.approx(0.0)

    def test_retrograde_is_one_eighty(self):
        r = [7000.0, 0, 0]
        assert orbit_plane_angle(r, [0, 7.5, 0], r, [0, -7.5, 0]) == pytest.approx(180.0)

    def test_perpendicular_planes(self):
        r = [7000.0, 0, 0]
        assert orbit_plane_angle(r, [0, 7.5, 0], r, [0, 0, 7.5]) == pytest.approx(90.0)

    def test_clipping_guards_round_off(self):
        """Nearly parallel momenta can push the cosine past 1."""
        r, v = [7000.0, 0, 0], [0, 7.5, 0]
        angle = orbit_plane_angle(r, v, r, [0, 7.5 * (1 + 1e-16), 0])
        assert np.isfinite(angle)


class TestFeatures:
    def test_names_and_order_are_stable(self):
        """The model consumes a matrix, so column order is part of the API."""
        assert FEATURES[0] == "miss_km"
        assert len(FEATURES) == len(set(FEATURES)) == 8

    def test_values_match_the_geometry(self, iss_tle):
        c = conjunction(miss_km=2.0)
        f = features_for(c, iss_tle, iss_tle)
        assert set(f) == set(FEATURES)
        assert f["miss_km"] == pytest.approx(2.0)
        assert f["vrel_km_s"] == pytest.approx(15.0)          # head-on
        assert f["rel_incl_deg"] == pytest.approx(180.0, abs=0.1)
        assert f["alt_km"] == pytest.approx(7000.0 - 6378.137, abs=1.0)
        assert f["period_diff_min"] == pytest.approx(0.0)

    def test_uses_the_worse_element_set(self, iss_tle):
        """Staleness and drag are maxima, because the worse TLE dominates the
        error.
        """
        c = conjunction()
        f = features_for(c, iss_tle, iss_tle)
        assert f["tle_age_max_d"] == pytest.approx(abs(iss_tle.age_days(TCA)))
        assert f["bstar_max"] == pytest.approx(abs(iss_tle.bstar))
        assert f["ecc_max"] == pytest.approx(iss_tle.eccentricity)

    def test_no_feature_depends_on_pc(self, iss_tle):
        """Triage must run before any Pc is computed, or it is circular."""
        close = features_for(conjunction(miss_km=0.001), iss_tle, iss_tle)
        far = features_for(conjunction(miss_km=40.0), iss_tle, iss_tle)
        assert close["miss_km"] != far["miss_km"]
        assert close["tle_age_max_d"] == far["tle_age_max_d"]


class TestLabels:
    def test_closer_approach_gives_higher_pc(self):
        near = log10_pc_for(conjunction(miss_km=0.05))
        far = log10_pc_for(conjunction(miss_km=5.0))
        assert near > far

    def test_pc_scales_as_hbr_squared(self):
        """Small-disk limit: Pc ~ HBR^2, so doubling HBR adds log10(4)."""
        c = conjunction(miss_km=1.0)
        single = log10_pc_for(c, hbr_km=DEFAULT_HBR_KM)
        double = log10_pc_for(c, hbr_km=2 * DEFAULT_HBR_KM)
        assert double - single == pytest.approx(np.log10(4.0), abs=1e-6)

    def test_wider_covariance_lowers_a_close_approach(self):
        c = conjunction(miss_km=0.05)
        assert log10_pc_for(c, sigma_r_km=10.0) < log10_pc_for(c, sigma_r_km=0.1)

    def test_labels_stay_in_log_space(self):
        """Screened misses are routinely tens of sigma out, where Pc
        underflows; the label must stay finite and ordered.
        """
        value = log10_pc_for(conjunction(miss_km=45.0))
        assert np.isfinite(value)
        assert value < -10


class TestBuild:
    def test_assembles_matrix_labels_and_names(self, iss_tle):
        conjunctions = [conjunction(miss_km=m, norad_i=25544, norad_j=25544)
                        for m in (0.5, 1.0, 2.0)]
        x, y, names = build(conjunctions, [iss_tle])
        assert x.shape == (3, len(FEATURES))
        assert y.shape == (3,)
        assert names == FEATURES
        assert np.all(np.diff(y) < 0)          # wider miss, lower Pc

    def test_skips_conjunctions_with_a_missing_element_set(self, iss_tle):
        pair = [conjunction(norad_i=25544, norad_j=25544),
                conjunction(norad_i=25544, norad_j=99999)]
        x, y, _ = build(pair, [iss_tle])
        assert x.shape[0] == 1

    def test_hbr_is_forwarded(self, iss_tle):
        c = [conjunction(norad_i=25544, norad_j=25544)]
        _, default, _ = build(c, [iss_tle])
        _, bigger, _ = build(c, [iss_tle], hbr_km=4 * DEFAULT_HBR_KM)
        assert bigger[0] - default[0] == pytest.approx(np.log10(16.0), abs=1e-6)


class TestFullRecall:
    def test_perfect_ranker_keeps_only_the_positives(self):
        scores = np.array([0.1, 0.2, 0.9, 0.95])
        labels = np.array([0, 0, 1, 1])
        r = full_recall_operating_point(scores, labels, "perfect")
        assert r.recall == 1.0
        assert r.kept_fraction == 0.5
        assert r.n_kept == 2
        assert r.precision == 1.0
        assert r.reduction == 0.5

    def test_useless_ranker_keeps_everything(self):
        """The lowest-scoring object is a positive, so full recall forces the
        threshold to the bottom.
        """
        scores = np.array([0.1, 0.2, 0.9, 0.95])
        labels = np.array([1, 0, 0, 0])
        r = full_recall_operating_point(scores, labels, "useless")
        assert r.kept_fraction == 1.0
        assert r.recall == 1.0

    def test_recall_is_always_one_when_positives_exist(self, rng):
        for _ in range(20):
            scores = rng.random(50)
            labels = (rng.random(50) < 0.2).astype(int)
            if labels.sum() == 0:
                continue
            assert full_recall_operating_point(scores, labels, "x").recall == 1.0

    def test_no_positives_reports_nan_recall(self):
        r = full_recall_operating_point(np.array([0.1, 0.2]), np.zeros(2), "none")
        assert r.n_positive == 0
        assert r.kept_fraction == 1.0
        assert np.isnan(r.recall)

    def test_ties_at_the_threshold_are_kept(self):
        """Keeping is `score >= cutoff`, so tied negatives cannot be dropped
        without losing the positive.
        """
        scores = np.array([0.5, 0.5, 0.9])
        labels = np.array([1, 0, 0])
        assert full_recall_operating_point(scores, labels, "ties").n_kept == 3

    def test_better_ranking_keeps_less(self):
        labels = np.array([0] * 8 + [1, 1])
        good = full_recall_operating_point(np.arange(10.0), labels, "good")
        bad = full_recall_operating_point(np.arange(10.0)[::-1].copy(), labels, "bad")
        assert good.kept_fraction < bad.kept_fraction

    def test_result_reduction_complements_kept(self):
        r = TriageResult("x", 0.25, 1.0, 0.5, 25, 10)
        assert r.reduction == pytest.approx(0.75)


class TestGroupedSplit:
    def test_splits_by_group_not_by_row(self):
        groups = np.array(["day1", "day1", "day2", "day3"])
        train, test = grouped_split(groups, ["day2"])
        assert list(test) == [False, False, True, False]
        assert list(train) == [True, True, False, True]

    def test_masks_are_complementary_and_cover_everything(self):
        groups = np.array([1, 1, 2, 2, 3])
        train, test = grouped_split(groups, [1, 3])
        assert np.array_equal(train, ~test)
        assert (train | test).all()

    def test_no_group_appears_on_both_sides(self):
        """A random split would leak: the same pair recurs across a window."""
        groups = np.array(["a", "b", "a", "c", "b"])
        train, test = grouped_split(groups, ["a"])
        assert set(groups[train]).isdisjoint(set(groups[test]))

    def test_unknown_group_yields_an_empty_test_set(self):
        train, test = grouped_split(np.array([1, 2]), [9])
        assert not test.any()
        assert train.all()
