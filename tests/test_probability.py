"""Collision probability: encounter-plane projection and the three estimators.

The Monte Carlo sampler and the polar quadrature share no code path, so
agreement between them is a genuine cross-check rather than a tautology.
"""
from __future__ import annotations

import numpy as np
import pytest

from orbital.conjunction.probability import (
    encounter_plane_basis,
    hard_body_radius_km,
    log10_pc_small_disk,
    pc_analytic,
    pc_monte_carlo,
    pc_small_disk,
    project_encounter,
)

COV_2D = np.array([[1.0, 0.3], [0.3, 0.5]])


@pytest.mark.parametrize(
    "v_rel",
    [[7.5, 0.0, 0.0], [0.0, 0.0, 12.0], [3.0, -4.0, 5.0], [1e-3, 0.0, 0.0]],
    ids=["along-x", "along-z", "oblique", "very-slow"],
)
def test_encounter_basis_spans_the_plane_normal_to_relative_velocity(v_rel):
    v = np.array(v_rel, dtype=float)
    b = encounter_plane_basis(v)
    assert np.allclose(b.T @ b, np.eye(2), atol=1e-12)
    assert np.allclose(b.T @ (v / np.linalg.norm(v)), 0.0, atol=1e-12)


class TestProjection:
    def test_drops_exactly_the_along_velocity_component(self):
        dr = np.array([1.0, 2.0, 3.0])
        mu, _ = project_encounter(dr, np.array([0.0, 0.0, 10.0]), np.diag([4.0, 9.0, 16.0]))
        assert np.linalg.norm(mu) == pytest.approx(np.linalg.norm(dr[:2]))

    def test_yields_a_valid_2x2_covariance(self):
        _, cov = project_encounter(np.array([1.0, 2.0, 3.0]),
                                   np.array([0.0, 0.0, 10.0]),
                                   np.diag([4.0, 9.0, 16.0]))
        assert cov.shape == (2, 2)
        assert np.allclose(cov, cov.T)
        assert np.all(np.linalg.eigvalsh(cov) > 0)


class TestEstimatorsAgree:
    @pytest.mark.parametrize("mu", [[0.0, 0.0], [0.4, -0.2], [1.5, 1.0]])
    def test_quadrature_matches_the_small_disk_closed_form(self, mu):
        """For HBR << sigma the Gaussian is flat across the disk."""
        mu = np.array(mu)
        assert pc_analytic(mu, COV_2D, 1e-3) == pytest.approx(
            pc_small_disk(mu, COV_2D, 1e-3), rel=1e-6)

    @pytest.mark.parametrize(("mu", "hbr"), [([0.0, 0.0], 0.05), ([0.5, 0.0], 0.08)])
    def test_monte_carlo_matches_quadrature_within_its_own_error_bar(self, mu, hbr, rng):
        mu = np.array(mu)
        exact = pc_analytic(mu, COV_2D, hbr)
        mc = pc_monte_carlo(mu, COV_2D, hbr, 4_000_000, rng)
        assert abs(mc.pc - exact) < 4.0 * mc.stderr

    def test_log_form_matches_the_linear_form(self):
        mu = np.array([0.4, -0.2])
        assert log10_pc_small_disk(mu, COV_2D, 1e-3) == pytest.approx(
            np.log10(pc_small_disk(mu, COV_2D, 1e-3)))

    def test_log_form_survives_where_the_linear_form_underflows(self):
        """Screened conjunctions routinely miss by tens of sigma, where Pc
        underflows to exactly zero in double precision."""
        mu = np.array([400.0, 400.0])
        linear = pc_small_disk(mu, COV_2D, 1e-3)
        logged = log10_pc_small_disk(mu, COV_2D, 1e-3)
        assert linear == 0.0, "the linear form is expected to underflow here"
        assert np.isfinite(logged) and logged < -1000, (
            "the log form must stay finite and ordered where the linear one dies")


class TestScaling:
    def test_probability_grows_as_the_square_of_the_hard_body_radius(self):
        mu = np.array([0.3, 0.1])
        assert pc_small_disk(mu, COV_2D, 2e-3) / pc_small_disk(mu, COV_2D, 1e-3) == \
            pytest.approx(4.0)

    def test_a_larger_miss_distance_lowers_the_probability(self):
        near = pc_small_disk(np.array([0.1, 0.0]), COV_2D, 1e-3)
        far = pc_small_disk(np.array([3.0, 0.0]), COV_2D, 1e-3)
        assert far < near

    def test_a_disk_far_larger_than_the_uncertainty_captures_everything(self):
        """Tolerance is the quadrature's own resolution: the midpoint rule
        carries a relative error of about (dr/sigma)^2 / 24, tuned to ~1e-6."""
        assert pc_analytic(np.zeros(2), np.diag([1e-4, 1e-4]), 10.0) == \
            pytest.approx(1.0, abs=1e-5)


class TestHardBodyRadius:
    def test_combines_the_two_size_classes(self):
        assert hard_body_radius_km("SMALL", "LARGE") == pytest.approx(0.0035)

    def test_unknown_class_falls_back_to_the_default(self):
        assert hard_body_radius_km("", "") == pytest.approx(0.002)

    def test_is_symmetric_in_its_arguments(self):
        assert hard_body_radius_km("SMALL", "LARGE") == hard_body_radius_km("LARGE", "SMALL")
