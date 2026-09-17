"""RTN frame construction and covariance rotation.

Rotation is an orthogonal similarity transform, so trace, determinant and
eigenvalues are invariant. Those invariants are the assertions here -- they
hold for any correct implementation, so they do not need updating when
constants change.
"""
from __future__ import annotations

import numpy as np
import pytest

from orbital.conjunction.covariance import (
    CALIBRATED_SIGMA_R_KM,
    DEFAULT_K_N,
    DEFAULT_K_T,
    RTNCovariance,
    combined_covariance,
    rtn_basis,
    sample_relative_offsets,
)


class TestRTNBasis:
    def test_is_orthonormal(self, circular_equatorial):
        a = rtn_basis(*circular_equatorial)
        assert np.allclose(a.T @ a, np.eye(3), atol=1e-12)

    def test_is_right_handed(self, circular_equatorial):
        assert np.linalg.det(rtn_basis(*circular_equatorial)) == pytest.approx(1.0)

    def test_radial_axis_points_along_position(self, circular_equatorial):
        r, v = circular_equatorial
        assert np.allclose(rtn_basis(r, v)[:, 0], r / np.linalg.norm(r))

    def test_in_track_axis_follows_velocity_when_circular(self, circular_equatorial):
        r, v = circular_equatorial
        assert np.allclose(rtn_basis(r, v)[:, 1], v / np.linalg.norm(v))

    def test_normal_axis_is_the_orbit_normal(self, circular_equatorial):
        r, v = circular_equatorial
        h = np.cross(r, v)
        assert np.allclose(rtn_basis(r, v)[:, 2], h / np.linalg.norm(h))


class TestAnisotropy:
    def test_ratios_scale_the_radial_term(self):
        c = RTNCovariance(0.2, k_t=10.0, k_n=1.5)
        assert c.sigma_t_km == pytest.approx(2.0)
        assert c.sigma_n_km == pytest.approx(0.3)

    def test_in_track_error_dominates(self):
        """A small period error integrates into a growing along-track lag."""
        c = RTNCovariance(CALIBRATED_SIGMA_R_KM, DEFAULT_K_T, DEFAULT_K_N)
        assert c.sigma_t_km > c.sigma_n_km > c.sigma_r_km

    def test_rtn_matrix_is_diagonal(self):
        m = RTNCovariance(0.2).matrix_rtn()
        assert np.allclose(m, np.diag(np.diag(m)))


class TestRotationInvariants:
    @pytest.fixture
    def cov(self):
        return RTNCovariance(0.2, k_t=10.0, k_n=1.5)

    def test_trace_preserved(self, cov, inclined_orbit):
        rtn, eci = cov.matrix_rtn(), cov.matrix_eci(*inclined_orbit)
        assert np.trace(eci) == pytest.approx(np.trace(rtn))

    def test_determinant_preserved(self, cov, inclined_orbit):
        rtn, eci = cov.matrix_rtn(), cov.matrix_eci(*inclined_orbit)
        assert np.linalg.det(eci) == pytest.approx(np.linalg.det(rtn))

    def test_eigenvalues_preserved(self, cov, inclined_orbit):
        rtn, eci = cov.matrix_rtn(), cov.matrix_eci(*inclined_orbit)
        assert np.allclose(np.sort(np.linalg.eigvalsh(rtn)),
                           np.sort(np.linalg.eigvalsh(eci)))

    def test_result_is_symmetric_positive_definite(self, cov, inclined_orbit):
        eci = cov.matrix_eci(*inclined_orbit)
        assert np.allclose(eci, eci.T)
        assert np.all(np.linalg.eigvalsh(eci) > 0)

    def test_aligned_orbit_leaves_the_matrix_unchanged(self, cov, circular_equatorial):
        """When RTN coincides with the inertial axes the rotation is identity."""
        assert np.allclose(cov.matrix_eci(*circular_equatorial), cov.matrix_rtn())

    def test_tilted_orbit_produces_off_diagonal_terms(self, cov, inclined_orbit):
        eci = cov.matrix_eci(*inclined_orbit)
        assert not np.allclose(eci, np.diag(np.diag(eci)))


class TestCombining:
    @pytest.fixture
    def cov(self):
        return RTNCovariance(0.2)

    def test_is_the_sum_of_the_rotated_parts(self, cov, circular_equatorial, inclined_orbit):
        r1, v1 = circular_equatorial
        r2, v2 = inclined_orbit
        expected = cov.matrix_eci(r1, v1) + cov.matrix_eci(r2, v2)
        assert np.allclose(combined_covariance(r1, v1, r2, v2, cov, cov), expected)

    def test_trace_adds(self, cov, circular_equatorial, inclined_orbit):
        c = combined_covariance(*circular_equatorial, *inclined_orbit, cov, cov)
        assert np.trace(c) == pytest.approx(2 * np.trace(cov.matrix_rtn()))

    def test_order_of_arguments_does_not_matter(self, cov, circular_equatorial, inclined_orbit):
        r1, v1 = circular_equatorial
        r2, v2 = inclined_orbit
        assert np.allclose(combined_covariance(r1, v1, r2, v2, cov, cov),
                           combined_covariance(r2, v2, r1, v1, cov, cov))


class TestSampling:
    def test_draws_reproduce_the_covariance_they_came_from(self, rng, circular_equatorial,
                                                           inclined_orbit):
        cov = RTNCovariance(0.2)
        c = combined_covariance(*circular_equatorial, *inclined_orbit, cov, cov)
        draws = sample_relative_offsets(c, 400_000, rng)
        assert np.allclose(draws.mean(axis=0), 0.0, atol=0.02)
        assert np.allclose(np.cov(draws, rowvar=False), c, rtol=0.03, atol=1e-3)

    def test_singular_covariance_does_not_raise(self, rng):
        """A degenerate covariance has no Cholesky factor; the eigen fallback
        must carry it.
        """
        singular = np.diag([1.0, 1.0, 0.0])
        draws = sample_relative_offsets(singular, 64, rng)
        assert draws.shape == (64, 3)
        assert np.allclose(draws[:, 2], 0.0)
