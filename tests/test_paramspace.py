"""Encounter-parameter spaces and the exact 6 -> 4 reduction.

The reduction rests on two invariances of the collision integral. They are
verified numerically over the whole box rather than argued from algebra,
because a silently wrong reduction would invalidate every surrogate result
downstream. Residuals here are quadrature noise, not model error.
"""
from __future__ import annotations

import numpy as np
import pytest

from orbital.conjunction.probability import pc_analytic
from orbital.surrogate.paramspace import (
    BOUNDS,
    BOUNDS_4D,
    DIM,
    DIM_4D,
    from_unit_cube,
    from_unit_cube_4d,
    reduce_to_4d,
    to_unit_cube,
    unpack,
    unpack_4d,
)

#: Residual tolerance for claims that are exact identities.
EXACT = 1e-6


def rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


@pytest.fixture
def box_points(rng):
    """A sample spanning the full 6-D parameter box."""
    return from_unit_cube(rng.random((200, DIM)))


class TestBoxMapping:
    def test_unit_cube_maps_inside_the_box(self, rng):
        x = from_unit_cube(rng.random((200, DIM)))
        assert np.all(x >= BOUNDS[:, 0] - 1e-12)
        assert np.all(x <= BOUNDS[:, 1] + 1e-12)

    def test_round_trip_is_exact(self, rng):
        u = rng.random((200, DIM))
        assert np.allclose(to_unit_cube(from_unit_cube(u)), u, atol=1e-12)

    def test_corners_map_to_the_bounds(self):
        assert np.allclose(from_unit_cube(np.zeros((1, DIM)))[0], BOUNDS[:, 0])
        assert np.allclose(from_unit_cube(np.ones((1, DIM)))[0], BOUNDS[:, 1])

    def test_wrong_width_is_rejected(self):
        with pytest.raises(ValueError, match="columns"):
            from_unit_cube(np.zeros((1, DIM + 1)))


class TestEveryPointIsPhysical:
    def test_covariance_is_symmetric_positive_definite(self, box_points):
        """A box in the raw entries (c11, c12, c22) would not guarantee this;
        parameterising by eigenvalues and an angle does."""
        for row in box_points:
            _, cov, hbr = unpack(row)
            assert np.allclose(cov, cov.T)
            assert np.all(np.linalg.eigvalsh(cov) > 0)
            assert hbr > 0

    def test_eigenvalues_recover_the_input_sigmas(self):
        row = from_unit_cube(np.array([[0.3, 0.7, 0.25, 0.8, 0.4, 0.5]]))[0]
        _, cov, _ = unpack(row)
        assert np.allclose(np.sort(np.sqrt(np.linalg.eigvalsh(cov))),
                           np.sort([10 ** row[2], 10 ** row[3]]))


class TestInvariances:
    """If either invariance fails, `reduce_to_4d` discards real information."""

    def test_probability_is_invariant_under_rotation(self, box_points, rng):
        worst = 0.0
        for row in box_points[:100]:
            mu, cov, hbr = unpack(row)
            base = pc_analytic(mu, cov, hbr)
            if base <= 0:
                continue
            q = rotation(rng.uniform(0, 2 * np.pi))
            rotated = pc_analytic(q @ mu, q @ cov @ q.T, hbr)
            worst = max(worst, abs(rotated - base) / base)
        assert worst < EXACT, f"max relative error {worst:.2e}"

    def test_probability_is_invariant_under_uniform_scaling(self, box_points, rng):
        worst = 0.0
        for row in box_points[:100]:
            mu, cov, hbr = unpack(row)
            base = pc_analytic(mu, cov, hbr)
            if base <= 0:
                continue
            s = rng.uniform(0.2, 5.0)
            scaled = pc_analytic(s * mu, s * s * cov, s * hbr)
            worst = max(worst, abs(scaled - base) / base)
        assert worst < EXACT, f"max relative error {worst:.2e}"

    def test_probability_is_even_in_each_miss_component(self, box_points):
        """Holds in the covariance eigenbasis, and is what lets the reduced
        box restrict to non-negative miss components."""
        worst = 0.0
        for row in box_points[:100]:
            mu, cov, hbr = unpack(row)
            vals, vecs = np.linalg.eigh(cov)
            mu_e, cov_e = vecs.T @ mu, np.diag(vals)
            base = pc_analytic(mu_e, cov_e, hbr)
            if base <= 0:
                continue
            for flip in ([-1, 1], [1, -1], [-1, -1]):
                alt = pc_analytic(mu_e * np.array(flip), cov_e, hbr)
                worst = max(worst, abs(alt - base) / base)
        assert worst < EXACT, f"max relative error {worst:.2e}"


class TestReduction:
    def test_reduced_form_reproduces_the_probability(self, box_points):
        worst = 0.0
        for row in box_points[:100]:
            mu, cov, hbr = unpack(row)
            base = pc_analytic(mu, cov, hbr)
            if base <= 0:
                continue
            m1, m2, s1, s2 = reduce_to_4d(mu, cov, hbr)
            reduced = pc_analytic(np.array([m1, m2]), np.diag([s1 ** 2, s2 ** 2]), 1.0)
            worst = max(worst, abs(reduced - base) / base)
        assert worst < EXACT, f"max relative error {worst:.2e}"

    def test_encounters_differing_only_by_rotation_and_scale_reduce_alike(self):
        mu, cov, hbr = unpack(from_unit_cube(np.array([[0.2, 0.6, 0.3, 0.7, 0.15, 0.4]]))[0])
        q, s = rotation(0.9), 2.7
        assert np.allclose(reduce_to_4d(mu, cov, hbr),
                           reduce_to_4d(s * (q @ mu), s * s * (q @ cov @ q.T), s * hbr),
                           atol=1e-9)

    def test_round_trip_through_the_reduced_space_preserves_probability(self, box_points):
        worst = 0.0
        for row in box_points[:100]:
            mu, cov, hbr = unpack(row)
            base = pc_analytic(mu, cov, hbr)
            if base <= 0:
                continue
            m1, m2, s1, s2 = reduce_to_4d(mu, cov, hbr)
            x = np.array([np.log10(s1), np.log10(s2 / s1), abs(m1) / s1, abs(m2) / s2])
            worst = max(worst, abs(pc_analytic(*unpack_4d(x)) - base) / base)
        assert worst < EXACT, f"max relative error {worst:.2e}"


class TestReducedSpace:
    def test_maps_inside_its_box(self, rng):
        x = from_unit_cube_4d(rng.random((200, DIM_4D)))
        assert np.all(x >= BOUNDS_4D[:, 0] - 1e-12)
        assert np.all(x <= BOUNDS_4D[:, 1] + 1e-12)

    def test_lengths_are_in_hard_body_radii(self, rng):
        for row in from_unit_cube_4d(rng.random((50, DIM_4D))):
            _, _, hbr = unpack_4d(row)
            assert hbr == 1.0

    def test_anisotropy_ordering_holds_by_construction(self, rng):
        """Parameterising by the ratio sigma_2/sigma_1 <= 1 removes the need
        for a rejection step."""
        for row in from_unit_cube_4d(rng.random((200, DIM_4D))):
            _, cov, _ = unpack_4d(row)
            assert cov[1, 1] <= cov[0, 0] + 1e-12
            assert np.all(np.linalg.eigvalsh(cov) > 0)
