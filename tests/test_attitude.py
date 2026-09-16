"""Quaternion, DCM and Euler-angle conversions.

Assertions are algebraic identities -- round trips, composition agreeing with
matrix products, orthogonality -- so they hold for any correct implementation.
"""
from __future__ import annotations

import numpy as np
import pytest

from orbital.attitude import dcm, euler
from orbital.attitude import quaternion as quat


def random_quaternions(rng: np.random.Generator, n: int = 200) -> np.ndarray:
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


class TestQuaternionAlgebra:
    def test_identity_is_neutral(self, rng):
        for q in random_quaternions(rng, 20):
            assert np.allclose(quat.multiply(quat.IDENTITY, q), q)
            assert np.allclose(quat.multiply(q, quat.IDENTITY), q)

    def test_conjugate_is_inverse(self, rng):
        for q in random_quaternions(rng, 20):
            assert np.allclose(quat.multiply(q, quat.conjugate(q)), quat.IDENTITY, atol=1e-14)

    def test_product_is_associative(self, rng):
        a, b, c = random_quaternions(rng, 3)
        left = quat.multiply(quat.multiply(a, b), c)
        right = quat.multiply(a, quat.multiply(b, c))
        assert np.allclose(left, right, atol=1e-14)

    def test_product_matches_dcm_product(self, rng):
        """p ⊗ q applies q first, so its DCM is C(p) C(q)."""
        qs = random_quaternions(rng, 40)
        for p, q in zip(qs[::2], qs[1::2], strict=True):
            assert np.allclose(dcm.to_dcm(quat.multiply(p, q)), dcm.to_dcm(p) @ dcm.to_dcm(q))

    def test_rotate_matches_dcm(self, rng):
        v = rng.normal(size=3)
        for q in random_quaternions(rng, 20):
            assert np.allclose(quat.rotate(q, v), dcm.to_dcm(q) @ v)

    def test_quarter_turn_about_z(self):
        q = quat.from_axis_angle([0, 0, 1], np.pi / 2)
        assert np.allclose(quat.rotate(q, [1, 0, 0]), [0, 1, 0])

    def test_rotation_angle_is_accurate_near_zero(self):
        """acos(w) would return 0 here; atan2 recovers the true 1e-9 rad."""
        q = quat.from_axis_angle([1, 2, 3], 1e-9)
        assert quat.rotation_angle(q) == pytest.approx(1e-9, rel=1e-6)

    def test_rotation_angle_ignores_sign(self):
        q = quat.from_axis_angle([0, 1, 0], 0.3)
        assert quat.rotation_angle(-q) == pytest.approx(0.3)

    def test_normalize_rejects_zero(self):
        with pytest.raises(ValueError):
            quat.normalize(np.zeros(4))

    def test_shape_is_checked(self):
        with pytest.raises(ValueError):
            quat.as_quaternion([1.0, 0.0, 0.0])


class TestKinematics:
    def test_matches_finite_difference_of_constant_rotation(self):
        """Constant omega gives q(t) = q0 ⊗ exp(omega t / 2) exactly."""
        omega = np.array([0.02, -0.01, 0.03])
        q0 = quat.from_axis_angle([1, 1, 0], 0.4)
        spin = np.linalg.norm(omega)

        def q_at(t: float) -> np.ndarray:
            return quat.multiply(q0, quat.from_axis_angle(omega, spin * t))

        h = 1e-4
        numeric = (q_at(h) - q_at(-h)) / (2 * h)
        assert np.allclose(quat.kinematics(q0, omega), numeric, atol=1e-10)

    def test_preserves_norm_to_first_order(self, rng):
        """q . dq/dt = 0, so the exact flow keeps |q| constant."""
        for q in random_quaternions(rng, 20):
            assert np.dot(q, quat.kinematics(q, rng.normal(size=3))) == pytest.approx(0, abs=1e-15)


class TestDCM:
    def test_round_trip(self, rng):
        for q in random_quaternions(rng):
            back = dcm.from_dcm(dcm.to_dcm(q))
            assert np.allclose(back, quat.canonical(q), atol=1e-12)

    @pytest.mark.parametrize("axis", [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    def test_round_trip_at_half_turn(self, axis):
        """Where the naive trace formula divides by ~zero."""
        q = quat.from_axis_angle(axis, np.pi)
        back = dcm.from_dcm(dcm.to_dcm(q))
        assert quat.rotation_angle(quat.multiply(back, quat.conjugate(q))) < 1e-12

    def test_output_is_proper_rotation(self, rng):
        for q in random_quaternions(rng, 50):
            assert dcm.is_rotation_matrix(dcm.to_dcm(q))

    def test_columns_are_body_axes_in_inertial(self):
        q = quat.from_axis_angle([0, 0, 1], np.pi / 2)
        c = dcm.to_dcm(q)
        assert np.allclose(c[:, 0], [0, 1, 0])  # body x now points along inertial y

    def test_elementary_rotations_agree(self):
        for axis, fn in ((0, dcm.rot_x), (1, dcm.rot_y), (2, dcm.rot_z)):
            assert np.allclose(fn(0.7), dcm.to_dcm(quat.from_axis_angle(np.eye(3)[axis], 0.7)))

    def test_rejects_reflection(self):
        with pytest.raises(ValueError):
            dcm.from_dcm(np.diag([1.0, 1.0, -1.0]))


class TestEuler321:
    def test_round_trip_angles(self, rng):
        for _ in range(200):
            angles = rng.uniform([-np.pi, -np.pi / 2 + 0.01, -np.pi], [np.pi, np.pi / 2 - 0.01, np.pi])
            assert np.allclose(euler.to_euler_321(euler.from_euler_321(*angles)), angles, atol=1e-9)

    def test_composition_order(self):
        q = euler.from_euler_321(0.3, -0.2, 0.9)
        expected = dcm.rot_z(0.3) @ dcm.rot_y(-0.2) @ dcm.rot_x(0.9)
        assert np.allclose(dcm.to_dcm(q), expected)

    @pytest.mark.parametrize("pitch", [np.pi / 2, -np.pi / 2])
    def test_gimbal_lock_preserves_rotation(self, pitch):
        q = euler.from_euler_321(0.4, pitch, 0.25)
        angles = euler.to_euler_321(q)
        assert angles[2] == 0.0
        back = euler.from_euler_321(*angles)
        assert quat.rotation_angle(quat.multiply(back, quat.conjugate(q))) < 1e-7

    def test_dcm_wrappers_round_trip(self):
        c = euler.dcm_from_euler_321(1.0, 0.5, -2.0)
        assert np.allclose(euler.euler_321_from_dcm(c), [1.0, 0.5, -2.0])


class TestEuler313:
    def test_round_trip_angles(self, rng):
        for _ in range(200):
            angles = rng.uniform([-np.pi, 0.01, -np.pi], [np.pi, np.pi - 0.01, np.pi])
            assert np.allclose(euler.to_euler_313(euler.from_euler_313(*angles)), angles, atol=1e-9)

    def test_composition_order(self):
        q = euler.from_euler_313(0.3, 1.1, -0.6)
        expected = dcm.rot_z(0.3) @ dcm.rot_x(1.1) @ dcm.rot_z(-0.6)
        assert np.allclose(dcm.to_dcm(q), expected)

    @pytest.mark.parametrize("theta", [0.0, np.pi])
    def test_singularity_preserves_rotation(self, theta):
        q = euler.from_euler_313(0.8, theta, -0.3)
        angles = euler.to_euler_313(q)
        assert angles[2] == 0.0
        back = euler.from_euler_313(*angles)
        assert quat.rotation_angle(quat.multiply(back, quat.conjugate(q))) < 1e-7
