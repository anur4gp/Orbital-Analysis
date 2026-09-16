"""6-DOF dynamics validated against physics, not stored values.

- torque-free motion conserves rotational energy and the inertial angular
  momentum vector;
- an axisymmetric body follows the closed-form torque-free precession;
- the quaternion norm stays within tolerance, and projection is what keeps it
  there;
- two-body and J2 orbits conserve their integrals and J2 regresses the node at
  the analytic secular rate;
- the gravity-gradient torque matches a direct sum over point masses and
  produces the analytic small-angle pitch libration.
"""
from __future__ import annotations

import numpy as np
import pytest

from orbital.attitude import quaternion as quat
from orbital.attitude.dcm import to_dcm
from orbital.conventions import Frame
from orbital.core.constants import J2_EARTH, MU_EARTH_KM3_S2, R_EARTH_KM
from orbital.dynamics import (
    GravityGradientTorque,
    InertiaTensor,
    J2Gravity,
    MassProperties,
    RigidBody,
    RigidBodyState,
    TwoBodyGravity,
)
from orbital.dynamics import diagnostics as dg
from orbital.integrators import DOP853, RK4

TIGHT = DOP853(rtol=1e-12, atol=1e-14)
A_KM = 7000.0
N_RAD_S = np.sqrt(MU_EARTH_KM3_S2 / A_KM**3)
V_CIRC = np.sqrt(MU_EARTH_KM3_S2 / A_KM)
PERIOD_S = 2 * np.pi / N_RAD_S


def at_rest(q=quat.IDENTITY, omega=(0.0, 0.0, 0.0)) -> RigidBodyState:
    """A state whose translation is irrelevant (no forces act)."""
    return RigidBodyState([A_KM, 0, 0], [0, 0, 0], q, omega)


# --------------------------------------------------------------------------
# Mass properties and state
# --------------------------------------------------------------------------
class TestInertiaTensor:
    def test_rejects_asymmetric(self):
        with pytest.raises(ValueError, match="symmetric"):
            InertiaTensor(np.array([[1.0, 0.1, 0], [0, 1, 0], [0, 0, 1]]))

    def test_rejects_non_positive_definite(self):
        with pytest.raises(ValueError, match="positive definite"):
            InertiaTensor.diagonal(1.0, 1.0, -1.0)

    def test_rejects_triangle_inequality_violation(self):
        """Positive definite, but no mass distribution has it."""
        with pytest.raises(ValueError, match="triangle"):
            InertiaTensor.diagonal(1.0, 1.0, 3.0)

    def test_flat_plate_is_the_limiting_case(self):
        """A thin plate has I3 = I1 + I2 exactly, which must be accepted."""
        plate = InertiaTensor.cuboid(12.0, 2.0, 1.0, 0.0)
        m = np.diag(plate.matrix)
        assert m[2] == pytest.approx(m[0] + m[1])

    def test_cuboid_formula(self):
        c = InertiaTensor.cuboid(12.0, 1.0, 2.0, 3.0)
        assert np.allclose(np.diag(c.matrix), [13.0, 10.0, 5.0])

    def test_principal_axes_diagonalise(self, rng):
        rot = to_dcm(quat.normalize(rng.normal(size=4)))
        tensor = InertiaTensor(rot @ np.diag([3.0, 4.0, 5.0]) @ rot.T)
        moments, axes = tensor.principal()
        assert np.allclose(moments, [3.0, 4.0, 5.0])
        assert np.allclose(axes.T @ tensor.matrix @ axes, np.diag(moments))
        assert np.linalg.det(axes) == pytest.approx(1.0)

    def test_matrix_is_read_only(self):
        tensor = InertiaTensor.diagonal(1.0, 2.0, 2.5)
        with pytest.raises(ValueError):
            tensor.matrix[0, 0] = 5.0

    def test_inverse(self):
        tensor = InertiaTensor.diagonal(2.0, 4.0, 5.0)
        assert np.allclose(tensor.inverse @ tensor.matrix, np.eye(3))

    def test_mass_must_be_positive(self):
        with pytest.raises(ValueError):
            MassProperties(0.0, InertiaTensor.diagonal(1, 1, 1))


class TestState:
    def test_round_trip(self):
        s = RigidBodyState([1, 2, 3], [4, 5, 6], quat.from_axis_angle([1, 0, 0], 0.2), [7, 8, 9])
        back = RigidBodyState.from_vector(s.to_vector())
        assert np.array_equal(back.to_vector(), s.to_vector())

    def test_quaternion_is_normalised(self):
        s = RigidBodyState([0, 0, 1], [0, 0, 0], [2, 0, 0, 0], [0, 0, 0])
        assert np.array_equal(s.q, quat.IDENTITY)

    def test_rejects_wrong_length(self):
        with pytest.raises(ValueError):
            RigidBodyState.from_vector(np.zeros(12))

    def test_propagator_refuses_teme(self):
        """SGP4 states are TEME; they must be converted explicitly."""
        body = RigidBody(MassProperties(1.0, InertiaTensor.diagonal(1, 1, 1)))
        teme = RigidBodyState([A_KM, 0, 0], [0, 7.5, 0], quat.IDENTITY, [0, 0, 0],
                              frame=Frame.TEME)
        with pytest.raises(ValueError, match="TEME"):
            body.propagate(teme, [0.0, 1.0], TIGHT)


# --------------------------------------------------------------------------
# Torque-free rotation
# --------------------------------------------------------------------------
TRIAXIAL = InertiaTensor.diagonal(80.0, 120.0, 150.0)
TUMBLE = at_rest(quat.from_axis_angle([1, 2, 3], 0.7), [0.1, 0.02, -0.15])
T_TUMBLE = np.linspace(0.0, 3000.0, 301)


def energy_and_momentum_drift(trajectory, inertia):
    energy = dg.rotational_kinetic_energy(trajectory.omega_rad_s, inertia)
    h = dg.angular_momentum_inertial(trajectory.q, trajectory.omega_rad_s, inertia)
    d_energy = np.max(np.abs(energy / energy[0] - 1.0))
    d_h = np.max(np.linalg.norm(h - h[0], axis=1)) / np.linalg.norm(h[0])
    return d_energy, d_h


class TestTorqueFreeConservation:
    body = RigidBody(MassProperties(500.0, TRIAXIAL))

    def test_adaptive_conserves_energy_and_momentum(self):
        trajectory = self.body.propagate(TUMBLE, T_TUMBLE, TIGHT)
        d_energy, d_h = energy_and_momentum_drift(trajectory, TRIAXIAL)
        assert d_energy < 1e-11
        assert d_h < 1e-10

    def test_rk4_drift_is_at_least_fourth_order(self):
        """Halving h must cut the energy drift by at least 2**4."""
        coarse = self.body.propagate(TUMBLE, T_TUMBLE, RK4(0.5))
        fine = self.body.propagate(TUMBLE, T_TUMBLE, RK4(0.25))
        de_coarse, dh_coarse = energy_and_momentum_drift(coarse, TRIAXIAL)
        de_fine, dh_fine = energy_and_momentum_drift(fine, TRIAXIAL)
        assert de_coarse < 1e-6
        assert de_coarse / de_fine > 15.0
        assert dh_coarse / dh_fine > 15.0

    def test_the_motion_is_not_trivial(self):
        """Guard: the rates really change, so conservation is non-vacuous."""
        trajectory = self.body.propagate(TUMBLE, T_TUMBLE, TIGHT)
        assert np.ptp(trajectory.omega_rad_s, axis=0).min() > 1e-2

    def test_pure_spin_about_principal_axis_is_steady(self):
        trajectory = self.body.propagate(at_rest(omega=[0, 0, 0.2]), [0.0, 100.0], TIGHT)
        assert np.allclose(trajectory.omega_rad_s[-1], [0, 0, 0.2], atol=1e-14)
        expected = quat.from_axis_angle([0, 0, 1], 20.0)
        assert quat.rotation_angle(quat.multiply(trajectory.q[-1], quat.conjugate(expected))) < 1e-10


class TestSymmetricPrecession:
    """Closed-form torque-free motion of an axisymmetric body.

    With I1 = I2 = It, the body rates rotate about the symmetry axis at
    lambda = (I3 - It) omega3 / It, and the symmetry axis cones about the fixed
    H at phi_dot = |H| / It. The full attitude is therefore

        q(t) = R(H_hat, phi_dot t) ⊗ q0 ⊗ R(e3, -lambda t)
    """

    W0 = np.array([0.05, 0.02, 0.3])
    Q0 = quat.from_axis_angle([0.3, -1.0, 0.5], 1.1)
    T = np.linspace(0.0, 600.0, 121)

    def analytic(self, inertia: InertiaTensor):
        it, i3 = inertia.matrix[0, 0], inertia.matrix[2, 2]
        h_inertial = to_dcm(self.Q0) @ (inertia.matrix @ self.W0)
        phi_dot = np.linalg.norm(h_inertial) / it
        lam = (i3 - it) * self.W0[2] / it
        a, b = self.W0[:2]
        c, s = np.cos(lam * self.T), np.sin(lam * self.T)
        omega = np.column_stack([a * c - b * s, a * s + b * c, np.full_like(self.T, self.W0[2])])
        q = [
            quat.multiply(
                quat.multiply(quat.from_axis_angle(h_inertial, phi_dot * t), self.Q0),
                quat.from_axis_angle([0, 0, 1], -lam * t),
            )
            for t in self.T
        ]
        return np.array(q), omega, h_inertial

    @pytest.mark.parametrize("i3", [150.0, 60.0], ids=["oblate", "prolate"])
    @pytest.mark.parametrize(
        ("integrator", "tol_rad"), [(TIGHT, 1e-10), (RK4(0.1), 1e-6)], ids=["DOP853", "RK4"]
    )
    def test_attitude_matches_closed_form(self, i3, integrator, tol_rad):
        inertia = InertiaTensor.diagonal(100.0, 100.0, i3)
        q_exact, omega_exact, _ = self.analytic(inertia)
        trajectory = RigidBody(MassProperties(10.0, inertia)).propagate(
            at_rest(self.Q0, self.W0), self.T, integrator
        )
        errors = [
            quat.rotation_angle(quat.multiply(qn, quat.conjugate(qe)))
            for qn, qe in zip(trajectory.q, q_exact, strict=True)
        ]
        assert max(errors) < tol_rad
        assert np.allclose(trajectory.omega_rad_s, omega_exact, atol=1e-6)

    def test_nutation_angle_is_constant(self):
        inertia = InertiaTensor.diagonal(100.0, 100.0, 150.0)
        _, _, h_inertial = self.analytic(inertia)
        h_hat = h_inertial / np.linalg.norm(h_inertial)
        trajectory = RigidBody(MassProperties(10.0, inertia)).propagate(
            at_rest(self.Q0, self.W0), self.T, TIGHT
        )
        cos_theta = [to_dcm(q)[:, 2] @ h_hat for q in trajectory.q]
        expected = 150.0 * self.W0[2] / np.linalg.norm(h_inertial)
        assert np.allclose(cos_theta, expected, atol=1e-10)


class TestQuaternionNorm:
    body = RigidBody(MassProperties(500.0, TRIAXIAL))

    def test_rk4_drifts_without_projection(self):
        """The drift is real -- this is what the projection exists to prevent."""
        trajectory = self.body.propagate(TUMBLE, T_TUMBLE, RK4(0.5), quaternion_tol=None)
        assert trajectory.integration.nprojections == 0
        assert quat.norm_error(trajectory.q[-1]) > 1e-8

    def test_rk4_projection_holds_the_norm(self):
        trajectory = self.body.propagate(TUMBLE, T_TUMBLE, RK4(0.5), quaternion_tol=1e-12)
        assert trajectory.integration.nprojections > 0
        assert max(quat.norm_error(q) for q in trajectory.q) <= 1e-12

    def test_adaptive_stays_within_tolerance(self):
        trajectory = self.body.propagate(TUMBLE, T_TUMBLE, TIGHT, quaternion_tol=1e-10)
        assert max(quat.norm_error(q) for q in trajectory.q) <= 1e-10

    def test_projection_does_not_change_the_physics(self):
        free = self.body.propagate(TUMBLE, T_TUMBLE, RK4(0.5), quaternion_tol=None)
        pinned = self.body.propagate(TUMBLE, T_TUMBLE, RK4(0.5), quaternion_tol=1e-12)
        assert np.allclose(free.omega_rad_s, pinned.omega_rad_s, atol=1e-12)


# --------------------------------------------------------------------------
# Translational dynamics
# --------------------------------------------------------------------------
POINT = MassProperties(1.0, InertiaTensor.diagonal(1.0, 1.0, 1.0))


def orbit_state(inclination_rad: float) -> RigidBodyState:
    ci, si = np.cos(inclination_rad), np.sin(inclination_rad)
    return RigidBodyState([A_KM, 0, 0], [0, V_CIRC * ci, V_CIRC * si], quat.IDENTITY, [0, 0, 0])


class TestGravityModels:
    @pytest.mark.parametrize("model", [TwoBodyGravity(), J2Gravity()], ids=["two-body", "J2"])
    def test_acceleration_is_minus_gradient_of_potential(self, model, rng):
        for _ in range(10):
            r = rng.normal(size=3)
            r = 7000.0 * r / np.linalg.norm(r)
            h = 1e-3
            grad = np.array([
                (model.potential(r + h * e) - model.potential(r - h * e)) / (2 * h)
                for e in np.eye(3)
            ])
            accel = model.acceleration(RigidBodyState(r, [0, 0, 0], quat.IDENTITY, [0, 0, 0]),
                                       POINT)
            assert np.allclose(accel, -grad, rtol=1e-7, atol=1e-16)

    def test_j2_is_a_small_perturbation(self):
        state = orbit_state(0.5)
        ratio = (np.linalg.norm(J2Gravity().acceleration(state, POINT))
                 / np.linalg.norm(TwoBodyGravity().acceleration(state, POINT)))
        assert 1e-4 < ratio < 1e-2


class TestOrbits:
    def test_two_body_orbit_closes_after_one_period(self):
        body = RigidBody(POINT, (TwoBodyGravity(),))
        trajectory = body.propagate(orbit_state(0.9), np.linspace(0, PERIOD_S, 50), TIGHT)
        assert np.linalg.norm(trajectory.r_km[-1] - trajectory.r_km[0]) < 1e-6
        energy = dg.specific_orbital_energy(trajectory.r_km, trajectory.v_km_s, body.forces)
        # E is a difference of two large terms, so allow 100x the local rtol.
        assert np.max(np.abs(energy / energy[0] - 1)) < 1e-10
        h = dg.specific_angular_momentum(trajectory.r_km, trajectory.v_km_s)
        assert np.max(np.linalg.norm(h - h[0], axis=1)) / np.linalg.norm(h[0]) < 1e-10

    def test_j2_integrals_and_nodal_regression(self):
        """J2 conserves energy and h_z (axisymmetry) but not the full h vector,
        and regresses the node at -3/2 n J2 (R/a)^2 cos i."""
        inclination = np.radians(50.0)
        gravity = (TwoBodyGravity(), J2Gravity())
        body = RigidBody(POINT, gravity)
        t = np.linspace(0.0, 20 * PERIOD_S, 801)
        trajectory = body.propagate(orbit_state(inclination), t, DOP853(1e-11, 1e-13))

        energy = dg.specific_orbital_energy(trajectory.r_km, trajectory.v_km_s, gravity)
        h = dg.specific_angular_momentum(trajectory.r_km, trajectory.v_km_s)
        assert np.max(np.abs(energy / energy[0] - 1)) < 1e-9
        assert np.max(np.abs(h[:, 2] / h[0, 2] - 1)) < 1e-9
        assert np.max(np.abs(np.linalg.norm(h, axis=1) / np.linalg.norm(h[0]) - 1)) > 1e-4

        raan = np.unwrap(np.arctan2(h[:, 0], -h[:, 1]))
        measured = np.polyfit(t, raan, 1)[0]
        analytic = -1.5 * N_RAD_S * J2_EARTH * (R_EARTH_KM / A_KM) ** 2 * np.cos(inclination)
        # Osculating vs mean semi-major axis differ at O(J2), hence the 1% band.
        assert measured == pytest.approx(analytic, rel=1e-2)


# --------------------------------------------------------------------------
# Gravity-gradient torque
# --------------------------------------------------------------------------
class TestGravityGradient:
    def test_matches_direct_sum_over_point_masses(self, rng):
        """Independent check: sum r_i x F_i over a discrete mass distribution."""
        points_m = np.array([[4, 0, 0], [-4, 0, 0], [0, 2.5, 0], [0, -2.5, 0],
                             [0, 0, 1.5], [0, 0, -1.5], [1, 1, 1], [-1, -1, -1]], float)
        masses = np.array([10, 10, 20, 20, 30, 30, 5, 5], float)
        inertia = InertiaTensor(sum(
            m * (p @ p * np.eye(3) - np.outer(p, p)) for m, p in zip(masses, points_m, strict=True)
        ))
        props = MassProperties(masses.sum(), inertia)
        q = quat.normalize(rng.normal(size=4))
        r = np.array([5000.0, -3000.0, 3500.0])
        state = RigidBodyState(r, [0, 0, 0], q, [0, 0, 0])

        c = to_dcm(q)
        direct = np.zeros(3)
        for m, p in zip(masses, points_m, strict=True):
            r_i = r + c @ (p * 1e-3)  # m -> km
            f_body = c.T @ (-MU_EARTH_KM3_S2 * m * r_i / np.linalg.norm(r_i) ** 3)  # kg km/s^2
            direct += np.cross(p, f_body * 1e3)  # m x (kg m/s^2) = N m

        model = GravityGradientTorque().torque(state, props)
        assert np.linalg.norm(direct) > 1e-6
        assert np.allclose(model, direct, rtol=1e-5)

    def test_vanishes_for_principal_axis_along_radius(self):
        state = RigidBodyState([A_KM, 0, 0], [0, 0, 0], quat.IDENTITY, [0, 0, 0])
        props = MassProperties(1.0, InertiaTensor.diagonal(20, 50, 60))
        assert np.allclose(GravityGradientTorque().torque(state, props), 0.0, atol=1e-20)

    @staticmethod
    def pitch_history(inertia: InertiaTensor, pitch0_rad: float, orbits: float):
        """Pitch of body x from local vertical, circular equatorial orbit."""
        body = RigidBody(MassProperties(100.0, inertia), (TwoBodyGravity(),),
                         (GravityGradientTorque(),))
        initial = RigidBodyState(
            [A_KM, 0, 0], [0, V_CIRC, 0],
            quat.from_axis_angle([0, 0, 1], pitch0_rad), [0, 0, N_RAD_S],
        )
        t = np.linspace(0.0, orbits * PERIOD_S, 1001)
        trajectory = body.propagate(initial, t, DOP853(1e-11, 1e-13))
        x_body = np.array([to_dcm(q)[:, 0] for q in trajectory.q])
        radial = trajectory.r_km / np.linalg.norm(trajectory.r_km, axis=1, keepdims=True)
        pitch = np.arctan2(np.cross(radial, x_body)[:, 2], np.sum(radial * x_body, axis=1))
        return t, pitch, trajectory

    def test_small_pitch_libration_frequency(self):
        """Linearised: pitch'' + 3 n^2 (I2 - I1)/I3 pitch = 0."""
        i1, i2, i3 = 20.0, 50.0, 60.0
        pitch0 = 0.01
        t, pitch, trajectory = self.pitch_history(InertiaTensor.diagonal(i1, i2, i3), pitch0, 5)
        w_lib = N_RAD_S * np.sqrt(3 * (i2 - i1) / i3)
        # 1% of amplitude; the nonlinear frequency shift is O(pitch0^2).
        assert np.max(np.abs(pitch - pitch0 * np.cos(w_lib * t))) < 1e-2 * pitch0
        # Planar problem: no roll or yaw rate is ever excited.
        assert np.max(np.abs(trajectory.omega_rad_s[:, :2])) < 1e-12

    def test_minimum_axis_off_vertical_is_unstable(self):
        """Swap I1 and I2: the long axis now lies along-track, and the
        equilibrium becomes a saddle -- a 0.01 rad offset grows to a tumble."""
        _, pitch, _ = self.pitch_history(InertiaTensor.diagonal(50.0, 20.0, 60.0), 0.01, 5)
        assert np.max(np.abs(pitch)) > 1.0

    def test_vertical_equilibrium_is_held(self):
        _, pitch, _ = self.pitch_history(InertiaTensor.diagonal(20.0, 50.0, 60.0), 0.0, 2)
        assert np.max(np.abs(pitch)) < 1e-9
