"""EKF / UKF orbit determination."""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from orbital.attitude.dcm import rot_x, rot_z
from orbital.core.constants import MU_EARTH_KM3_S2
from orbital.core.frames import EarthRotation
from orbital.dynamics import J2Gravity, TwoBodyGravity
from orbital.estimation import EKF, UKF, Observation, OrbitModel
from orbital.estimation import consistency as cs
from orbital.estimation.measurements import GroundStation, PositionFix, RangeRangeRate
from orbital.estimation.orbit_model import white_noise_acceleration
from orbital.estimation.simulation import Scenario, monte_carlo, simulate_observations

FORCES = (TwoBodyGravity(), J2Gravity())
MODEL = OrbitModel(FORCES)
EARTH = EarthRotation(datetime(2026, 9, 16, tzinfo=UTC))
MADRID = GroundStation("Madrid", 40.4314, -4.2481, 0.83, EARTH)
A_KM = 7000.0
_C = rot_z(np.radians(250.0)) @ rot_x(np.radians(51.6))
X0 = np.concatenate([_C @ [A_KM, 0, 0], _C @ [0, np.sqrt(MU_EARTH_KM3_S2 / A_KM), 0]])


def diag_p(sigma_r_km: float, sigma_v_km_s: float) -> np.ndarray:
    return np.diag([sigma_r_km**2] * 3 + [sigma_v_km_s**2] * 3)


def numeric_jacobian(f, x, steps):
    cols = []
    for j, h in enumerate(steps):
        e = np.zeros_like(x)
        e[j] = h
        cols.append((f(x + e) - f(x - e)) / (2 * h))
    return np.column_stack(cols)


STEPS = np.array([1e-3] * 3 + [1e-6] * 3)


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
class TestMeasurements:
    def overhead_state(self, t: float) -> np.ndarray:
        r_s = MADRID.position_eci(t)
        return np.concatenate([r_s * (1 + 500 / np.linalg.norm(r_s)), [1.0, 7.0, 0.5]])

    def test_range_rate_jacobian_matches_finite_differences(self):
        m = RangeRangeRate(MADRID)
        x = X0 + np.array([100.0, -2000.0, 3000.0, 0.1, -0.2, 0.3])
        numeric = numeric_jacobian(lambda s: m.predict(50.0, s), x, STEPS)
        assert np.allclose(m.jacobian(50.0, x), numeric, rtol=1e-6, atol=1e-10)

    def test_position_fix_jacobian(self):
        m = PositionFix()
        numeric = numeric_jacobian(lambda s: m.predict(0.0, s), X0, STEPS)
        assert np.allclose(m.jacobian(0.0, X0), numeric)

    def test_zenith_pass_is_at_ninety_degrees(self):
        # Along the ellipsoid normal, not the geocentric radius.
        up = EARTH.ecef_to_eci(10.0) @ np.array([
            np.cos(np.radians(40.4314)) * np.cos(np.radians(-4.2481)),
            np.cos(np.radians(40.4314)) * np.sin(np.radians(-4.2481)),
            np.sin(np.radians(40.4314)),
        ])
        r = MADRID.position_eci(10.0) + 500.0 * up
        assert np.degrees(MADRID.elevation_rad(10.0, r)) == pytest.approx(90.0)

    def test_station_velocity_is_earth_rotation(self):
        v = MADRID.velocity_eci(0.0)
        r = MADRID.position_eci(0.0)
        assert v @ r == pytest.approx(0.0, abs=1e-9)
        # |v| = omega_E * distance from the spin axis ~ 0.35 km/s at 40 deg.
        assert np.linalg.norm(v) == pytest.approx(7.292115e-5 * np.hypot(r[0], r[1]))

    def test_range_rate_of_a_co_moving_object_is_zero(self):
        m = RangeRangeRate(MADRID)
        r = MADRID.position_eci(0.0) + np.array([100.0, 200.0, 300.0])
        x = np.concatenate([r, MADRID.velocity_eci(0.0)])
        # Pure translation with the station's velocity: rho is constant.
        assert m.predict(0.0, x)[1] == pytest.approx(0.0, abs=1e-12)

    def test_availability_follows_the_mask(self):
        m = RangeRangeRate(MADRID)
        assert not m.is_available(0.0, -self.overhead_state(0.0))
        assert m.is_available(0.0, self.overhead_state(0.0))


class TestOrbitModel:
    def test_vectorised_path_matches_general_path(self, rng):
        r = X0[:3] + rng.normal(scale=100.0, size=(5, 3))
        batch = MODEL.accelerations(0.0, r)

        class Opaque:  # hides acceleration_many, forcing the loop
            def __init__(self, inner):
                self.inner = inner

            def acceleration(self, state, mp):
                return self.inner.acceleration(state, mp)

        loop = OrbitModel(tuple(Opaque(f) for f in FORCES)).accelerations(0.0, r)
        assert np.allclose(batch, loop, rtol=1e-14)

    def test_stm_matches_finite_differences(self):
        _, phi = MODEL.propagate_with_stm(X0, 0.0, 1800.0)
        numeric = numeric_jacobian(lambda s: MODEL.propagate(s, 0.0, 1800.0), X0, STEPS)
        assert np.allclose(phi, numeric, rtol=1e-5, atol=1e-6)

    def test_stm_is_symplectic(self):
        """Conservative dynamics: Phi^T J Phi = J, hence det Phi = 1."""
        _, phi = MODEL.propagate_with_stm(X0, 0.0, 3000.0)
        j = np.block([[np.zeros((3, 3)), np.eye(3)], [-np.eye(3), np.zeros((3, 3))]])
        # Phi entries reach ~5e3, so 1e-4 absolute is ~4e-9 relative.
        assert np.allclose(phi.T @ j @ phi, j, atol=1e-4)
        assert np.linalg.det(phi) == pytest.approx(1.0, abs=1e-7)

    def test_stm_state_matches_plain_propagation(self):
        x_stm, _ = MODEL.propagate_with_stm(X0, 0.0, 1000.0)
        assert np.allclose(x_stm, MODEL.propagate(X0, 0.0, 1000.0), atol=1e-8)

    def test_propagate_many_matches_individual(self, rng):
        xs = X0 + rng.normal(scale=[10, 10, 10, 0.01, 0.01, 0.01], size=(4, 6))
        batch = MODEL.propagate_many(xs, 0.0, 900.0)
        single = np.array([MODEL.propagate(x, 0.0, 900.0) for x in xs])
        assert np.allclose(batch, single, atol=1e-7)

    def test_process_noise_is_symmetric_psd(self):
        q = white_noise_acceleration(60.0, 1e-12)
        assert np.allclose(q, q.T)
        assert np.all(np.linalg.eigvalsh(q) >= -1e-30)
        assert np.array_equal(white_noise_acceleration(-60.0, 1e-12), q)


class TestSigmaPoints:
    def test_weights_sum_to_one(self):
        ukf = UKF(MODEL)
        assert ukf.wm.sum() == pytest.approx(1.0)

    @pytest.mark.parametrize("alpha, kappa", [(1.0, 0.0), (0.5, 3.0), (1.0, -3.0 + 1e-3)])
    def test_reproduce_mean_and_covariance(self, alpha, kappa, rng):
        ukf = UKF(MODEL, alpha=alpha, kappa=kappa)
        a = rng.normal(size=(6, 6))
        p = a @ a.T + np.eye(6)
        x = rng.normal(size=6)
        mean, _, cov = ukf._moments(ukf.sigma_points(x, p))
        assert np.allclose(mean, x)
        assert np.allclose(cov, p)

    def test_rejects_degenerate_spread(self):
        with pytest.raises(ValueError):
            UKF(MODEL, alpha=1.0, kappa=-6.0)


# --------------------------------------------------------------------------
# Exactness where the problem is linear
# --------------------------------------------------------------------------
class TestLinearUpdate:
    P = diag_p(1.0, 1e-3)
    OBS = Observation(0.0, X0[:3] + np.array([0.3, -0.2, 0.1]), PositionFix(0.05))

    def test_ekf_update_is_the_information_form(self):
        x, p, _ = EKF(MODEL).update(X0, self.P, self.OBS)
        h = self.OBS.model.jacobian(0.0, X0)
        info = np.linalg.inv(self.P) + h.T @ np.linalg.inv(self.OBS.model.noise_covariance) @ h
        assert np.allclose(p, np.linalg.inv(info), rtol=1e-9)
        # Diagonal P and R: each position moves by the scalar Kalman gain.
        gain = 1.0 / (1.0 + 0.05**2)
        assert np.allclose(x[:3], X0[:3] + gain * np.array([0.3, -0.2, 0.1]))

    def test_ukf_equals_ekf_for_linear_measurement(self):
        xe, pe, ie = EKF(MODEL).update(X0, self.P, self.OBS)
        xu, pu, iu = UKF(MODEL).update(X0, self.P, self.OBS)
        assert np.allclose(xe, xu, atol=1e-10)
        assert np.allclose(pe, pu, atol=1e-12)
        assert ie.nis == pytest.approx(iu.nis)

    def test_predict_agrees_for_small_uncertainty(self):
        p = diag_p(1e-3, 1e-6)
        xe, pe = EKF(MODEL).predict(X0, p, 0.0, 1200.0)
        xu, pu = UKF(MODEL).predict(X0, p, 0.0, 1200.0)
        assert np.allclose(xe, xu, atol=1e-8)
        assert np.allclose(pe, pu, rtol=1e-4, atol=1e-14)

    def test_run_reports_prior_then_posterior(self):
        obs = [Observation(60.0, X0[:3], PositionFix())]
        history = EKF(MODEL).run(X0, self.P, 0.0, obs, t_out_s=[30.0, 60.0])
        assert list(history.t_s) == [0.0, 30.0, 60.0, 60.0]
        assert list(history.updated) == [False, False, False, True]
        assert history.sigma()[3, 0] < history.sigma()[2, 0]

    def test_run_rejects_observations_before_start(self):
        with pytest.raises(ValueError):
            EKF(MODEL).run(X0, self.P, 10.0, [Observation(5.0, X0[:3], PositionFix())])


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
class TestConsistencyStatistics:
    def test_nees_of_correct_covariance_averages_to_dimension(self, rng):
        a = rng.normal(size=(6, 6))
        p = a @ a.T + np.eye(6)
        e = rng.multivariate_normal(np.zeros(6), p, size=20000)
        assert cs.nees(e, np.broadcast_to(p, (20000, 6, 6))).mean() == pytest.approx(6.0, rel=0.03)

    def test_bounds_bracket_the_mean_and_tighten(self):
        lo, hi = cs.average_bounds(6, 50)
        assert lo < 6.0 < hi
        lo2, hi2 = cs.average_bounds(6, 500)
        assert lo < lo2 < hi2 < hi

    def test_rms_over_runs(self):
        e = np.zeros((2, 1, 3))
        e[0, 0] = [3.0, 4.0, 0.0]
        assert cs.rms_over_runs(e)[0] == pytest.approx(np.sqrt(12.5))

    def test_nonlinearity_index_is_zero_for_linear_model(self):
        index = cs.measurement_nonlinearity(PositionFix(), 0.0, X0, diag_p(10.0, 0.01))
        assert np.allclose(index, 0.0, atol=1e-6)

    def test_nonlinearity_index_scales_with_covariance(self):
        """The second-order spread is linear in P, so 100x P gives 100x index."""
        m = RangeRangeRate(MADRID)
        x = X0 + np.array([100.0, -2000.0, 3000.0, 0.1, -0.2, 0.3])
        small = cs.measurement_nonlinearity(m, 0.0, x, diag_p(0.1, 1e-4))
        large = cs.measurement_nonlinearity(m, 0.0, x, diag_p(1.0, 1e-3))
        assert np.all(small > 0)
        assert np.allclose(large / small, 100.0, rtol=1e-3)

    def test_simulated_noise_matches_r(self):
        """True-state residuals, normalised by R, must be chi-square."""
        t = np.arange(0.0, 60.0, 1.0)
        truth = np.tile(X0, (t.size, 1))
        m = PositionFix(0.02)
        obs = simulate_observations(t, truth, [m], np.random.default_rng(3))
        r_inv = np.linalg.inv(m.noise_covariance)
        stats = [(o.z - X0[:3]) @ r_inv @ (o.z - X0[:3]) for o in obs]
        assert np.mean(stats) == pytest.approx(3.0, rel=0.25)


def ground_scenario(sigma_r_km: float, sigma_v_km_s: float, t_end_s: float) -> Scenario:
    stations = [
        MADRID,
        GroundStation("Goldstone", 35.4267, -116.8900, 1.0, EARTH),
        GroundStation("Canberra", -35.4014, 148.9817, 0.69, EARTH),
    ]
    return Scenario(FORCES, X0, diag_p(sigma_r_km, sigma_v_km_s),
                    np.arange(0.0, t_end_s, 60.0), [RangeRangeRate(s) for s in stations])


N_RUNS = 12


class TestMonteCarloInputs:
    @pytest.mark.parametrize("n_runs", [0, -1, 1.5, True])
    def test_rejects_invalid_run_count_before_propagation(self, n_runs, monkeypatch):
        def unexpected_truth(self):
            pytest.fail("invalid input reached truth propagation")

        monkeypatch.setattr(Scenario, "truth", unexpected_truth)
        with pytest.raises(ValueError, match="positive integer"):
            monte_carlo(ground_scenario(0.01, 1e-5, 120.0), [EKF(MODEL)], n_runs, seed=1)

    @pytest.mark.parametrize("filters, message", [
        ([], "at least one filter"),
        ([EKF(MODEL), EKF(MODEL)], "filter names must be unique"),
    ])
    def test_rejects_ambiguous_filter_selection(self, filters, message, monkeypatch):
        def unexpected_truth(self):
            pytest.fail("invalid input reached truth propagation")

        monkeypatch.setattr(Scenario, "truth", unexpected_truth)
        with pytest.raises(ValueError, match=message):
            monte_carlo(ground_scenario(0.01, 1e-5, 120.0), filters, 1, seed=1)


@pytest.mark.slow
class TestFilterConsistency:
    """First pass over Madrid begins at t = 1500 s in this geometry."""

    def test_both_filters_consistent_with_precise_prior(self):
        """Only the final time is tested; NEES at successive times is correlated."""
        n_runs = 40
        results = monte_carlo(ground_scenario(0.01, 1e-5, 2400.0), [EKF(MODEL), UKF(MODEL)],
                              n_runs, seed=11)
        lo, hi = cs.average_bounds(6, n_runs, confidence=0.99)
        for name, r in results.items():
            assert r.updated.sum() > 5
            final = cs.nees(r.errors[:, -1], r.P[:, -1]).mean()
            assert lo < final < hi, name

    def test_ekf_overconfident_ukf_holds_with_km_prior(self):
        results = monte_carlo(ground_scenario(1.0, 1e-3, 5400.0), [EKF(MODEL), UKF(MODEL)],
                              N_RUNS, seed=11)
        lo, hi = cs.average_bounds(6, N_RUNS)
        ekf = cs.nees(results["EKF"].errors, results["EKF"].P).mean(axis=0)
        ukf = cs.nees(results["UKF"].errors, results["UKF"].P).mean(axis=0)
        first = int(np.argmax(results["EKF"].updated))
        # Before any measurement both are just propagating the same prior.
        assert ekf[first - 1] == pytest.approx(ukf[first - 1], rel=1e-3)
        # Measured EKF NEES ~62 at 90 min (7.6x the bound); UKF within a small factor.
        assert ekf[-1] > 5 * hi
        assert ukf[-1] < 2 * hi
