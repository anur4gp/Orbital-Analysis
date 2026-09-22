"""The Gaussian-process surrogate, checked against GP identities."""
from __future__ import annotations

import numpy as np
import pytest

from orbital.surrogate.gp import GP, _neg_log_marginal, ard_sqexp, fit_gp


def smooth(x: np.ndarray) -> np.ndarray:
    """A smooth 4-D test function standing in for log10(Pc)."""
    return (np.sin(3.0 * x[:, 0]) + 2.0 * x[:, 1] - x[:, 2] ** 2
            + 0.5 * x[:, 0] * x[:, 3])


@pytest.fixture
def training(rng):
    x = rng.random((40, 4))
    return x, smooth(x)


class TestKernel:
    def test_diagonal_is_the_amplitude(self):
        log_theta = np.array([np.log(2.0), 0.0, 0.0])
        x = np.array([[0.1, 0.2], [0.7, 0.9]])
        assert np.allclose(np.diag(ard_sqexp(x, x, log_theta)), 4.0)

    def test_symmetric_and_positive_semidefinite(self, rng):
        x = rng.random((15, 3))
        k = ard_sqexp(x, x, np.zeros(4))
        assert np.allclose(k, k.T)
        assert np.linalg.eigvalsh(k).min() > -1e-10

    def test_decays_with_distance(self):
        log_theta = np.zeros(2)
        near = ard_sqexp(np.array([[0.0]]), np.array([[0.1]]), log_theta)[0, 0]
        far = ard_sqexp(np.array([[0.0]]), np.array([[2.0]]), log_theta)[0, 0]
        assert 1.0 > near > far > 0.0

    def test_per_dimension_length_scales(self):
        log_theta = np.array([0.0, np.log(0.1), np.log(10.0)])
        origin = np.array([[0.0, 0.0]])
        assert (ard_sqexp(origin, np.array([[0.5, 0.0]]), log_theta)[0, 0]
                < ard_sqexp(origin, np.array([[0.0, 0.5]]), log_theta)[0, 0])

    def test_extreme_parameters_stay_finite(self):
        """The optimiser is bounded, but clipping must hold regardless."""
        x = np.array([[0.0, 0.0], [1.0, 1.0]])
        for theta in (np.array([50.0, -50.0, -50.0]), np.array([-50.0, 50.0, 50.0])):
            assert np.all(np.isfinite(ard_sqexp(x, x, theta)))


class TestFitAndPredict:
    def test_interpolates_its_training_points(self, training):
        x, y = training
        gp = fit_gp(x, y, n_restarts=2)
        assert np.allclose(gp.predict(x), y, atol=1e-3)

    def test_posterior_std_is_small_at_training_points_and_larger_away(self, training, rng):
        x, y = training
        gp = fit_gp(x, y, n_restarts=2)
        _, std_train = gp.predict(x, return_std=True)
        _, std_far = gp.predict(np.full((1, 4), 5.0), return_std=True)
        assert std_train.max() < 0.05
        assert std_far[0] > 10 * std_train.max()

    def test_generalises_to_held_out_points(self, training, rng):
        x, y = training
        gp = fit_gp(x, y, n_restarts=3)
        x_test = rng.random((50, 4))
        error = np.abs(gp.predict(x_test) - smooth(x_test))
        assert np.sqrt(np.mean(error**2)) < 0.1

    def test_reproduces_a_linear_function(self, rng):
        x = rng.random((30, 2))
        y = 3.0 * x[:, 0] - 2.0 * x[:, 1]
        gp = fit_gp(x, y, n_restarts=2)
        x_test = rng.random((20, 2))
        assert np.allclose(gp.predict(x_test), 3.0 * x_test[:, 0] - 2.0 * x_test[:, 1],
                           atol=0.05)

    def test_constant_target_is_handled(self):
        """y_std is zero here; the guard against dividing by it must work."""
        x = np.linspace(0, 1, 10)[:, None]
        gp = fit_gp(x, np.full(10, 7.0), n_restarts=1)
        assert np.allclose(gp.predict(np.array([[0.42]])), 7.0, atol=1e-6)

    def test_prediction_accepts_a_single_point(self, training):
        x, y = training
        gp = fit_gp(x, y, n_restarts=1)
        assert gp.predict(x[0]).shape == (1,)

    def test_standardisation_is_recorded_and_undone(self, training):
        x, y = training
        gp = fit_gp(x, y, n_restarts=1)
        assert gp.y_mean == pytest.approx(y.mean())
        assert gp.y_std == pytest.approx(y.std())

    def test_fit_is_reproducible(self, training):
        x, y = training
        a = fit_gp(x, y, n_restarts=3, seed=5)
        b = fit_gp(x, y, n_restarts=3, seed=5)
        assert np.array_equal(a.log_theta, b.log_theta)
        assert np.allclose(a.predict(x), b.predict(x))

    def test_restarts_do_not_worsen_the_likelihood(self, training):
        x, y = training
        y_s = (y - y.mean()) / y.std()
        one = fit_gp(x, y, n_restarts=1)
        many = fit_gp(x, y, n_restarts=5)
        nlml_one = _neg_log_marginal(np.append(one.log_theta, one.log_noise), x, y_s)
        nlml_many = _neg_log_marginal(np.append(many.log_theta, many.log_noise), x, y_s)
        assert nlml_many <= nlml_one + 1e-6

    def test_fitted_beats_the_initial_guess(self, training):
        x, y = training
        y_s = (y - y.mean()) / y.std()
        gp = fit_gp(x, y, n_restarts=2)
        start = np.concatenate([[0.0], np.full(x.shape[1], np.log(0.5)), [np.log(1e-3)]])
        assert (_neg_log_marginal(np.append(gp.log_theta, gp.log_noise), x, y_s)
                < _neg_log_marginal(start, x, y_s))


class TestMarginalLikelihood:
    def test_jitter_survives_duplicate_points(self):
        x = np.zeros((3, 2))
        params = np.array([0.0, 0.0, 0.0, -50.0])
        assert np.isfinite(_neg_log_marginal(params, x, np.zeros(3)))

    def test_hopeless_kernel_returns_a_large_penalty(self):
        x = np.zeros((50, 2))
        params = np.array([10.0, 0.0, 0.0, -50.0])
        assert _neg_log_marginal(params, x, np.zeros(50)) == pytest.approx(1e12)

    def test_is_finite_for_sane_inputs(self, training):
        x, y = training
        y_s = (y - y.mean()) / y.std()
        params = np.concatenate([[0.0], np.full(4, np.log(0.5)), [np.log(1e-3)]])
        assert np.isfinite(_neg_log_marginal(params, x, y_s))


def test_gp_dataclass_round_trip(training):
    x, y = training
    gp = fit_gp(x, y, n_restarts=1)
    clone = GP(x=gp.x, y_mean=gp.y_mean, y_std=gp.y_std, log_theta=gp.log_theta,
               log_noise=gp.log_noise, _lower=gp._lower, _alpha=gp._alpha)
    assert np.allclose(clone.predict(x), gp.predict(x))
