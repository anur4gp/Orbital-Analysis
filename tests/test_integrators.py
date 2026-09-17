"""Integrator interface: order of accuracy, output timing, projection.

Checked on the harmonic oscillator, whose exact solution is known, so the
tests measure error rather than compare against stored values.
"""
from __future__ import annotations

import numpy as np
import pytest

from orbital.integrators import DOP853, RK4, Constraint


def oscillator(t: float, y: np.ndarray) -> np.ndarray:
    return np.array([y[1], -y[0]])


def exact(t: np.ndarray) -> np.ndarray:
    return np.column_stack([np.cos(t), -np.sin(t)])


T = np.linspace(0.0, 10.0, 11)
Y0 = np.array([1.0, 0.0])


@pytest.mark.parametrize("integrator", [RK4(0.01), DOP853()])
def test_returns_exactly_the_requested_times(integrator):
    t = np.array([0.0, 0.37, 1.0, 2.5])
    result = integrator.integrate(oscillator, Y0, t)
    assert np.array_equal(result.t, t)
    assert result.y.shape == (4, 2)
    assert np.array_equal(result.y[0], Y0)


def test_rk4_is_fourth_order():
    """Halving h should cut the global error by 2**4 = 16."""
    errors = []
    for h in (0.2, 0.1, 0.05):
        y = RK4(h).integrate(oscillator, Y0, T).y
        errors.append(np.max(np.abs(y - exact(T))))
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all((orders > 3.8) & (orders < 4.2)), orders


def test_rk4_counts_four_evaluations_per_step():
    assert RK4(0.5).integrate(oscillator, Y0, [0.0, 10.0]).nfev == 80


def test_rk4_splits_steps_to_land_on_outputs():
    """H = 0.3 does not divide 1.0; the interval is split into 4 steps of 0.25."""
    assert RK4(0.3).integrate(oscillator, Y0, [0.0, 1.0]).nfev == 16


@pytest.mark.parametrize("rtol", [1e-6, 1e-9, 1e-12])
def test_dop853_error_tracks_tolerance(rtol):
    y = DOP853(rtol=rtol, atol=rtol * 1e-2).integrate(oscillator, Y0, T).y
    # Global error is a modest multiple of the local tolerance over 1.6 periods.
    assert np.max(np.abs(y - exact(T))) < 100 * rtol


def test_integrators_agree():
    a = RK4(0.01).integrate(oscillator, Y0, T).y
    b = DOP853(rtol=1e-12, atol=1e-14).integrate(oscillator, Y0, T).y
    assert np.allclose(a, b, atol=1e-9)


def test_adaptive_is_cheaper_at_equal_accuracy():
    """The reason to have both: DOP853 reaches 1e-10 in far fewer evaluations."""
    rk4 = RK4(0.01).integrate(oscillator, Y0, T)
    dop = DOP853(rtol=1e-11, atol=1e-13).integrate(oscillator, Y0, T)
    assert np.max(np.abs(rk4.y - exact(T))) > np.max(np.abs(dop.y - exact(T)))
    assert dop.nfev < rk4.nfev / 5


def test_rejects_bad_times():
    with pytest.raises(ValueError):
        RK4(0.1).integrate(oscillator, Y0, [0.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        DOP853().integrate(oscillator, Y0, [0.0])


def test_rejects_nonpositive_step():
    with pytest.raises(ValueError):
        RK4(0.0)


# The oscillator conserves y0^2 + y1^2 = 1, which makes a convenient constraint.
def circle_violation(y: np.ndarray) -> float:
    return abs(float(np.hypot(*y)) - 1.0)


def circle_project(y: np.ndarray) -> np.ndarray:
    return y / np.hypot(*y)


class TestConstraintProjection:
    def test_rk4_projects_every_step_at_zero_tolerance(self):
        c = Constraint(circle_violation, circle_project, 0.0)
        result = RK4(0.5).integrate(oscillator, Y0, [0.0, 10.0], c)
        assert result.nprojections == 20
        assert circle_violation(result.y[-1]) < 1e-15

    def test_rk4_projection_removes_drift(self):
        """RK4 slowly dissipates the invariant; projection pins it."""
        t = [0.0, 1000.0]
        free = RK4(0.5).integrate(oscillator, Y0, t)
        pinned = RK4(0.5).integrate(
            oscillator, Y0, t, Constraint(circle_violation, circle_project, 1e-12)
        )
        assert circle_violation(free.y[-1]) > 1e-3
        assert circle_violation(pinned.y[-1]) < 1e-12
        assert pinned.max_violation > 0.0

    def test_dop853_restarts_at_the_threshold(self):
        c = Constraint(circle_violation, circle_project, 1e-8)
        result = DOP853(rtol=1e-6, atol=1e-8).integrate(oscillator, Y0, np.linspace(0, 200, 5), c)
        assert result.nprojections > 0
        assert np.array_equal(result.t, np.linspace(0, 200, 5))
        # Outputs come from the dense interpolant, good to about rtol.
        assert all(circle_violation(y) < 1e-6 for y in result.y)

    def test_no_constraint_reports_zero(self):
        result = DOP853().integrate(oscillator, Y0, T)
        assert result.nprojections == 0
        assert result.max_violation == 0.0

    def test_initial_violation_is_projected(self):
        c = Constraint(circle_violation, circle_project, 1e-12)
        result = RK4(0.1).integrate(oscillator, np.array([2.0, 0.0]), [0.0, 0.1], c)
        assert np.allclose(result.y[0], [1.0, 0.0])
        assert result.max_violation == pytest.approx(1.0)
