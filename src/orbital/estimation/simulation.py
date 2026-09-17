"""Truth trajectories, synthetic measurements, and Monte Carlo runs.

Truth comes from the Phase 1 6-DOF propagator (translation block; the
attitude is carried but does not affect the gravity models). Measurements
are the model predictions at the true state plus Gaussian noise drawn from
the model's own R, optionally scaled -- a scale above 1 makes the filter's
assumed noise optimistic, which is a controlled way to break consistency.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
from numpy.typing import ArrayLike

from orbital.attitude.dcm import rot_x, rot_z
from orbital.attitude.quaternion import IDENTITY, FloatArray
from orbital.core.constants import MU_EARTH_KM3_S2
from orbital.core.frames import EarthRotation
from orbital.dynamics.forces import ForceModel
from orbital.dynamics.inertia import InertiaTensor, MassProperties
from orbital.dynamics.rigid_body import RigidBody
from orbital.dynamics.state import RigidBodyState
from orbital.estimation.base import FilterHistory, Observation, SequentialFilter
from orbital.estimation.measurements.base import MeasurementModel
from orbital.estimation.measurements.ground_station import GroundStation
from orbital.estimation.measurements.range_rate import RangeRangeRate
from orbital.integrators.adaptive import DOP853

TRUTH_INTEGRATOR = DOP853(rtol=1e-12, atol=1e-13)

#: Reference epoch for the tracking scenarios.
DEFAULT_EPOCH = datetime(2026, 9, 16, tzinfo=UTC)

#: Three widely separated sites: (name, latitude deg, east longitude deg,
#: altitude km). Coordinates are approximate DSN complex locations, used as
#: a plausible geometry rather than as survey data.
DEFAULT_SITES = (
    ("Goldstone", 35.4267, -116.8900, 1.00),
    ("Canberra", -35.4014, 148.9817, 0.69),
    ("Madrid", 40.4314, -4.2481, 0.83),
)
_NOMINAL_BODY = MassProperties(100.0, InertiaTensor.diagonal(10.0, 12.0, 15.0))


def truth_trajectory(
    forces: Sequence[ForceModel], x0: ArrayLike, t_s: ArrayLike
) -> FloatArray:
    """True 6-element states at ``t_s`` from the 6-DOF propagator, shape ``(K, 6)``."""
    x0 = np.asarray(x0, dtype=float)
    body = RigidBody(_NOMINAL_BODY, tuple(forces))
    initial = RigidBodyState(x0[:3], x0[3:], IDENTITY, np.zeros(3), float(np.asarray(t_s)[0]))
    trajectory = body.propagate(initial, t_s, TRUTH_INTEGRATOR)
    return np.hstack([trajectory.r_km, trajectory.v_km_s])


def simulate_observations(
    t_s: FloatArray,
    truth: FloatArray,
    models: Sequence[MeasurementModel],
    rng: np.random.Generator,
    noise_scale: float = 1.0,
) -> list[Observation]:
    """Noisy measurements from every available model at every time.

    Parameters
    ----------
    t_s, truth
        Times (s) and true states (shape ``(K, 6)``).
    models
        Sensors; each is sampled only where ``is_available`` is true.
    rng
        Random generator (seed it for reproducibility).
    noise_scale
        Multiplier on the true noise standard deviation relative to the R the
        filter assumes.
    """
    observations = []
    for t, x in zip(t_s, truth, strict=True):
        for m in models:
            if not m.is_available(float(t), x):
                continue
            root = np.linalg.cholesky(m.noise_covariance)
            z = m.predict(float(t), x) + noise_scale * root @ rng.standard_normal(m.dim)
            observations.append(Observation(float(t), z, m))
    return observations


@dataclass(frozen=True)
class Scenario:
    """A fixed truth, a tracking network, and an initial uncertainty.

    Attributes
    ----------
    forces
        Force models shared by truth and filter.
    x0_true
        True initial state, km and km/s.
    p0
        Initial covariance the filter is given; initial estimates are drawn
        from N(x0_true, p0), so p0 is also the true initial error covariance.
    t_s
        Time grid, s. Observations are taken on it and estimates reported on it.
    models
        Measurement models.
    noise_scale
        See :func:`simulate_observations`.
    """

    forces: Sequence[ForceModel]
    x0_true: FloatArray
    p0: FloatArray
    t_s: FloatArray
    models: Sequence[MeasurementModel]
    noise_scale: float = 1.0

    def truth(self) -> FloatArray:
        """True states on ``t_s``."""
        return truth_trajectory(self.forces, self.x0_true, self.t_s)


@dataclass(frozen=True)
class MonteCarloResult:
    """Per-run errors and covariances for one filter.

    Attributes
    ----------
    t_s
        Report times, shape ``(K,)``.
    errors
        ``x_true - x_hat``, shape ``(N, K, 6)``.
    P
        Filter covariances, shape ``(N, K, 6, 6)``.
    nis, nis_dof
        Per-entry NIS and its dof, shape ``(N, K)``; NaN / 0 where no
        observation was processed.
    updated
        True where the entry is a posterior, shape ``(K,)``.
    """

    t_s: FloatArray
    errors: FloatArray
    P: FloatArray
    nis: FloatArray
    nis_dof: np.ndarray
    updated: np.ndarray


def monte_carlo(
    scenario: Scenario,
    filters: Sequence[SequentialFilter],
    n_runs: int,
    seed: int,
) -> dict[str, MonteCarloResult]:
    """Run every filter on the same ``n_runs`` noise realisations.

    Each run draws one initial estimate and one set of measurement noise;
    all filters see identical inputs, so their differences are the filters'.
    """
    truth = scenario.truth()
    rng = np.random.default_rng(seed)
    root_p0 = np.linalg.cholesky(scenario.p0)
    histories: dict[str, list[FilterHistory]] = {f.name: [] for f in filters}

    for _ in range(n_runs):
        x0_hat = scenario.x0_true + root_p0 @ rng.standard_normal(6)
        obs = simulate_observations(scenario.t_s, truth, scenario.models, rng,
                                    scenario.noise_scale)
        for f in filters:
            histories[f.name].append(
                f.run(x0_hat, scenario.p0, float(scenario.t_s[0]), obs, scenario.t_s[1:])
            )

    results = {}
    for name, runs in histories.items():
        t = runs[0].t_s
        index = np.searchsorted(scenario.t_s, t)
        if not np.allclose(scenario.t_s[index], t):
            raise RuntimeError("filter history times are not on the scenario grid")
        errors = np.array([truth[index] - h.x for h in runs])
        results[name] = MonteCarloResult(
            t_s=t,
            errors=errors,
            P=np.array([h.P for h in runs]),
            nis=np.array([h.nis for h in runs]),
            nis_dof=runs[0].nis_dof,
            updated=runs[0].updated,
        )
    return results


def circular_orbit_state(
    a_km: float, inclination_deg: float, raan_deg: float = 0.0
) -> FloatArray:
    """State of a circular orbit at its ascending node, km and km/s, ECI_J2000."""
    c = rot_z(np.radians(raan_deg)) @ rot_x(np.radians(inclination_deg))
    speed = np.sqrt(MU_EARTH_KM3_S2 / a_km)
    return np.concatenate([c @ [a_km, 0.0, 0.0], c @ [0.0, speed, 0.0]])


def tracking_network(
    sigma_range_km: float = 0.010,
    sigma_range_rate_km_s: float = 1.0e-5,
    min_elevation_deg: float = 10.0,
    epoch: datetime | None = None,
) -> list[RangeRangeRate]:
    """Range/range-rate models for the three default sites."""
    earth = EarthRotation(DEFAULT_EPOCH if epoch is None else epoch)
    return [
        RangeRangeRate(
            GroundStation(name, lat, lon, alt, earth, min_elevation_deg),
            sigma_range_km=sigma_range_km,
            sigma_range_rate_km_s=sigma_range_rate_km_s,
        )
        for name, lat, lon, alt in DEFAULT_SITES
    ]
