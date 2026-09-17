"""Shared filter machinery: observations, histories, and the run loop.

Both filters are sequential: predict to the next observation time, then
update with that observation. They differ only in *how* they predict and
update, so the loop and the bookkeeping live here and each filter supplies
:meth:`SequentialFilter.predict` and :meth:`SequentialFilter.update`.

Every quantity recorded is in the units of the state (km, km/s) or of the
measurement model that produced it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from orbital.attitude.quaternion import FloatArray
from orbital.estimation.measurements.base import MeasurementModel
from orbital.estimation.orbit_model import OrbitModel, white_noise_acceleration

STATE_DIM = 6


@dataclass(frozen=True)
class Observation:
    """One measurement vector ``z`` taken at ``t_s`` by ``model``."""

    t_s: float
    z: FloatArray
    model: MeasurementModel


@dataclass(frozen=True)
class UpdateInfo:
    """What a measurement update saw: innovation and its covariance."""

    innovation: FloatArray
    innovation_cov: FloatArray

    @property
    def nis(self) -> float:
        """Normalised innovation squared ``y^T S^-1 y`` (chi-square, dim dof)."""
        return float(self.innovation @ np.linalg.solve(self.innovation_cov, self.innovation))


@dataclass(frozen=True)
class FilterHistory:
    """Estimates at every output and observation time.

    Attributes
    ----------
    t_s
        Times, s, shape ``(K,)``, non-decreasing.
    x
        Estimates, shape ``(K, 6)``: posterior where an observation was
        processed, prior elsewhere.
    P
        Covariances, shape ``(K, 6, 6)``.
    updated
        True where the entry is a posterior.
    nis
        NIS per processed observation, NaN elsewhere.
    nis_dof
        Measurement dimension per processed observation, 0 elsewhere.
    """

    t_s: FloatArray
    x: FloatArray
    P: FloatArray
    updated: np.ndarray
    nis: FloatArray
    nis_dof: np.ndarray

    def sigma(self) -> FloatArray:
        """1-sigma per state component, shape ``(K, 6)``."""
        return np.sqrt(np.einsum("kii->ki", self.P))


def symmetrize(p: FloatArray) -> FloatArray:
    """Remove the antisymmetric round-off that accumulates in P."""
    return 0.5 * (p + p.T)


class SequentialFilter(ABC):
    """Base class for EKF and UKF.

    Parameters
    ----------
    model
        Orbit dynamics.
    process_noise_psd
        White-acceleration spectral density, km^2/s^3. Zero means the filter
        trusts its dynamics exactly -- correct when truth and filter share a
        force model, as they do in the simulations here.
    """

    name: str = "filter"

    def __init__(self, model: OrbitModel, process_noise_psd: float = 0.0) -> None:
        self.model = model
        self.process_noise_psd = process_noise_psd

    def process_noise(self, dt_s: float) -> FloatArray:
        """Q for a propagation of ``dt_s`` seconds."""
        return white_noise_acceleration(dt_s, self.process_noise_psd)

    @abstractmethod
    def predict(
        self, x: FloatArray, p: FloatArray, t0_s: float, t1_s: float
    ) -> tuple[FloatArray, FloatArray]:
        """Propagate mean and covariance.

        Parameters
        ----------
        x
            Mean state ``[r (km), v (km/s)]``, shape (6,).
        p
            Covariance, km^2 and (km/s)^2 blocks, shape (6, 6).
        t0_s, t1_s
            Start and end times, s since the reference epoch.

        Returns
        -------
        x1 : numpy.ndarray
            Predicted mean, shape (6,).
        p1 : numpy.ndarray
            Predicted covariance, shape (6, 6).
        """

    @abstractmethod
    def update(
        self, x: FloatArray, p: FloatArray, obs: Observation
    ) -> tuple[FloatArray, FloatArray, UpdateInfo]:
        """Condition on one observation taken at the current time.

        Parameters
        ----------
        x
            Prior mean, shape (6,).
        p
            Prior covariance, shape (6, 6).
        obs
            The observation, carrying its own measurement model.

        Returns
        -------
        x_post : numpy.ndarray
            Posterior mean, shape (6,).
        p_post : numpy.ndarray
            Posterior covariance, shape (6, 6).
        info : UpdateInfo
            Innovation and its covariance, for the NIS test.
        """

    def run(
        self,
        x0: ArrayLike,
        p0: ArrayLike,
        t0_s: float,
        observations: Sequence[Observation],
        t_out_s: ArrayLike | None = None,
    ) -> FilterHistory:
        """Process ``observations`` in time order.

        Parameters
        ----------
        x0, p0
            Initial estimate (km, km/s) and covariance at ``t0_s``.
        t0_s
            Initial time, s.
        observations
            Measurements at or after ``t0_s``; sorted here by time.
        t_out_s
            Extra times to report the (prior) estimate at, e.g. to show
            covariance growth between tracking passes.
        """
        x = np.array(x0, dtype=float)
        p = symmetrize(np.array(p0, dtype=float))
        if x.shape != (STATE_DIM,) or p.shape != (STATE_DIM, STATE_DIM):
            raise ValueError("x0 must have shape (6,) and p0 shape (6, 6)")

        # (time, order, observation-or-None); order puts outputs before
        # observations at equal times, so the output shows the prior.
        events: list[tuple[float, int, Observation | None]] = [
            (float(o.t_s), 1, o) for o in observations
        ]
        if t_out_s is not None:
            events += [(float(t), 0, None) for t in np.asarray(t_out_s, dtype=float)]
        events.sort(key=lambda e: (e[0], e[1]))
        if events and events[0][0] < t0_s:
            raise ValueError("observations and outputs must not precede t0_s")

        ts, xs, ps, upd, nis, dof = [t0_s], [x], [p], [False], [np.nan], [0]
        t = t0_s
        for t_event, _, obs in events:
            if t_event > t:
                x, p = self.predict(x, p, t, t_event)
                t = t_event
            if obs is not None:
                x, p, info = self.update(x, p, obs)
                nis.append(info.nis)
                dof.append(obs.model.dim)
                upd.append(True)
            else:
                nis.append(np.nan)
                dof.append(0)
                upd.append(False)
            ts.append(t)
            xs.append(x)
            ps.append(p)

        return FilterHistory(
            np.array(ts), np.array(xs), np.array(ps),
            np.array(upd), np.array(nis), np.array(dof),
        )
