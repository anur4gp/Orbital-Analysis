"""Translational process model for orbit determination.

The filter state is ``x = [r (km), v (km/s)]`` in ECI_J2000. The dynamics
reuse the Phase 1 force models, so truth and filter share one implementation
of the physics::

    dx/dt = f(x) = [v, a(r)]

The EKF also needs the state transition matrix, obtained by integrating the
variational equations alongside the state::

    dPhi/dt = A(x) Phi,     A = [[0, I], [da/dr, 0]],     Phi(t0) = I

``da/dr`` is taken by central differences, so any conservative force model
can be dropped in without deriving its gradient by hand. With a 1 m step the
truncation error is O(h^2 |d^3 a|) and the relative error is about 1e-9,
far below anything the filter can resolve.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from orbital.attitude.quaternion import IDENTITY, FloatArray
from orbital.dynamics.forces import ForceModel, PositionOnlyForce
from orbital.dynamics.inertia import InertiaTensor, MassProperties
from orbital.dynamics.state import RigidBodyState
from orbital.integrators.adaptive import DOP853
from orbital.integrators.base import Integrator

#: Translational forces are per unit mass; attitude does not enter the
#: gravity models, so a nominal unit body is enough.
_UNIT_BODY = MassProperties(1.0, InertiaTensor.diagonal(1.0, 1.0, 1.0))
_ZERO3 = np.zeros(3)

#: Central-difference step for da/dr, km.
JACOBIAN_STEP_KM = 1e-3


@dataclass(frozen=True)
class OrbitModel:
    """Orbit dynamics and their linearisation.

    Parameters
    ----------
    forces
        Force models (accelerations are summed).
    integrator
        Integrator used for every propagation, state and STM alike.
    """

    forces: Sequence[ForceModel]
    integrator: Integrator = field(default_factory=lambda: DOP853(rtol=1e-11, atol=1e-12))

    def accelerations(self, t_s: float, r_km: FloatArray) -> FloatArray:
        """Total acceleration, km/s^2, for positions of shape ``(k, 3)``.

        Vectorised when every force model depends only on position;
        otherwise each position is evaluated through the general force
        interface.

        Parameters
        ----------
        t_s
            Time since the reference epoch, s.
        r_km
            Positions, km, ECI_J2000, shape (k, 3) or (3,).

        Returns
        -------
        numpy.ndarray
            Accelerations, km/s^2, shape (k, 3).
        """
        r = np.atleast_2d(r_km)
        if all(isinstance(f, PositionOnlyForce) for f in self.forces):
            total = np.zeros_like(r)
            for force in self.forces:
                total = total + force.acceleration_many(r)  # type: ignore[attr-defined]
            return total
        out = np.zeros_like(r)
        for i, ri in enumerate(r):
            state = RigidBodyState(ri, _ZERO3, IDENTITY, _ZERO3, t_s)
            for force in self.forces:
                out[i] += force.acceleration(state, _UNIT_BODY)
        return out

    def acceleration(self, t_s: float, r_km: FloatArray) -> FloatArray:
        """Total acceleration at one position.

        Parameters
        ----------
        t_s
            Time since the reference epoch, s.
        r_km
            Position, km, ECI_J2000, shape (3,).

        Returns
        -------
        numpy.ndarray
            Acceleration, km/s^2, shape (3,).
        """
        return self.accelerations(t_s, r_km)[0]

    def acceleration_gradient(self, t_s: float, r_km: FloatArray) -> FloatArray:
        """``da/dr`` by central differences, in one batched evaluation.

        Parameters
        ----------
        t_s
            Time since the reference epoch, s.
        r_km
            Position, km, shape (3,).

        Returns
        -------
        numpy.ndarray
            Gradient, 1/s^2, shape (3, 3).
        """
        h = JACOBIAN_STEP_KM
        stencil = np.vstack([r_km + h * np.eye(3), r_km - h * np.eye(3)])
        a = self.accelerations(t_s, stencil)
        return ((a[:3] - a[3:]) / (2 * h)).T

    def derivative(self, t_s: float, x: FloatArray) -> FloatArray:
        """Right-hand side ``dx/dt`` for one 6-element state.

        Parameters
        ----------
        t_s
            Time since the reference epoch, s.
        x
            State ``[r (km), v (km/s)]``, shape (6,).

        Returns
        -------
        numpy.ndarray
            Derivative, shape (6,), in km/s and km/s^2.
        """
        return np.concatenate([x[3:6], self.acceleration(t_s, x[:3])])

    def propagate(self, x: FloatArray, t0_s: float, t1_s: float) -> FloatArray:
        """Propagate one state.

        Parameters
        ----------
        x
            State at ``t0_s``, shape (6,).
        t0_s, t1_s
            Start and end times, s since the reference epoch.

        Returns
        -------
        numpy.ndarray
            State at ``t1_s``, shape (6,).
        """
        if t1_s == t0_s:
            return np.array(x, dtype=float)
        return self.integrator.integrate(self.derivative, x, [t0_s, t1_s]).y[-1]

    def propagate_many(self, xs: FloatArray, t0_s: float, t1_s: float) -> FloatArray:
        """Propagate ``k`` states, shape ``(k, 6)``, as one stacked ODE.

        One integration with shared step control is much cheaper than ``k``
        separate calls, and the steps are at least as small as the most
        demanding member needs.

        Parameters
        ----------
        xs
            States at ``t0_s``, shape (k, 6).
        t0_s, t1_s
            Start and end times, s since the reference epoch.

        Returns
        -------
        numpy.ndarray
            States at ``t1_s``, shape (k, 6).
        """
        xs = np.asarray(xs, dtype=float)
        if t1_s == t0_s:
            return xs.copy()
        k = xs.shape[0]

        def rhs(t: float, y: FloatArray) -> FloatArray:
            ys = y.reshape(k, 6)
            return np.hstack([ys[:, 3:], self.accelerations(t, ys[:, :3])]).ravel()

        return self.integrator.integrate(rhs, xs.ravel(), [t0_s, t1_s]).y[-1].reshape(k, 6)

    def propagate_with_stm(
        self, x: FloatArray, t0_s: float, t1_s: float
    ) -> tuple[FloatArray, FloatArray]:
        """Propagate a state together with its state transition matrix.

        Parameters
        ----------
        x
            State at ``t0_s``, shape (6,).
        t0_s, t1_s
            Start and end times, s since the reference epoch.

        Returns
        -------
        x1 : numpy.ndarray
            State at ``t1_s``, shape (6,).
        phi : numpy.ndarray
            State transition matrix, shape (6, 6).
        """
        if t1_s == t0_s:
            return np.array(x, dtype=float), np.eye(6)

        h = JACOBIAN_STEP_KM
        offsets = np.vstack([np.zeros(3), h * np.eye(3), -h * np.eye(3)])

        def rhs(t: float, y: FloatArray) -> FloatArray:
            phi = y[6:].reshape(6, 6)
            # Nominal point and the 6-point stencil in one batched evaluation.
            a = self.accelerations(t, y[:3] + offsets)
            grad = ((a[1:4] - a[4:7]) / (2 * h)).T
            # A Phi with A = [[0, I], [grad, 0]], without forming A.
            dphi = np.vstack([phi[3:], grad @ phi[:3]])
            return np.concatenate([y[3:6], a[0], dphi.ravel()])

        y0 = np.concatenate([x, np.eye(6).ravel()])
        y1 = self.integrator.integrate(rhs, y0, [t0_s, t1_s]).y[-1]
        return y1[:6], y1[6:].reshape(6, 6)


def white_noise_acceleration(dt_s: float, psd_km2_s3: float) -> FloatArray:
    """Process noise Q for white acceleration noise of spectral density ``psd``.

    Discretised state-noise compensation::

        Q = q [[dt^3/3 I, dt^2/2 I],
               [dt^2/2 I, dt     I]]

    Parameters
    ----------
    dt_s
        Propagation interval, s. The sign is ignored.
    psd_km2_s3
        Acceleration power spectral density, km^2/s^3.
    """
    dt = abs(dt_s)
    eye = np.eye(3)
    return psd_km2_s3 * np.block([
        [dt**3 / 3.0 * eye, dt**2 / 2.0 * eye],
        [dt**2 / 2.0 * eye, dt * eye],
    ])
