"""Orbit process model ``dx/dt = [v, a(r)]`` and its STM.

    dPhi/dt = [[0, I], [da/dr, 0]] Phi,   Phi(t0) = I

``da/dr`` is a central difference (1 m step), so any force model plugs in.
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

_UNIT_BODY = MassProperties(1.0, InertiaTensor.diagonal(1.0, 1.0, 1.0))
_ZERO3 = np.zeros(3)

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
        """Total acceleration, km/s^2, shape (k, 3), for positions (k, 3) in km."""
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
        """Total acceleration at one position, km/s^2."""
        return self.accelerations(t_s, r_km)[0]

    def acceleration_gradient(self, t_s: float, r_km: FloatArray) -> FloatArray:
        """``da/dr`` by central differences, 1/s^2, shape (3, 3)."""
        h = JACOBIAN_STEP_KM
        stencil = np.vstack([r_km + h * np.eye(3), r_km - h * np.eye(3)])
        a = self.accelerations(t_s, stencil)
        return ((a[:3] - a[3:]) / (2 * h)).T

    def derivative(self, t_s: float, x: FloatArray) -> FloatArray:
        """Right-hand side ``dx/dt`` for one 6-element state."""
        return np.concatenate([x[3:6], self.acceleration(t_s, x[:3])])

    def propagate(self, x: FloatArray, t0_s: float, t1_s: float) -> FloatArray:
        """Propagate one state from ``t0_s`` to ``t1_s``."""
        if t1_s == t0_s:
            return np.array(x, dtype=float)
        return self.integrator.integrate(self.derivative, x, [t0_s, t1_s]).y[-1]

    def propagate_many(self, xs: FloatArray, t0_s: float, t1_s: float) -> FloatArray:
        """Propagate states ``(k, 6)`` as one stacked ODE with shared step control."""
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
        """Propagate a state with its 6x6 state transition matrix."""
        if t1_s == t0_s:
            return np.array(x, dtype=float), np.eye(6)

        h = JACOBIAN_STEP_KM
        offsets = np.vstack([np.zeros(3), h * np.eye(3), -h * np.eye(3)])

        def rhs(t: float, y: FloatArray) -> FloatArray:
            phi = y[6:].reshape(6, 6)
            a = self.accelerations(t, y[:3] + offsets)
            grad = ((a[1:4] - a[4:7]) / (2 * h)).T
            dphi = np.vstack([phi[3:], grad @ phi[:3]])
            return np.concatenate([y[3:6], a[0], dphi.ravel()])

        y0 = np.concatenate([x, np.eye(6).ravel()])
        y1 = self.integrator.integrate(rhs, y0, [t0_s, t1_s]).y[-1]
        return y1[:6], y1[6:].reshape(6, 6)


def white_noise_acceleration(dt_s: float, psd_km2_s3: float) -> FloatArray:
    """White-acceleration process noise ``q [[dt^3/3 I, dt^2/2 I], [dt^2/2 I, dt I]]``."""
    dt = abs(dt_s)
    eye = np.eye(3)
    return psd_km2_s3 * np.block([
        [dt**3 / 3.0 * eye, dt**2 / 2.0 * eye],
        [dt**2 / 2.0 * eye, dt * eye],
    ])
