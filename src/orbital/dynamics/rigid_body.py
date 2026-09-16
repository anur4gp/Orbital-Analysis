"""The 6-DOF equations of motion and the propagation entry point.

Translation (ECI_J2000, per unit mass)::

    dr/dt = v
    dv/dt = sum of force-model accelerations

Rotation (BODY axes)::

    dq/dt     = 1/2 q ⊗ [0, omega]
    I domega/dt = tau - omega x (I omega)          (Euler's equations)

Translation and rotation are coupled only through the models: a torque may
depend on position (gravity gradient), a force on attitude (drag, SRP).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from orbital.attitude.quaternion import FloatArray, kinematics
from orbital.conventions import Frame
from orbital.dynamics.forces import ForceModel
from orbital.dynamics.inertia import MassProperties
from orbital.dynamics.state import STATE_SIZE, Q, R, RigidBodyState, V, W
from orbital.dynamics.torques import TorqueModel
from orbital.integrators.base import Constraint, IntegrationResult, Integrator

#: Default projection threshold for the quaternion norm, dimensionless.
QUATERNION_NORM_TOL = 1e-12


def quaternion_norm_violation(y: FloatArray) -> float:
    """``| |q| - 1 |`` for a packed 13-element state."""
    return abs(float(np.linalg.norm(y[Q])) - 1.0)


def project_quaternion(y: FloatArray) -> FloatArray:
    """Copy of the packed state with its quaternion renormalised."""
    out = np.array(y, dtype=float)
    out[Q] /= np.linalg.norm(out[Q])
    return out


def quaternion_constraint(tolerance: float = QUATERNION_NORM_TOL) -> Constraint:
    """The unit-quaternion constraint, projecting above ``tolerance``."""
    return Constraint(quaternion_norm_violation, project_quaternion, tolerance)


@dataclass(frozen=True)
class Trajectory:
    """Propagated 6-DOF history.

    Attributes
    ----------
    t_s
        Times, s since the reference epoch, shape ``(N,)``.
    r_km, v_km_s
        Position (km) and velocity (km/s), shape ``(N, 3)``, in ``frame``.
    q
        Attitude quaternions BODY -> ``frame``, shape ``(N, 4)``.
    omega_rad_s
        Body angular velocity, rad/s, shape ``(N, 3)``.
    integration
        Raw integrator output, including cost and projection statistics.
    frame
        Frame of the translational states.
    """

    t_s: FloatArray
    r_km: FloatArray
    v_km_s: FloatArray
    q: FloatArray
    omega_rad_s: FloatArray
    integration: IntegrationResult
    frame: Frame = Frame.ECI_J2000

    def state(self, index: int) -> RigidBodyState:
        """The state at output ``index``."""
        return RigidBodyState(
            self.r_km[index], self.v_km_s[index], self.q[index],
            self.omega_rad_s[index], float(self.t_s[index]), self.frame,
        )


@dataclass(frozen=True)
class RigidBody:
    """A rigid spacecraft with its force and torque models.

    Parameters
    ----------
    mass_properties
        Mass and inertia tensor about the centre of mass.
    forces
        Force models; their accelerations are summed. Empty means the centre
        of mass moves in a straight line.
    torques
        Torque models; their torques are summed. Empty means torque-free.
    """

    mass_properties: MassProperties
    forces: Sequence[ForceModel] = field(default_factory=tuple)
    torques: Sequence[TorqueModel] = field(default_factory=tuple)

    def derivative(self, t_s: float, y: FloatArray) -> FloatArray:
        """Right-hand side of the 13-state ODE.

        The kinematics integrate the raw quaternion so that norm drift stays
        visible to the constraint check; the force and torque models see a
        normalised copy, so they are never evaluated at a scaled attitude.
        """
        state = RigidBodyState.from_vector(y, t_s)
        mp = self.mass_properties
        inertia = mp.inertia

        accel = np.zeros(3)
        for force in self.forces:
            accel = accel + force.acceleration(state, mp)

        tau = np.zeros(3)
        for torque_model in self.torques:
            tau = tau + torque_model.torque(state, mp)

        omega = y[W]
        omega_dot = inertia.inverse @ (tau - np.cross(omega, inertia.matrix @ omega))

        dy = np.empty(STATE_SIZE)
        dy[R] = y[V]
        dy[V] = accel
        dy[Q] = kinematics(y[Q], omega)
        dy[W] = omega_dot
        return dy

    def propagate(
        self,
        initial: RigidBodyState,
        t_eval_s: ArrayLike,
        integrator: Integrator,
        quaternion_tol: float | None = QUATERNION_NORM_TOL,
    ) -> Trajectory:
        """Propagate ``initial`` to each time in ``t_eval_s``.

        Parameters
        ----------
        initial
            Starting state; must be in an inertial frame (ECI_J2000).
        t_eval_s
            Output times, s since the reference epoch, strictly increasing.
            The first entry is the start time and is normally ``initial.t_s``.
        integrator
            Any :class:`~orbital.integrators.base.Integrator`.
        quaternion_tol
            Norm-violation threshold above which the quaternion is projected
            back to unit length. ``None`` disables projection -- useful only
            for measuring the drift it prevents.

        Raises
        ------
        ValueError
            If the initial state is not in ECI_J2000. TEME states from SGP4
            must be converted explicitly first.
        """
        if initial.frame is not Frame.ECI_J2000:
            raise ValueError(
                f"dynamics integrate in {Frame.ECI_J2000}, got {initial.frame}; "
                "convert explicitly before propagating"
            )
        constraint = None if quaternion_tol is None else quaternion_constraint(quaternion_tol)
        result = integrator.integrate(
            self.derivative, initial.to_vector(), t_eval_s, constraint
        )
        y = result.y
        return Trajectory(
            t_s=result.t, r_km=y[:, R], v_km_s=y[:, V], q=y[:, Q],
            omega_rad_s=y[:, W], integration=result,
        )
