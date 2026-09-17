"""The integrator interface and the constraint-projection contract.

Every integrator takes a right-hand side ``f(t, y)``, an initial state, and
the output times, and returns an :class:`IntegrationResult`. Units are those
of the caller's state; time is seconds.

Constraints
-----------
Some states live on a manifold the ODE preserves but a discrete integrator
does not -- the unit quaternion is the case here. A :class:`Constraint` says
how to measure the violation and how to project back. Integrators apply the
projection whenever the violation exceeds ``tolerance`` and report how often
they did, so drift is handled explicitly and remains visible in the output
rather than being silently absorbed.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
RHS = Callable[[float, FloatArray], FloatArray]


@dataclass(frozen=True)
class Constraint:
    """A state constraint and its projection.

    Attributes
    ----------
    violation
        Maps a state to a non-negative violation measure (dimensionless).
    project
        Maps a state to the nearest state satisfying the constraint.
    tolerance
        Violation above which the integrator projects. ``0.0`` means
        project after every step.
    """

    violation: Callable[[FloatArray], float]
    project: Callable[[FloatArray], FloatArray]
    tolerance: float = 0.0


@dataclass(frozen=True)
class IntegrationResult:
    """Output of an integration.

    Attributes
    ----------
    t
        Output times, s, shape ``(N,)``. Equal to the requested times.
    y
        States at ``t``, shape ``(N, n)``.
    nfev
        Right-hand-side evaluations -- the cost measure used for comparisons.
    nprojections
        Times the constraint projection was applied.
    max_violation
        Largest constraint violation observed before any projection
        (0.0 when no constraint was given).
    """

    t: FloatArray
    y: FloatArray
    nfev: int
    nprojections: int
    max_violation: float


class Integrator(Protocol):
    """Anything that can advance ``dy/dt = f(t, y)`` to a set of output times."""

    @property
    def name(self) -> str:
        """Short label for tables and plots."""
        ...

    def integrate(
        self,
        f: RHS,
        y0: ArrayLike,
        t_eval: ArrayLike,
        constraint: Constraint | None = None,
    ) -> IntegrationResult:
        """Integrate from ``t_eval[0]`` and report the state at every output time.

        Parameters
        ----------
        f
            Right-hand side ``f(t, y)``, in the caller's units per second.
        y0
            Initial state at ``t_eval[0]``.
        t_eval
            Output times, s, strictly increasing.
        constraint
            Optional state constraint to enforce during the integration.

        Returns
        -------
        IntegrationResult
            States at ``t_eval``, plus cost and projection statistics.
        """
        ...


def check_times(t_eval: ArrayLike) -> FloatArray:
    """Validate output times: 1-D, at least two, strictly increasing."""
    t = np.asarray(t_eval, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("t_eval must be a 1-D array with at least two times")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("t_eval must be strictly increasing")
    return t
