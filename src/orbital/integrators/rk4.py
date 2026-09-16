"""Classical fixed-step fourth-order Runge-Kutta."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from orbital.integrators.base import (
    RHS,
    Constraint,
    IntegrationResult,
    check_times,
)


@dataclass(frozen=True)
class RK4:
    """Fixed-step RK4. Global error scales as ``step_s**4``.

    Parameters
    ----------
    step_s
        Maximum step, s. Each interval between output times is split into
        the fewest equal steps no longer than this, so every output time is
        hit exactly rather than interpolated.
    """

    step_s: float

    def __post_init__(self) -> None:
        if not self.step_s > 0.0:
            raise ValueError("step_s must be positive")

    @property
    def name(self) -> str:
        return f"RK4 (h={self.step_s:g} s)"

    def integrate(
        self,
        f: RHS,
        y0: ArrayLike,
        t_eval: ArrayLike,
        constraint: Constraint | None = None,
    ) -> IntegrationResult:
        t_out = check_times(t_eval)
        y = np.array(y0, dtype=float)
        out = np.empty((t_out.size, y.size))
        nprojections = 0
        max_violation = 0.0

        if constraint is not None:
            max_violation = constraint.violation(y)
            if max_violation > constraint.tolerance:
                y = constraint.project(y)
                nprojections += 1
        out[0] = y

        nfev = 0
        for i in range(1, t_out.size):
            t0, t1 = t_out[i - 1], t_out[i]
            nsub = max(1, int(np.ceil((t1 - t0) / self.step_s - 1e-12)))
            h = (t1 - t0) / nsub
            for j in range(nsub):
                t = t0 + j * h
                k1 = f(t, y)
                k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
                k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
                k4 = f(t + h, y + h * k3)
                y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                nfev += 4
                if constraint is not None:
                    v = constraint.violation(y)
                    max_violation = max(max_violation, v)
                    if v > constraint.tolerance:
                        y = constraint.project(y)
                        nprojections += 1
            out[i] = y

        return IntegrationResult(t_out, out, nfev, nprojections, max_violation)
