"""Adaptive eighth-order Dormand-Prince, via ``scipy.integrate.solve_ivp``."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp

from orbital.integrators.base import (
    RHS,
    Constraint,
    FloatArray,
    IntegrationResult,
    check_times,
)

#: Guard against a constraint that re-triggers immediately after projection.
_MAX_RESTARTS = 10_000


@dataclass(frozen=True)
class DOP853:
    """Adaptive DOP853 with error control ``|err_i| <= atol + rtol |y_i|``.

    Constraint handling: ``solve_ivp`` offers no hook to modify the state
    between steps, so the violation is watched with a terminal event. When it
    crosses the tolerance the integration stops at that instant, the state is
    projected, and integration restarts from the projected state. With tight
    tolerances this rarely fires; the count is reported either way.

    Parameters
    ----------
    rtol, atol
        Relative and absolute local error tolerances.
    max_step_s
        Upper bound on the step, s. ``inf`` lets the controller decide.
    """

    rtol: float = 1e-10
    atol: float = 1e-12
    max_step_s: float = np.inf

    @property
    def name(self) -> str:
        return f"DOP853 (rtol={self.rtol:g})"

    def integrate(
        self,
        f: RHS,
        y0: ArrayLike,
        t_eval: ArrayLike,
        constraint: Constraint | None = None,
    ) -> IntegrationResult:
        t_out = check_times(t_eval)
        y = np.array(y0, dtype=float)
        nprojections = 0
        max_violation = 0.0

        events = None
        if constraint is not None:
            max_violation = constraint.violation(y)
            if max_violation > constraint.tolerance:
                y = constraint.project(y)
                nprojections += 1

            def drift(t: float, state: FloatArray) -> float:
                return constraint.violation(state) - constraint.tolerance

            drift.terminal = True  # type: ignore[attr-defined]
            drift.direction = 1.0  # type: ignore[attr-defined]
            events = [drift]

        ts: list[FloatArray] = [t_out[:1]]
        ys: list[FloatArray] = [y[None, :]]
        t_start, t_end = float(t_out[0]), float(t_out[-1])
        nfev = 0

        for _ in range(_MAX_RESTARTS):
            pending = t_out[t_out > t_start]
            sol = solve_ivp(
                f, (t_start, t_end), y, method="DOP853", t_eval=pending,
                events=events, rtol=self.rtol, atol=self.atol,
                max_step=self.max_step_s,
            )
            if not sol.success:
                raise RuntimeError(f"DOP853 failed after t={t_start}: {sol.message}")
            nfev += int(sol.nfev)
            t_seg = np.asarray(sol.t, dtype=float)
            if t_seg.size:
                ts.append(t_seg)
                ys.append(np.asarray(sol.y, dtype=float).T)
            if sol.status != 1:  # reached t_end
                break
            # Terminal event: project and restart from the event state.
            t_start = float(sol.t_events[0][0])
            y_event = sol.y_events[0][0]
            max_violation = max(max_violation, constraint.violation(y_event))  # type: ignore[union-attr]
            y = constraint.project(y_event)  # type: ignore[union-attr]
            nprojections += 1
        else:
            raise RuntimeError("constraint projection restarted too many times")

        t_all = np.concatenate(ts)
        y_all = np.concatenate(ys)
        if constraint is not None:
            max_violation = max(max_violation, max(constraint.violation(s) for s in y_all))
        return IntegrationResult(t_all, y_all, nfev, nprojections, max_violation)
