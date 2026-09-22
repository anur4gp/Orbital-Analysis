"""6-DOF propagator validation curves (fig5); assertions live in tests/test_dynamics.py.

Run: python scripts/validate_6dof.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from orbital.attitude import quaternion as quat
from orbital.attitude.dcm import to_dcm
from orbital.dynamics import InertiaTensor, MassProperties, RigidBody, RigidBodyState
from orbital.dynamics import diagnostics as dg
from orbital.integrators import DOP853, RK4
from orbital.plotting import AQUA, BLUE, FIGURE_DIR, INK2, MUTED, ORANGE, despine, save, style

# Floor for zero errors on log axes.
FLOOR = 1e-17

T = np.linspace(0.0, 3000.0, 301)
TRIAXIAL = InertiaTensor.diagonal(80.0, 120.0, 150.0)
TUMBLE = RigidBodyState([7000.0, 0, 0], [0, 0, 0],
                        quat.from_axis_angle([1, 2, 3], 0.7), [0.1, 0.02, -0.15])

SYMMETRIC = InertiaTensor.diagonal(100.0, 100.0, 150.0)
W0 = np.array([0.05, 0.02, 0.3])
Q0 = quat.from_axis_angle([0.3, -1.0, 0.5], 1.1)

# (label, integrator, quaternion_tol, colour, marker, dash)
CASES = [
    ("RK4, h = 0.5 s", RK4(0.5), 1e-12, ORANGE, "s", "--"),
    ("RK4, h = 0.25 s", RK4(0.25), 1e-12, AQUA, "^", ":"),
    ("DOP853, rtol = 1e-12", DOP853(1e-12, 1e-14), 1e-12, BLUE, "o", "-"),
]


def closed_form_attitude(t: np.ndarray) -> list[np.ndarray]:
    """Torque-free axisymmetric attitude; see tests/test_dynamics.py."""
    it, i3 = SYMMETRIC.matrix[0, 0], SYMMETRIC.matrix[2, 2]
    h = to_dcm(Q0) @ (SYMMETRIC.matrix @ W0)
    phi_dot = np.linalg.norm(h) / it
    lam = (i3 - it) * W0[2] / it
    return [
        quat.multiply(quat.multiply(quat.from_axis_angle(h, phi_dot * ti), Q0),
                      quat.from_axis_angle([0, 0, 1], -lam * ti))
        for ti in t
    ]


def main() -> int:
    tumbler = RigidBody(MassProperties(500.0, TRIAXIAL))
    top = RigidBody(MassProperties(10.0, SYMMETRIC))
    top_initial = RigidBodyState([7000.0, 0, 0], [0, 0, 0], Q0, W0)
    q_exact = closed_form_attitude(T)

    rows = []
    energy_curves, momentum_curves, attitude_curves = [], [], []
    for label, integrator, tol, *_ in CASES:
        tr = tumbler.propagate(TUMBLE, T, integrator, quaternion_tol=tol)
        energy = dg.rotational_kinetic_energy(tr.omega_rad_s, TRIAXIAL)
        h = dg.angular_momentum_inertial(tr.q, tr.omega_rad_s, TRIAXIAL)
        d_e = np.abs(energy / energy[0] - 1.0)
        d_h = np.linalg.norm(h - h[0], axis=1) / np.linalg.norm(h[0])

        sym = top.propagate(top_initial, T, integrator, quaternion_tol=tol)
        att = np.array([quat.rotation_angle(quat.multiply(qn, quat.conjugate(qe)))
                        for qn, qe in zip(sym.q, q_exact, strict=True)])

        energy_curves.append(d_e)
        momentum_curves.append(d_h)
        attitude_curves.append(att)
        rows.append((label, d_e.max(), d_h.max(), att.max(),
                     tr.integration.nfev, tr.integration.nprojections))

    unprojected = tumbler.propagate(TUMBLE, T, RK4(0.5), quaternion_tol=None)
    norm_free = np.abs(np.linalg.norm(unprojected.q, axis=1) - 1.0)
    projected = tumbler.propagate(TUMBLE, T, RK4(0.5), quaternion_tol=1e-12)
    norm_pinned = np.abs(np.linalg.norm(projected.q, axis=1) - 1.0)

    print(f"Torque-free validation over {T[-1]:.0f} s\n")
    print(f"{'integrator':<22}{'max |dT/T|':>12}{'max |dH|/|H|':>14}"
          f"{'attitude err':>14}{'RHS evals':>11}{'projections':>13}")
    for label, de, dh, att, nfev, nproj in rows:
        print(f"{label:<22}{de:>12.2e}{dh:>14.2e}{att:>12.2e} rad{nfev:>9d}{nproj:>13d}")
    print(f"\nQuaternion norm, RK4 h = 0.5 s: {norm_free.max():.2e} unprojected, "
          f"{norm_pinned.max():.2e} projected at 1e-12 "
          f"({projected.integration.nprojections} projections)")

    style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharex=True, constrained_layout=True)
    panels = [
        (axes[0], energy_curves, "(a) Energy drift", r"$|\Delta T / T_0|$"),
        (axes[1], attitude_curves, "(b) Precession vs closed form", "attitude error (rad)"),
    ]
    for ax, curves, title, ylabel in panels:
        for (label, _, _, colour, marker, dash), y in zip(CASES, curves, strict=True):
            ax.semilogy(T, np.maximum(y, FLOOR), color=colour, linestyle=dash, linewidth=1.6,
                        marker=marker, markevery=50, markersize=4, label=label)
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)

    ax = axes[2]
    ax.semilogy(T, np.maximum(norm_free, FLOOR), color=ORANGE, linestyle="--", linewidth=1.6,
                marker="s", markevery=50, markersize=4, label="RK4, h = 0.5 s, no projection")
    ax.semilogy(T, np.maximum(norm_pinned, FLOOR), color=INK2, linestyle="-", linewidth=1.6,
                marker="D", markevery=50, markersize=4, label="RK4, h = 0.5 s, projected")
    ax.axhline(1e-12, color=MUTED, linewidth=0.8, linestyle=(0, (1, 2)))
    ax.text(T[-1], 1.8e-12, "projection threshold", ha="right", va="bottom",
            fontsize=7, color=INK2)
    ax.set_title("(c) Quaternion norm", loc="left")
    ax.set_ylabel(r"$|\,\|q\| - 1\,|$")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper left", ncols=3, title="(a), (b):",
               alignment="left", title_fontsize=8)
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right", ncols=1, title="(c):",
               alignment="left", title_fontsize=8)
    for ax in axes:
        ax.set_xlabel("time (s)")
        ax.set_ylim(FLOOR / 3, 1e-2)
        despine(ax)

    save(fig, "fig5_6dof_validation")
    print(f"\nwrote {FIGURE_DIR / 'fig5_6dof_validation.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
