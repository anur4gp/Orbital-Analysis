"""Phase 2 benchmark: EKF vs UKF orbit determination from ground tracking.

Truth: a 7000 km, 51.6 deg orbit under two-body + J2, propagated with the
6-DOF propagator for two revolutions. Three stations (Goldstone, Canberra,
Madrid; 10 deg mask) take range + range-rate every 60 s while in view
(sigma 10 m, 1 cm/s). The first pass starts 25 min in; then there is a
73 min gap before the next one.

Two initial uncertainties, same everything else:

  precise    10 m / 1 cm/s per axis   -- both filters should be consistent
  TLE-grade  1 km / 1 m/s per axis    -- the EKF loses consistency

For the TLE-grade case the script also isolates *where* the EKF fails, by
comparing each filter's covariance prediction across the long gap with a
Monte Carlo cloud pushed through the full nonlinear dynamics.

Monte Carlo results are cached to data/estimation_mc.npz; --force reruns.

Run: python scripts/run_estimation.py [--runs N] [--force]
"""
from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from orbital.conjunction.covariance import rtn_basis
from orbital.core.constants import MU_EARTH_KM3_S2
from orbital.dynamics import J2Gravity, TwoBodyGravity
from orbital.estimation import EKF, UKF, OrbitModel
from orbital.estimation import consistency as cs
from orbital.estimation.simulation import (
    Scenario,
    circular_orbit_state,
    monte_carlo,
    simulate_observations,
    tracking_network,
)
from orbital.paths import DATA_DIR
from orbital.plotting import BAND, BLUE, INK2, ORANGE, despine, save, style

CACHE = DATA_DIR / "estimation_mc.npz"
FORCES = (TwoBodyGravity(), J2Gravity())
MODEL = OrbitModel(FORCES)
A_KM, INC_DEG, RAAN_DEG = 7000.0, 51.6, 250.0
PERIOD_S = 2 * np.pi * np.sqrt(A_KM**3 / MU_EARTH_KM3_S2)
T_GRID = np.arange(0.0, 2 * PERIOD_S, 60.0)
CASES = {"precise": (0.01, 1e-5), "TLE-grade": (1.0, 1e-3)}
FILTERS = (EKF(MODEL), UKF(MODEL))
STYLE = {"EKF": (ORANGE, "--"), "UKF": (BLUE, "-")}
CONFIDENCE = 0.95


def initial_state() -> np.ndarray:
    return circular_orbit_state(A_KM, INC_DEG, RAAN_DEG)


def network() -> list:
    return tracking_network(sigma_range_km=0.010, sigma_range_rate_km_s=1e-5)


def scenario(sigma_r_km: float, sigma_v_km_s: float) -> Scenario:
    p0 = np.diag([sigma_r_km**2] * 3 + [sigma_v_km_s**2] * 3)
    return Scenario(FORCES, initial_state(), p0, T_GRID, network())


def run_or_load(n_runs: int, force: bool) -> dict[str, np.ndarray]:
    if CACHE.exists() and not force:
        data = dict(np.load(CACHE))
        if int(data["n_runs"]) == n_runs:
            return data
    out: dict[str, np.ndarray] = {"n_runs": np.array(n_runs)}
    for case, (sr, sv) in CASES.items():
        t0 = time.perf_counter()
        results = monte_carlo(scenario(sr, sv), FILTERS, n_runs, seed=2026)
        print(f"  {case:<10} {n_runs} runs x {len(FILTERS)} filters in "
              f"{time.perf_counter() - t0:.0f} s")
        for name, r in results.items():
            key = f"{case}/{name}"
            out[f"{key}/t"] = r.t_s
            out[f"{key}/errors"] = r.errors
            out[f"{key}/P"] = r.P
            out[f"{key}/nis"] = r.nis
            out[f"{key}/nis_dof"] = r.nis_dof
            out[f"{key}/updated"] = r.updated
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **out)
    return out


def rtn_errors(t: np.ndarray, errors: np.ndarray, cov: np.ndarray, truth: np.ndarray):
    """Position errors and 1-sigma in each time's RTN frame (km)."""
    idx = np.searchsorted(T_GRID, t)
    bases = np.array([rtn_basis(truth[i, :3], truth[i, 3:]) for i in idx])
    e_rtn = np.einsum("kji,nkj->nki", bases, errors[..., :3])
    p_rtn = np.einsum("kai,nkab,kbj->nkij", bases, cov[..., :3, :3], bases)
    return e_rtn, np.sqrt(np.einsum("nkii->nki", p_rtn))


def summary(data: dict[str, np.ndarray]) -> None:
    n = int(data["n_runs"])
    lo, hi = cs.average_bounds(6, n, CONFIDENCE)
    lo2, hi2 = cs.average_bounds(2, n, CONFIDENCE)
    print(f"\nConsistency over {n} runs ({CONFIDENCE:.0%} intervals: "
          f"NEES [{lo:.2f}, {hi:.2f}], NIS [{lo2:.2f}, {hi2:.2f}])\n")
    print(f"{'case':<11}{'filter':<7}{'NEES end':>9}{'NEES max':>10}{'in band':>9}"
          f"{'NIS mean':>10}{'pos RMSE end':>14}{'vel RMSE end':>14}")
    for case in CASES:
        for f in FILTERS:
            k = f"{case}/{f.name}"
            avg = cs.nees(data[f"{k}/errors"], data[f"{k}/P"]).mean(axis=0)
            nis = data[f"{k}/nis"][:, data[f"{k}/updated"]]
            pos = cs.rms_over_runs(data[f"{k}/errors"][..., :3])[-1]
            vel = cs.rms_over_runs(data[f"{k}/errors"][..., 3:])[-1]
            print(f"{case:<11}{f.name:<7}{avg[-1]:>9.1f}{avg.max():>10.1f}"
                  f"{cs.fraction_inside(avg[1:], (lo, hi)):>9.0%}{np.mean(nis):>10.2f}"
                  f"{pos * 1e3:>11.1f} m{vel * 1e6:>10.1f} mm/s")


def diagnose_gap() -> None:
    """Where the EKF fails: covariance prediction across the long gap."""
    sc = scenario(*CASES["TLE-grade"])
    truth = sc.truth()
    rng = np.random.default_rng(7)
    x = sc.x0_true + np.linalg.cholesky(sc.p0) @ rng.standard_normal(6)
    obs = simulate_observations(sc.t_s, truth, sc.models, rng)
    gaps = np.diff([o.t_s for o in obs])
    k = int(np.argmax(gaps))
    ekf = FILTERS[0]

    p, t = sc.p0, 0.0
    first = True
    for o in obs[: k + 1]:
        x, p = ekf.predict(x, p, t, o.t_s)
        t = o.t_s
        if first:
            index = cs.measurement_nonlinearity(o.model, t, x, p)
            print(f"\nFirst update (t = {t:.0f} s): second-order spread / noise = "
                  f"{index[0]:.2f} (range), {index[1]:.2f} (range-rate)")
            first = False
        x, p, _ = ekf.update(x, p, o)

    t1 = obs[k + 1].t_s
    cloud = rng.multivariate_normal(x, p, 4000)
    pushed = MODEL.propagate_many(cloud, t, t1)
    true_eig = np.linalg.eigvalsh(np.cov(pushed.T))
    print(f"Gap t = {t:.0f} -> {t1:.0f} s: covariance eigenvalues, smallest three "
          "(km/s-mixed units)")
    print(f"  {'Monte Carlo (4000 pts)':<24}" + "".join(f"{v:>11.2e}" for v in true_eig[:3]))
    for f in FILTERS:
        m, c = f.predict(x, p, t, t1)
        eig = np.linalg.eigvalsh(c)
        d = pushed - m
        cloud_nees = np.mean(np.einsum("ki,ki->k", d, np.linalg.solve(c, d.T).T))
        print(f"  {f.name:<24}" + "".join(f"{v:>11.2e}" for v in eig[:3])
              + f"   NEES of the cloud {cloud_nees:8.1f}")
    a = rtn_basis(pushed.mean(0)[:3], pushed.mean(0)[3:])
    print("  position sigma R/T/N, km: MC "
          + " ".join(f"{s:.3f}" for s in np.sqrt(np.diag(a.T @ np.cov(pushed[:, :3].T) @ a))))


def figures(data: dict[str, np.ndarray]) -> None:
    style()
    n = int(data["n_runs"])
    truth = scenario(*CASES["TLE-grade"]).truth()
    minutes = T_GRID / 60.0
    updated_t = data["TLE-grade/EKF/t"][data["TLE-grade/EKF/updated"]] / 60.0
    passes = np.split(updated_t, np.where(np.diff(updated_t) > 2)[0] + 1)

    def shade_passes(ax):
        for p in passes:
            ax.axvspan(p[0], p[-1], color=BAND, lw=0, zorder=0)

    # Figure 6: one run's along-track error against its 3-sigma, TLE-grade prior.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True, constrained_layout=True)
    for ax, f in zip(axes, FILTERS, strict=True):
        k = f"TLE-grade/{f.name}"
        t = data[f"{k}/t"] / 60.0
        e_rtn, s_rtn = rtn_errors(data[f"{k}/t"], data[f"{k}/errors"][:1], data[f"{k}/P"][:1],
                                  truth)
        colour, _ = STYLE[f.name]
        shade_passes(ax)
        ax.semilogy(t, np.abs(e_rtn[0, :, 1]) * 1e3, color=colour, lw=1.2, ls="-",
                    label=f"|{f.name} along-track error|")
        ax.semilogy(t, 3 * s_rtn[0, :, 1] * 1e3, color="#0b0b0b", lw=1.3, ls=(0, (4, 2)),
                    label=rf"{f.name} $3\sigma$ bound")
        ax.set_title(f"({'ab'[FILTERS.index(f)]}) {f.name}", loc="left")
        ax.set_xlabel("time (min)")
        ax.legend(loc="lower left")
        despine(ax)
    axes[0].set_ylabel("along-track (m)")
    axes[0].set_ylim(1e-2, 3e4)
    axes[1].text(minutes[-1], 1.4e-2, "tracking passes shaded", ha="right", fontsize=7,
                 color=INK2)
    save(fig, "fig6_estimation_error")
    plt.close(fig)

    # Figure 7: averaged NEES for both priors, and position RMSE.
    lo, hi = cs.average_bounds(6, n, CONFIDENCE)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)
    for ax, case, tag in ((axes[0], "precise", "a"), (axes[1], "TLE-grade", "b")):
        shade_passes(ax)
        ax.axhspan(lo, hi, color="#cfe0f5", lw=0, zorder=0.5)
        for f in FILTERS:
            k = f"{case}/{f.name}"
            avg = cs.nees(data[f"{k}/errors"], data[f"{k}/P"]).mean(axis=0)
            colour, dash = STYLE[f.name]
            ax.semilogy(data[f"{k}/t"] / 60.0, avg, color=colour, ls=dash, lw=1.6, label=f.name)
        sr, sv = CASES[case]
        ax.set_title(f"({tag}) NEES, prior {sr * 1e3:g} m / {sv * 1e3:g} m/s", loc="left",
                     fontsize=9)
        ax.set_xlabel("time (min)")
        ax.set_ylim(1, 1e4)
        despine(ax)
    axes[0].set_ylabel(f"average NEES ({n} runs)")
    axes[0].text(minutes[-1], hi * 1.1, f"{CONFIDENCE:.0%} band", ha="right", va="bottom",
                 fontsize=7, color=INK2)
    axes[0].text(minutes[-1], 60, "EKF and UKF coincide", ha="right", fontsize=7, color=INK2)

    ax = axes[2]
    shade_passes(ax)
    for f in FILTERS:
        k = f"TLE-grade/{f.name}"
        colour, dash = STYLE[f.name]
        ax.semilogy(data[f"{k}/t"] / 60.0, cs.rms_over_runs(data[f"{k}/errors"][..., :3]) * 1e3,
                    color=colour, ls=dash, lw=1.6, label=f.name)
    ax.set_title("(c) position RMSE, TLE-grade", loc="left", fontsize=9)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("RMSE (m)")
    despine(ax)
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right", ncols=2)
    save(fig, "fig7_estimation_consistency")
    plt.close(fig)
    print("\nwrote fig6_estimation_error, fig7_estimation_consistency")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    print(f"Orbit: a = {A_KM:.0f} km, i = {INC_DEG} deg, {T_GRID[-1] / 60:.0f} min arc; "
          f"{len(network())} stations, range 10 m / range-rate 1 cm/s")
    data = run_or_load(args.runs, args.force)
    summary(data)
    diagnose_gap()
    figures(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
