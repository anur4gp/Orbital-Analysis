"""Phase 2 baseline: brute-force Monte Carlo Pc on real conjunctions.

Produces the expensive ground truth Phase 3's surrogate is benchmarked
against, and measures what that expense actually is.

Run: ./venv/bin/python src/run_montecarlo.py
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conjunctions import build_geometry, deduplicate, fetch_tles_for, load_events, tractable
from covariance import CALIBRATED_SIGMA_R_KM, RTNCovariance, combined_covariance
from montecarlo import (hard_body_radius_km, pc_analytic, pc_monte_carlo,
                        project_encounter)
from spacetrack import SpaceTrack

N_EVENTS = 10
N_DRAWS = 5_000_000
MIN_VREL_KM_S = 1.0
SEED = 2024


@dataclass
class Case:
    """One conjunction reduced to the 2D encounter-plane problem."""

    event: object
    mu_2d: np.ndarray
    cov_2d: np.ndarray
    hbr_km: float
    vrel: float


def build_cases(limit: int) -> list[Case]:
    with SpaceTrack() as st:
        events = deduplicate(tractable(load_events(st, limit=500)))
        ids = {i for e in events[: limit * 3] for i in e.object_ids}
        tles = fetch_tles_for(st, ids)

    cov_model = RTNCovariance(CALIBRATED_SIGMA_R_KM)
    cases = []
    for event in events:
        g = build_geometry(event, tles)
        if g is None or g.relative_velocity_km_s < MIN_VREL_KM_S:
            continue
        cov = combined_covariance(g.r1, g.v1, g.r2, g.v2, cov_model, cov_model)
        mu_2d, cov_2d = project_encounter(g.r1 - g.r2, g.v1 - g.v2, cov)
        cases.append(Case(
            event=event,
            mu_2d=mu_2d,
            cov_2d=cov_2d,
            hbr_km=hard_body_radius_km(event.sat1_rcs, event.sat2_rcs),
            vrel=g.relative_velocity_km_s,
        ))
        if len(cases) >= limit:
            break
    return cases


def main() -> int:
    cov_model = RTNCovariance(CALIBRATED_SIGMA_R_KM)
    print(f"sigma_R = {cov_model.sigma_r_km:.4f} km, "
          f"sigma_T = {cov_model.sigma_t_km:.4f} km, "
          f"sigma_N = {cov_model.sigma_n_km:.4f} km")
    print(f"{N_DRAWS:,} draws per event\n")

    cases = build_cases(N_EVENTS)
    rng = np.random.default_rng(SEED)

    header = (f"{'objects':>13}  {'RCS':>13}  {'HBR':>7}  {'miss':>7}  "
              f"{'Pc (MC)':>11}  {'Pc (quad)':>11}  {'Pc (18 SDS)':>11}  {'z':>5}")
    print(header)
    print("-" * len(header))

    total_time, zs = 0.0, []
    for c in cases:
        start = time.perf_counter()
        mc = pc_monte_carlo(c.mu_2d, c.cov_2d, c.hbr_km, N_DRAWS, rng)
        total_time += time.perf_counter() - start

        quad = pc_analytic(c.mu_2d, c.cov_2d, c.hbr_km)
        z = abs(mc.pc - quad) / mc.stderr if mc.stderr > 0 else 0.0
        zs.append(z)
        e = c.event
        print(f"{e.sat1_id:>6}/{e.sat2_id:<6}  {e.sat1_rcs[:6]:>6}/{e.sat2_rcs[:6]:<6}  "
              f"{c.hbr_km*1000:>6.1f}m  {np.linalg.norm(c.mu_2d):>6.3f}  "
              f"{mc.pc:>11.3e}  {quad:>11.3e}  "
              f"{(e.pc if e.pc else float('nan')):>11.3e}  {z:>5.2f}")

    print(f"\nMC vs quadrature: max |z| = {max(zs):.2f}  (z is MC error bars, so <3 is agreement)")
    print(f"MC cost: {total_time:.1f}s for {len(cases)} events, "
          f"{total_time/len(cases):.2f}s each at {N_DRAWS:,} draws")

    cost_curve(cases[0], rng)
    return 0


def cost_curve(case: Case, rng) -> None:
    """What brute force costs to resolve a Pc of this magnitude."""
    truth = pc_analytic(case.mu_2d, case.cov_2d, case.hbr_km)
    print(f"\nconvergence, one event (quadrature Pc = {truth:.4e}):")
    print(f"  {'draws':>12}  {'Pc (MC)':>11}  {'rel err':>8}  {'time':>8}")
    for n in (10_000, 100_000, 1_000_000, 10_000_000, 50_000_000):
        start = time.perf_counter()
        est = pc_monte_carlo(case.mu_2d, case.cov_2d, case.hbr_km, n, rng)
        elapsed = time.perf_counter() - start
        rel = abs(est.pc - truth) / truth if truth > 0 else float("nan")
        print(f"  {n:>12,}  {est.pc:>11.3e}  {rel:>7.1%}  {elapsed:>7.2f}s")

    for target in (0.10, 0.01):
        needed = (1 - truth) / (truth * target ** 2)
        print(f"  draws for {target:.0%} relative error: ~{needed:,.0f}")
    print("\n  MC error falls as 1/sqrt(N): a 10x tighter answer costs 100x more.")
    print("  That cost, across a whole catalog of pairs, is what Phase 3 replaces.")


if __name__ == "__main__":
    raise SystemExit(main())
