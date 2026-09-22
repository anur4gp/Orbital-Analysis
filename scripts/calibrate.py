"""Fit the RTN covariance scale sigma_R to the rebuilt-vs-reported miss spread.

k_T and k_N stay fixed. Run: python scripts/calibrate.py
"""
from __future__ import annotations

import numpy as np

from orbital.conjunction.covariance import (
    DEFAULT_K_N,
    DEFAULT_K_T,
    RTNCovariance,
    combined_covariance,
)
from orbital.conjunction.events import (
    build_geometry,
    deduplicate,
    fetch_tles_for,
    load_events,
    tractable,
)
from orbital.sgp4tools.spacetrack import SpaceTrack

N_EVENTS = 40
N_DRAWS = 20_000
SEED = 12345
MIN_VREL_KM_S = 1.0


def geometries():
    with SpaceTrack() as st:
        events = deduplicate(tractable(load_events(st, limit=500)))[:N_EVENTS]
        ids = {i for e in events for i in e.object_ids}
        tles = fetch_tles_for(st, ids)

    out = []
    for e in events:
        g = build_geometry(e, tles)
        if g is None or g.reported_miss_km is None:
            continue
        if g.relative_velocity_km_s < MIN_VREL_KM_S:
            continue
        out.append(g)
    return out


def simulate_errors(geoms, sigma_r_km, unit_normals):
    """Predicted |rebuilt - reported| per event; unit-scale draws are rescaled by sigma_R."""
    predicted = []
    for g, z in zip(geoms, unit_normals, strict=False):
        unit_cov = combined_covariance(
            g.r1, g.v1, g.r2, g.v2,
            RTNCovariance(1.0, DEFAULT_K_T, DEFAULT_K_N),
            RTNCovariance(1.0, DEFAULT_K_T, DEFAULT_K_N),
        )
        factor = np.linalg.cholesky(unit_cov)
        errors = (z @ factor.T) * sigma_r_km

        # Reported magnitude, isotropic direction.
        directions = z[:, :3] / np.linalg.norm(z[:, :3], axis=1, keepdims=True)
        truth = directions * g.reported_miss_km
        simulated = np.linalg.norm(truth + errors, axis=1)
        predicted.append(np.median(np.abs(simulated - g.reported_miss_km)))
    return np.array(predicted)


def main() -> int:
    geoms = geometries()
    observed = np.array([abs(g.miss_error_km) for g in geoms])
    print(f"{len(geoms)} unique events (deduplicated, vrel >= {MIN_VREL_KM_S} km/s)")
    print(f"observed |rebuilt - reported| (km): "
          f"median {np.median(observed):.3f}, mean {observed.mean():.3f}, "
          f"range {observed.min():.3f}-{observed.max():.3f}")

    rng = np.random.default_rng(SEED)
    unit_normals = [rng.standard_normal((N_DRAWS, 3)) for _ in geoms]

    target = np.median(observed)
    grid = np.geomspace(0.005, 5.0, 240)
    losses = []
    for sigma_r in grid:
        predicted = simulate_errors(geoms, sigma_r, unit_normals)
        losses.append(abs(np.median(predicted) - target))
    losses = np.array(losses)
    best = grid[int(np.argmin(losses))]

    print(f"\ncalibrated sigma_R = {best:.4f} km")
    print(f"  implied sigma_T  = {best * DEFAULT_K_T:.4f} km  (k_T = {DEFAULT_K_T})")
    print(f"  implied sigma_N  = {best * DEFAULT_K_N:.4f} km  (k_N = {DEFAULT_K_N})")

    predicted = simulate_errors(geoms, best, unit_normals)
    print(f"\nmodel vs observed median |error| (km): "
          f"{np.median(predicted):.3f} vs {target:.3f}")
    print(f"per-event correlation: {np.corrcoef(predicted, observed)[0, 1]:+.3f}")

    ages = np.array([max(abs(g.tle_age1_days), abs(g.tle_age2_days)) for g in geoms])
    print(f"residual vs TLE age correlation: "
          f"{np.corrcoef(observed - predicted, ages)[0, 1]:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
