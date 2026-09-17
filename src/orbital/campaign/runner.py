"""Executing a campaign: one design point at a time, in parallel or sharded.

Determinism is the design constraint. Each point derives its own random
stream from ``(seed, index)``, so a point's rows never depend on how many
workers ran, in what order they finished, or which shard owned it. That is
what makes a local 8-core run and a 20-container batch array interchangeable
-- and it is asserted in the tests, not assumed.

Callers must be import-safe: with more than one worker the pool uses the
"spawn" start method, so child processes import the calling module. A script
that runs a campaign at module level (no ``if __name__ == "__main__"``
guard) re-runs itself in every child.
"""
from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from multiprocessing import get_context
from typing import Any

import numpy as np
import pandas as pd

from orbital.campaign.config import CampaignConfig
from orbital.campaign.design import PointSettings, design_matrix, settings_for
from orbital.core.constants import MU_EARTH_KM3_S2
from orbital.dynamics import J2Gravity, TwoBodyGravity
from orbital.estimation import EKF, UKF, OrbitModel, SequentialFilter
from orbital.estimation import consistency as cs
from orbital.estimation.simulation import (
    Scenario,
    circular_orbit_state,
    monte_carlo,
    tracking_network,
)

FORCES = (TwoBodyGravity(), J2Gravity())
_FILTER_TYPES = {"EKF": EKF, "UKF": UKF}


def available_cpus() -> int:
    """Usable CPUs: the affinity mask where the platform reports one.

    ``os.cpu_count()`` reports the host's cores, which over-counts inside a
    container restricted to fewer vCPUs. The affinity mask reflects the
    restriction on Linux; macOS has no such call, and a CPU *quota* (as
    opposed to a CPU set) is invisible either way, which is why the batch
    documentation says to pass ``--workers`` explicitly.
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def build_filters(config: CampaignConfig) -> list[SequentialFilter]:
    """Filter instances named by the config, sharing one dynamics model."""
    model = OrbitModel(FORCES)
    return [_FILTER_TYPES[name](model) for name in config.filters]


def point_scenario(config: CampaignConfig, settings: PointSettings) -> Scenario:
    """Build the tracking scenario for one design point.

    Parameters
    ----------
    config
        The campaign, supplying the orbit and arc length.
    settings
        Physical parameter values at this design point.

    Returns
    -------
    Scenario
    """
    a = config.semi_major_axis_km
    period_s = 2.0 * np.pi * np.sqrt(a**3 / MU_EARTH_KM3_S2)
    t_grid = np.arange(0.0, config.revolutions * period_s, settings.cadence_s)
    p0 = np.diag(
        [settings.sigma_r0_km**2] * 3 + [settings.sigma_v0_km_s**2] * 3
    )
    return Scenario(
        forces=FORCES,
        x0_true=circular_orbit_state(a, config.inclination_deg, config.raan_deg),
        p0=p0,
        t_s=t_grid,
        models=tracking_network(
            sigma_range_km=settings.sigma_range_km,
            sigma_range_rate_km_s=settings.sigma_range_rate_km_s,
            min_elevation_deg=settings.min_elevation_deg,
        ),
    )


def point_seed(config: CampaignConfig, index: int) -> int:
    """Stream seed for one design point: order- and worker-independent.

    Parameters
    ----------
    config
        The campaign, supplying the base seed.
    index
        Design-point index.

    Returns
    -------
    int
        Seed derived from ``(seed, index)``.
    """
    return int(np.random.SeedSequence([config.seed, index]).generate_state(1)[0])


def run_point(index: int, config: CampaignConfig) -> list[dict[str, Any]]:
    """Run every filter at one design point; one row per filter.

    Rows carry the point's physical parameters, so the parquet file is
    self-describing without joining back to the design.

    Parameters
    ----------
    index
        Design-point index.
    config
        The campaign.

    Returns
    -------
    list of dict
        One row per filter.
    """
    settings = settings_for(design_matrix(config)[index])
    scenario = point_scenario(config, settings)
    results = monte_carlo(scenario, build_filters(config), config.n_trials,
                          point_seed(config, index))

    lo, hi = cs.average_bounds(6, config.n_trials, config.confidence)
    rows = []
    for name, r in results.items():
        nees = cs.nees(r.errors, r.P).mean(axis=0)
        observed = r.nis[:, r.nis_dof > 0]
        pos = cs.rms_over_runs(r.errors[..., :3])
        vel = cs.rms_over_runs(r.errors[..., 3:])
        rows.append({
            "point": index,
            "filter": name,
            **settings.to_dict(),
            # One entry per processed observation, whatever its dimension.
            "n_observations": int((r.nis_dof > 0).sum()),
            "nees_end": float(nees[-1]),
            "nees_max": float(nees.max()),
            "nees_in_band": float(cs.fraction_inside(nees[1:], (lo, hi))),
            "consistent": bool(lo <= nees[-1] <= hi),
            "nis_mean": float(observed.mean()) if observed.size else float("nan"),
            "rmse_pos_end_m": float(pos[-1] * 1e3),
            "rmse_vel_end_mm_s": float(vel[-1] * 1e6),
            "rmse_pos_max_m": float(pos.max() * 1e3),
            "nees_band_low": lo,
            "nees_band_high": hi,
            "fingerprint": config.fingerprint(),
        })
    return rows


def shard_indices(n_points: int, shard: int, shards: int) -> list[int]:
    """Design-point indices owned by ``shard``, round-robin over ``shards``.

    Round-robin rather than contiguous blocks: cost per point varies with
    cadence and mask, so interleaving keeps shards closer in runtime.

    Parameters
    ----------
    n_points
        Design size.
    shard
        Zero-based shard index.
    shards
        Total shards.

    Returns
    -------
    list of int
        Design-point indices owned by this shard.

    Raises
    ------
    ValueError
        If ``shard`` is outside ``0..shards - 1``.
    """
    if not 0 <= shard < shards:
        raise ValueError(f"shard {shard} outside 0..{shards - 1}")
    return list(range(shard, n_points, shards))


def _worker(args: tuple[int, CampaignConfig]) -> list[dict[str, Any]]:
    return run_point(*args)


def run_campaign(
    config: CampaignConfig,
    workers: int | None = None,
    shard: int = 0,
    shards: int = 1,
    indices: Sequence[int] | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Run a campaign (or one shard of it) and return its rows.

    Parameters
    ----------
    config
        The campaign.
    workers
        Process count. ``None`` uses :func:`available_cpus`; 1 runs
        in-process, which keeps tracebacks readable while debugging.
    shard, shards
        Which slice of the design to run. Defaults run everything.
    indices
        Explicit design-point indices, overriding ``shard``/``shards``.
    progress
        Print each point as it finishes.
    """
    todo = list(indices) if indices is not None else shard_indices(
        config.n_points, shard, shards
    )
    workers = available_cpus() if workers is None else workers
    workers = max(1, min(workers, len(todo)))

    rows: list[dict[str, Any]] = []
    stream: Iterator[list[dict[str, Any]]]
    if workers == 1:
        stream = (run_point(i, config) for i in todo)
    else:
        # "spawn": fork with numpy/BLAS threads already started is unsafe on
        # macOS and deadlocks on some Linux builds.
        pool = get_context("spawn").Pool(workers)
        stream = pool.imap_unordered(_worker, [(i, config) for i in todo], chunksize=1)

    try:
        for done, point_rows in enumerate(stream, start=1):
            rows.extend(point_rows)
            if progress:
                print(f"  point {point_rows[0]['point']:>4}  "
                      f"({done}/{len(todo)})", flush=True)
    finally:
        if workers > 1:
            pool.close()
            pool.join()

    frame = pd.DataFrame(rows)
    return frame.sort_values(["point", "filter"]).reset_index(drop=True)
