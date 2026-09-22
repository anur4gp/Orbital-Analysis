"""``python -m orbital.campaign``: the container entrypoint.

``--shard`` defaults to ``AWS_BATCH_JOB_ARRAY_INDEX`` / ``BATCH_TASK_INDEX``.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from orbital.campaign.config import CampaignConfig, Provenance
from orbital.campaign.runner import run_campaign, shard_indices
from orbital.campaign.storage import summarize, write_runs
from orbital.paths import DATA_DIR

ARRAY_INDEX_VARS = ("AWS_BATCH_JOB_ARRAY_INDEX", "BATCH_TASK_INDEX")

OUT_DIR_VAR = "ORBITAL_CAMPAIGN_OUT"


def default_shard() -> int:
    """Array index from the environment, or 0 when running standalone."""
    for var in ARRAY_INDEX_VARS:
        value = os.environ.get(var)
        if value is not None and value.strip() != "":
            return int(value)
    return 0


def default_out_root() -> Path:
    """Base directory for results: ``$ORBITAL_CAMPAIGN_OUT`` or ``data/campaign``."""
    configured = os.environ.get(OUT_DIR_VAR)
    return Path(configured) if configured else DATA_DIR / "campaign"


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser for the campaign runner."""
    p = argparse.ArgumentParser(
        prog="python -m orbital.campaign",
        description="Run a Monte Carlo filter campaign over a space-filling design.",
    )
    p.add_argument("--points", type=int, default=64, help="design size")
    p.add_argument("--trials", type=int, default=8, help="Monte Carlo trials per point")
    p.add_argument("--design", default="maxpro", choices=("maxpro", "random_lhd", "uniform"))
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--revolutions", type=float, default=2.0)
    p.add_argument("--workers", type=int, default=None, help="processes (default: all cores)")
    p.add_argument("--shard", type=int, default=None,
                   help="array index (default: from the batch environment, else 0)")
    p.add_argument("--shards", type=int, default=1, help="total shards")
    p.add_argument("--out", default=None,
                   help=f"output directory (default: ${OUT_DIR_VAR} or data/campaign, "
                        "plus the campaign id)")
    return p


def main(argv: list[str] | None = None) -> int:
    """Run one campaign (or shard) and write its rows. Returns an exit code."""
    args = build_parser().parse_args(argv)
    config = CampaignConfig(
        n_points=args.points, n_trials=args.trials, design=args.design,
        seed=args.seed, revolutions=args.revolutions,
    )
    shard = default_shard() if args.shard is None else args.shard
    out_dir = default_out_root() / config.fingerprint() if args.out is None else Path(args.out)

    indices = shard_indices(config.n_points, shard, args.shards)
    print(f"campaign {config.fingerprint()}  design {config.design} "
          f"n={config.n_points} k={config.dimension}  trials={config.n_trials}")
    print(f"shard {shard + 1}/{args.shards}: {len(indices)} points -> {out_dir}")

    start = time.perf_counter()
    rows = run_campaign(config, workers=args.workers, shard=shard, shards=args.shards,
                        progress=True)
    path = write_runs(rows, config, out_dir, shard, args.shards, Provenance())
    print(f"\n{len(rows)} rows in {time.perf_counter() - start:.0f} s -> {path}")
    print(summarize(rows).to_string(float_format=lambda v: f"{v:.3g}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
