"""Persistence: parquet rows plus the config that produced them.

A campaign directory holds

    runs.parquet      one row per (design point, filter), with the physical
                      parameter values and the config fingerprint on every row
    config.json       the config, its fingerprint, and provenance

Shards write ``runs.shard<i>of<n>.parquet`` into the same directory and are
merged afterwards, which is what lets the same image run as a batch array job.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from orbital.campaign.config import CampaignConfig, Provenance

CONFIG_NAME = "config.json"
RUNS_NAME = "runs.parquet"


def shard_name(shard: int, shards: int) -> str:
    """File name for one shard's rows."""
    return RUNS_NAME if shards == 1 else f"runs.shard{shard}of{shards}.parquet"


def write_runs(
    rows: pd.DataFrame,
    config: CampaignConfig,
    out_dir: Path,
    shard: int = 0,
    shards: int = 1,
    provenance: Provenance | None = None,
) -> Path:
    """Write ``rows`` as parquet and the config as JSON. Returns the parquet path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / shard_name(shard, shards)
    rows.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    payload = {
        "config": config.to_dict(),
        "fingerprint": config.fingerprint(),
        "provenance": (provenance or Provenance()).to_dict(),
        "shards": shards,
    }
    (out_dir / CONFIG_NAME).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def read_runs(out_dir: Path) -> tuple[pd.DataFrame, CampaignConfig, dict[str, Any]]:
    """Read a campaign directory: rows (all shards merged), config, metadata."""
    out_dir = Path(out_dir)
    payload = json.loads((out_dir / CONFIG_NAME).read_text())
    config = CampaignConfig.from_dict(payload["config"])
    return merge_shards(out_dir), config, payload


def merge_shards(out_dir: Path) -> pd.DataFrame:
    """Concatenate every parquet file in ``out_dir``, ordered by design point.

    Raises
    ------
    FileNotFoundError
        If the directory holds no parquet files.
    """
    paths = sorted(Path(out_dir).glob("runs*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no runs*.parquet in {out_dir}")
    frame = pd.concat([pd.read_parquet(p, engine="pyarrow") for p in paths], ignore_index=True)
    return frame.sort_values(["point", "filter"]).reset_index(drop=True)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """Per-filter summary: consistency rate and typical accuracy.

    Medians, not means: NEES spans orders of magnitude across the design, so
    a mean is dominated by its worst points.
    """
    grouped = rows.groupby("filter")
    return pd.DataFrame({
        "points": grouped.size(),
        "consistent": grouped["consistent"].mean(),
        "nees_end_median": grouped["nees_end"].median(),
        "nees_end_p90": grouped["nees_end"].quantile(0.9),
        "nis_median": grouped["nis_mean"].median(),
        "rmse_pos_m_median": grouped["rmse_pos_end_m"].median(),
        "observations_median": grouped["n_observations"].median(),
    })
