"""Run the EKF/UKF filter campaign over a 6-parameter MaxPro design and plot fig8.

Run:
    python scripts/run_campaign.py                     # 64 points, all cores
    python scripts/run_campaign.py --points 16 --trials 4
    python scripts/run_campaign.py --load data/campaign/<id>   # figures only

See docker/README.md for the containerised path.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from orbital.campaign import (
    CampaignConfig,
    Provenance,
    merge_shards,
    read_runs,
    run_campaign,
    summarize,
    write_runs,
)
from orbital.campaign.config import PARAMETERS
from orbital.paths import DATA_DIR
from orbital.plotting import BLUE, INK2, ORANGE, despine, save, style

STYLE = {"EKF": (ORANGE, "s"), "UKF": (BLUE, "o")}


def binned_median(x: np.ndarray, y: np.ndarray, edges: np.ndarray):
    """Median of ``y`` in each bin of ``x``; bins with no points are dropped."""
    idx = np.digitize(x, edges) - 1
    centres, medians = [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() >= 3:
            centres.append(np.sqrt(edges[b] * edges[b + 1]))
            medians.append(np.median(y[sel]))
    return np.array(centres), np.array(medians)


def figure(rows: pd.DataFrame, config: CampaignConfig) -> None:
    """Three views of the campaign: NEES, relative accuracy, and the map."""
    style()
    lo, hi = rows["nees_band_low"].iloc[0], rows["nees_band_high"].iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), constrained_layout=True)
    edges = np.geomspace(PARAMETERS[0].low, PARAMETERS[0].high, 7)

    ax = axes[0]
    ax.axhspan(lo, hi, color="#cfe0f5", lw=0, zorder=0.5)
    for name, (colour, marker) in STYLE.items():
        d = rows[rows["filter"] == name]
        ax.scatter(d["sigma_r0_km"], d["nees_end"], s=16, facecolor="none",
                   edgecolor=colour, linewidth=1.0, marker=marker, label=name, zorder=3)
        cx, cy = binned_median(d["sigma_r0_km"].to_numpy(), d["nees_end"].to_numpy(), edges)
        ax.plot(cx, cy, color=colour, lw=1.8, zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"initial position $\sigma_{r0}$ (km)")
    ax.set_ylabel("final NEES")
    ax.set_title("(a) consistency vs prior width", loc="left")
    ax.set_ylim(bottom=1.0)
    ax.text(PARAMETERS[0].low * 1.15, 1.35, f"{config.confidence:.0%} band",
            fontsize=7, color=INK2)
    ax.legend(loc="upper right")
    despine(ax)

    ax = axes[1]
    pivot = rows.pivot_table(index="point", columns="filter", values="rmse_pos_end_m")
    prior = rows.groupby("point")["sigma_r0_km"].first()
    ratio = (pivot["EKF"] / pivot["UKF"]).to_numpy()
    ax.axhline(1.0, color=INK2, lw=0.8, ls=(0, (1, 2)))
    ax.scatter(prior, ratio, s=16, facecolor="none", edgecolor=ORANGE, linewidth=1.0,
               marker="s", zorder=3)
    cx, cy = binned_median(prior.to_numpy(), ratio, edges)
    ax.plot(cx, cy, color=ORANGE, lw=1.8, zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"initial position $\sigma_{r0}$ (km)")
    ax.set_ylabel("EKF / UKF final position RMSE")
    ax.set_title("(b) accuracy penalty", loc="left")
    ax.set_ylim(bottom=0.45)
    ax.text(PARAMETERS[0].low * 1.15, 0.55, "equal accuracy", fontsize=7, color=INK2)
    despine(ax)

    ax = axes[2]
    ekf = rows[rows["filter"] == "EKF"]
    for consistent, colour, marker, label in (
        (True, BLUE, "o", "EKF consistent"),
        (False, ORANGE, "X", "EKF inconsistent"),
    ):
        d = ekf[ekf["consistent"] == consistent]
        ax.scatter(d["sigma_r0_km"], d["sigma_range_km"] * 1e3, s=22, marker=marker,
                   facecolor="none" if consistent else colour, edgecolor=colour,
                   linewidth=1.1, label=label, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"initial position $\sigma_{r0}$ (km)")
    ax.set_ylabel("range noise (m)")
    ax.set_title("(c) where the EKF fails", loc="left")
    ax.set_ylim(0.7, 400.0)
    ax.legend(loc="upper center", ncols=2, columnspacing=1.0, handletextpad=0.3,
              fontsize=7)
    despine(ax)

    save(fig, "fig8_campaign")
    plt.close(fig)
    print("\nwrote fig8_campaign")


def report(rows: pd.DataFrame, config: CampaignConfig) -> None:
    """Print the summary table and the transition the sweep exposes."""
    print("\n" + summarize(rows).to_string(float_format=lambda v: f"{v:.3g}"))
    edges = np.geomspace(PARAMETERS[0].low, PARAMETERS[0].high, 5)
    print(f"\nConsistency rate by initial position sigma ({config.n_trials} trials/point, "
          f"{config.confidence:.0%} band)")
    header = "".join(f"{lo:>8.3g}-{hi:<8.3g}" for lo, hi in zip(edges[:-1], edges[1:],
                                                                strict=True))
    print(f"{'sigma_r0 (km)':<16}{header}")
    for name in config.filters:
        d = rows[rows["filter"] == name]
        idx = np.digitize(d["sigma_r0_km"], edges) - 1
        cells = []
        for b in range(len(edges) - 1):
            sel = idx == b
            cells.append(f"{d['consistent'][sel].mean():>16.0%}" if sel.sum()
                         else f"{'-':>16}")
        print(f"{name:<16}" + "".join(cells))

    pivot = rows.pivot_table(index="point", columns="filter", values="rmse_pos_end_m")
    if {"EKF", "UKF"} <= set(pivot.columns):
        ratio = pivot["EKF"] / pivot["UKF"]
        worst = ratio.idxmax()
        print(f"\nEKF/UKF final position RMSE: median {ratio.median():.2f}, "
              f"90th pct {ratio.quantile(0.9):.2f}, worst {ratio.max():.1f} "
              f"at point {worst}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--design", default="maxpro",
                        choices=("maxpro", "random_lhd", "uniform"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--load", default=None, help="plot an existing campaign directory")
    parser.add_argument("--force", action="store_true", help="rerun even if results exist")
    args = parser.parse_args()

    if args.load:
        rows, config, meta = read_runs(Path(args.load))
        print(f"loaded {len(rows)} rows from {args.load} "
              f"(campaign {meta['fingerprint']}, commit {meta['provenance']['git_commit']})")
    else:
        config = CampaignConfig(n_points=args.points, n_trials=args.trials,
                                design=args.design, seed=args.seed)
        out_dir = Path(args.out) if args.out else DATA_DIR / "campaign" / config.fingerprint()
        if (out_dir / "config.json").exists() and not args.force:
            rows = merge_shards(out_dir)
            print(f"reusing {len(rows)} rows in {out_dir} (--force to rerun)")
        else:
            print(f"campaign {config.fingerprint()}: {config.n_points} points x "
                  f"{config.n_trials} trials x {len(config.filters)} filters")
            start = time.perf_counter()
            rows = run_campaign(config, workers=args.workers, progress=True)
            write_runs(rows, config, out_dir, provenance=Provenance())
            print(f"{len(rows)} rows in {time.perf_counter() - start:.0f} s -> {out_dir}")

    report(rows, config)
    figure(rows, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
