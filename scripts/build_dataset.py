"""Build the Phase 4 triage dataset by screening debris catalogs.

Screens several debris families over a multi-day window and labels every
conjunction with log10(Pc). Debris clouds are the right hunting ground: they
are dense and co-orbital, so they generate far more close approaches per
object than a general catalog.

The window is processed one day at a time. Holding 1971 objects x 7 days of
positions and velocities at 60 s cadence would need ~950 MB; a day at a time
needs ~140 MB.

Run: python scripts/build_dataset.py
"""
from __future__ import annotations

import time
from datetime import timedelta

import numpy as np

from orbital.conjunction.screening import screen
from orbital.paths import DATA_DIR
from orbital.sgp4tools.celestrak import fetch_gp
from orbital.sgp4tools.tle import parse_tle_file
from orbital.triage.features import FEATURES, build

GROUPS = ("fengyun-1c-debris", "cosmos-2251-debris", "iridium-33-debris")
DAYS = 7
REFINE_THRESHOLD_KM = 50.0
OUT = DATA_DIR / "triage_dataset.csv"


def main() -> int:
    all_x, all_y, all_meta = [], [], []
    t_start = time.perf_counter()

    for group in GROUPS:
        tles = parse_tle_file(fetch_gp(group=group))
        start = max(t.epoch for t in tles)
        print(f"\n{group}: {len(tles)} objects, epoch {start:%Y-%m-%d %H:%M} UTC")

        for day in range(DAYS):
            window_start = start + timedelta(days=day)
            t0 = time.perf_counter()
            cj = screen(tles, window_start, minutes=1440.0,
                        refine_threshold_km=REFINE_THRESHOLD_KM, verbose=False)
            x, y, _ = build(cj, tles)
            if len(y):
                all_x.append(x)
                all_y.append(y)
                all_meta.extend([(group, day, c.norad_i, c.norad_j) for c in cj])
            print(f"  day {day+1}: {len(cj):>6,} conjunctions, "
                  f"{int(np.sum(y > -10)) if len(y) else 0:>4} with log10(Pc) > -10 "
                  f"[{time.perf_counter()-t0:.0f}s]")

    x = np.vstack(all_x)
    y = np.concatenate(all_y)
    print(f"\ntotal: {len(y):,} conjunctions in {time.perf_counter()-t_start:.0f}s")
    print(f"log10(Pc): min {y.min():.1f}  median {np.median(y):.1f}  max {y.max():.1f}")
    for thr in (-6, -8, -10, -12):
        print(f"  log10(Pc) > {thr:>4}: {int(np.sum(y > thr)):>6,} "
              f"({100*np.mean(y > thr):.3f}%)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join(FEATURES + ["log10_pc", "group", "day", "norad_i", "norad_j"])
    with open(OUT, "w") as fh:
        fh.write(header + "\n")
        for row, label, meta in zip(x, y, all_meta, strict=False):
            fh.write(",".join(f"{v:.6g}" for v in row) +
                     f",{label:.6g},{meta[0]},{meta[1]},{meta[2]},{meta[3]}\n")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
