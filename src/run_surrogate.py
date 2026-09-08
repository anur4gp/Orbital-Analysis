"""Phase 3 step 2: train the surrogate and benchmark it against brute force.

Labels come from the polar quadrature, not the Monte Carlo. That is forced,
not preferred: Pc spans ~13 orders of magnitude over the parameter box, and
brute-force MC resolves under a third of design points even at 1e8 draws --
roughly 30% sit below Pc = 1e-12, which no Monte Carlo reaches. The
quadrature is exact here (it agrees with MC to within MC error bars wherever
MC works at all, max |z| = 1.4 in Phase 2), so it serves as ground truth,
and the MC stands in for the general case where no closed form exists.

Run: ./venv/bin/python src/run_surrogate.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from designs import design_path, generate_maxpro, random_lhd, uniform_sample
from montecarlo import pc_analytic
from paramspace import DIM_4D, from_unit_cube_4d, unpack_4d
from surrogate import fit_gp

N_TEST = 3000
SIZES = (32, 64, 128, 256)
SEED = 11
FLOOR = 1e-300
# Uniform and random-LHD designs are random draws, so a single realization
# says nothing. Each is repeated and reported as a mean over replicates.
N_REPLICATES = 5


def label(u: np.ndarray) -> np.ndarray:
    """log10(Pc) at unit-cube points."""
    x = from_unit_cube_4d(u)
    return np.array([np.log10(max(pc_analytic(*unpack_4d(row)), FLOOR)) for row in x])


def main() -> int:
    rng = np.random.default_rng(SEED)

    u_test = rng.random((N_TEST, DIM_4D))
    t0 = time.perf_counter()
    y_test = label(u_test)
    label_time = time.perf_counter() - t0
    print(f"test set: {N_TEST} points, log10(Pc) from {y_test.min():.1f} to {y_test.max():.1f}")
    print(f"labelling cost: {label_time/N_TEST*1e3:.2f} ms/point (quadrature)\n")

    header = f"{'n':>5}  {'design':>12}  {'RMSE':>8}  {'spread':>9}  {'reps':>4}  {'fit':>7}"
    print(header + "     (RMSE in log10 Pc = orders of magnitude)")
    print("-" * (len(header) + 10))

    results = {}
    for n in SIZES:
        for name in ("maxpro", "random_lhd", "uniform"):
            rmses, fits = [], []
            # Every design type gets the same number of replicates. Parallel
            # tempering is seeded from random_device, so repeated calls give
            # genuinely different designs -- comparing one PT run against five
            # random draws would understate the baselines' variance.
            reps = N_REPLICATES
            for rep in range(reps):
                if name == "uniform":
                    u_train = uniform_sample(n, DIM_4D, rng)
                elif name == "random_lhd":
                    u_train = random_lhd(n, DIM_4D, rng)
                else:
                    u_train, _, _ = generate_maxpro(n, DIM_4D)
                y_train = label(u_train)

                t0 = time.perf_counter()
                gp = fit_gp(u_train, y_train, seed=SEED + rep)
                fits.append(time.perf_counter() - t0)
                err = np.abs(gp.predict(u_test) - y_test)
                rmses.append(np.sqrt((err ** 2).mean()))

            rmses = np.array(rmses)
            results[(n, name)] = rmses
            spread = f"+/-{rmses.std():.3f}"
            print(f"{n:>5}  {name:>12}  {rmses.mean():>8.3f} {spread:>9}  "
                  f"{reps:>4}  {np.mean(fits):>6.2f}s")
        print()

    print("MaxPro advantage (RMSE ratio vs the alternative, >1 means MaxPro wins):")
    for n in SIZES:
        mp = results[(n, 'maxpro')].mean()
        rl = results[(n, 'random_lhd')].mean()
        un = results[(n, 'uniform')].mean()
        print(f"  n={n:>4}:  vs random LHD {rl/mp:>5.2f}x   vs uniform {un/mp:>5.2f}x")

    # Points needed by each design to reach the accuracy MaxPro hits at n=64.
    target = results[(64, 'maxpro')].mean()
    print(f"\ntarget accuracy = MaxPro at n=64  (RMSE {target:.3f} orders of magnitude)")
    for name in ("random_lhd", "uniform"):
        reached = [n for n in SIZES if results[(n, name)].mean() <= target]
        msg = f"n = {min(reached)}" if reached else f"not reached by n = {max(SIZES)}"
        print(f"  {name:>12} reaches it at {msg}")

    print("\ncost comparison, per Pc query:")
    print(f"  brute-force MC, 10% rel err at Pc~4e-6 : ~22,000,000 draws")
    print(f"  surrogate                              : one GP predict, "
          f"{_predict_cost(gp, u_test)*1e6:.1f} us/point")
    print(f"  training cost, paid once               : 64 quadrature labels")
    return 0


def _predict_cost(gp, u_test) -> float:
    t0 = time.perf_counter()
    for _ in range(5):
        gp.predict(u_test)
    return (time.perf_counter() - t0) / (5 * len(u_test))


if __name__ == "__main__":
    raise SystemExit(main())
