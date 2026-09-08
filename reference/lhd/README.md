# RUSIS space-filling design code (reference)

Prior REU work, kept here as reference and as the generator for Phase 3
sample designs. Not imported by the Python code -- see "How this gets used".

## Drop files here

- `maximin_PTLHD.cpp` — maximin / Morris-Mitchell phi_p, Rcpp (7.1.2025 rev,
  the one with the convergence check; an earlier 6.30.2025 rev also exists)
- `maxpro_PTLHD.cpp`  — MaxPro criterion, Rcpp
- `pt_maxpro.cpp`     — **MaxPro ported to pybind11**, the one actually used
- `setup.py`, `test.py` — build and sanity check for the pybind11 module

Keep the originals unmodified. If a file needs project-specific changes,
copy it to a new name rather than editing in place, so the REU version stays
citable in the writeup.

## How this gets used

A space-filling design is a **static artifact**: for a given (n points,
d dimensions, criterion) it is computed once and reused forever. So the C++
does not need to be ported to Python or wrapped with pybind11.

Run the C++/R once, write the design to `designs/` as CSV, and have the
Python side load it. That keeps the project dependency-light and keeps the
expensive optimizer out of the runtime path.

    designs/maximin_n64_d6.csv     # 64 points in 6 dims, one row per point
    designs/maxpro_n64_d6.csv

## Build

    cd reference/lhd
    ../../venv/bin/python setup.py build_ext --inplace
    ../../venv/bin/python test.py          # sanity check

The `.so` and `build/` are gitignored -- they are platform-specific, rebuild
rather than commit. `src/designs.py` puts this directory on `sys.path` and
degrades to a clear error if the extension is missing.

The pybind11 signature is
`pt_maxpro_lhd(n, k, M, Nmax, Nswap, tolerance, temp_sched)`, taking the
temperature ladder from Python (the Rcpp versions derive T0 internally). It
returns a dict: `design` (n x k, **already scaled to [0,1]** as
`(level - 0.5)/n`), `measure`, `t0`, `ntotal`.

`src/designs.py` reimplements the psi and phi_p criteria in numpy; the numpy
psi agrees with the extension's to 6 decimals, which cross-checks the C++.

## Dimensions this feeds (Phase 3)

Designs are cached in `designs/` as `<name>_n<n>_k<k>.csv`, values in [0,1],
one point per row, no header. Generated at **k = 4 and k = 6**, n = 32/64/
128/256, for both `maxpro` (parallel tempering) and `random_lhd` (best of 20
unoptimized draws, the comparison baseline).

Parallel tempering beats the best of 20 random LHDs by **2.1-3.1x on psi**,
and generation costs 0.2-1.4s -- negligible, and paid once.
