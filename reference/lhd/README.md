# RUSIS space-filling design code (reference)

Prior REU work, kept here as reference and as the generator for Phase 3
sample designs. Not imported by the Python code -- see "How this gets used".

## Drop files here

- `maximin.cpp`   — maximin distance criterion / design
- `maxpro.cpp`    — MaxPro criterion / design
- `tempering.R`   — parallel tempering driver (+ its Rcpp import), optional

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

Both `pt_maximin_lhd(n, k, M, Nmax, Nswap, p, tolerance)` and
`pt_maxpro_lhd(n, k, M, Nmax, Nswap, tolerance)` return `Design` as an n x k
matrix of **integer levels 1..n**, column-major -- not unit-scaled. The
loader maps levels to the unit hypercube with `(level - 0.5) / n`, then onto
the physical uncertainty ranges, so designs stay independent of any
particular conjunction event.

Write CSVs as levels or as [0,1] values; note which in the filename.

## Dimensions this feeds (Phase 3)

Phase 2 settled on **Option B** (diagonal RTN covariance per object), which
fixes **k = 6**: three position-error components for each of the two objects.
Generate designs at k = 6.
