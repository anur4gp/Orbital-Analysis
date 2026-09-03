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

Convention: values scaled to the unit hypercube [0,1]^d, no header row, one
sample per line. `src/` maps [0,1]^d onto the physical uncertainty ranges,
so the designs stay independent of any particular conjunction event.

## Dimensions this feeds (Phase 3)

The surrogate's input space is the uncertainty parameterization of a
conjunction — position/velocity error components for both objects. d is set
by the covariance model chosen in Phase 2, so generate designs after that is
settled, not before.
