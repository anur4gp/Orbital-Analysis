# Parallel-tempering Latin hypercube designs

Space-filling design code from the RUSIS REU (Beilis, Paudel, Mireles, 2025).
The originals are kept unmodified.

- `maximin_PTLHD.cpp`: maximin / Morris-Mitchell phi_p (Rcpp)
- `maxpro_PTLHD.cpp`: MaxPro criterion (Rcpp)
- `pt_maxpro.cpp`: MaxPro ported to pybind11, the version the project uses
- `setup.py`, `test.py`: build and sanity check for the pybind11 module

## Build

    cd reference/lhd
    python setup.py build_ext --inplace
    python test.py

The `.so` and `build/` are gitignored. `orbital.surrogate.designs` adds this
directory to `sys.path` and raises a clear error if the extension is missing.

Signature: `pt_maxpro_lhd(n, k, M, Nmax, Nswap, tolerance, temp_sched)`. Returns
a dict with `design` (n x k, scaled to [0, 1] as `(level - 0.5)/n`), `measure`,
`t0`, `ntotal`. The numpy psi in `orbital.surrogate.designs` matches the
extension to 6 decimals.

## Cached designs

Designs are static artifacts, cached in `designs/` as `<name>_n<n>_k<k>.csv`
(values in [0, 1], one point per row, no header), so the optimizer never runs
at runtime. Cached for k = 4 and k = 6 (n = 32, 64, 128, 256, plus smaller
sizes for k = 6).

Parallel tempering beats the best of 20 random LHDs by 2.1-3.1x on psi, at
0.2-1.4 s per design.
