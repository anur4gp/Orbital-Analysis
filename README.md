# Orbital Analysis

Flight dynamics, state estimation, and surrogate-accelerated conjunction
assessment.

Two things are built here. The first is a conjunction-assessment pipeline:
TLE ingestion, SGP4 propagation, a calibrated uncertainty model, three
independent collision-probability estimators, a Gaussian-process surrogate
trained at space-filling design points, and a risk-triage classifier. The
second, in progress, is a 6-DOF rigid-body propagator and an EKF/UKF
estimation stack built on top of it.

A short report covering the conjunction work, including its negative
results, is in [`writeup/report.tex`](writeup/report.tex).

## Layout

```
src/orbital/
  conventions.py      frames, time systems, units -- the single source of truth
  paths.py            project-root resolution
  sgp4tools/          TLE parsing, SGP4 propagation, CelesTrak / Space-Track
  conjunction/        covariance, collision probability, screening, cases
  surrogate/          space-filling designs, parameter spaces, the GP
  triage/             pre-Pc features and full-recall evaluation
scripts/              runnable entry points (benchmarks, dataset builds, figures)
tests/                offline pytest suite
tools/                behaviour lock used during the package migration
reference/lhd/        parallel-tempering design code (C++/pybind11)
writeup/              report, running project log, figures
```

## Install

Python 3.11 or newer.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,triage]"
```

The `campaign` extra adds pandas and pyarrow; `triage` adds scikit-learn.
The core install needs only numpy, scipy, matplotlib, sgp4 and requests.

The parallel-tempering design code is a C++ extension built separately:

```bash
cd reference/lhd && python setup.py build_ext --inplace
```

Space-Track access needs credentials in `.env` (see `.env.example`).

## Conventions

Frames, time systems, and units are defined once in
[`src/orbital/conventions.py`](src/orbital/conventions.py) and are not
restated elsewhere. The point worth knowing up front: **SGP4 outputs TEME,
while the 6-DOF dynamics work in J2000 ECI.** These differ by the equation of
the equinoxes -- hundreds of metres to kilometres in low Earth orbit -- so
states carry their frame and conversions are always an explicit call.

## Running

```bash
pytest                              # offline suite, no network required
python scripts/validate_iss.py      # propagation sanity checks
python scripts/run_montecarlo.py    # collision-probability cost baseline
python scripts/run_surrogate.py     # surrogate accuracy benchmark
python scripts/build_dataset.py     # catalog screen (~11 min)
python scripts/run_triage.py        # triage evaluation
python scripts/make_figures.py      # report figures
```

Scripts that touch CelesTrak or Space-Track cache locally; both services
rate-limit aggressively, so repeated runs are served from `data/`.

## Testing

The suite is deliberately offline and deterministic, so CI never depends on a
third-party service. Assertions are physics-based where possible -- rotation
preserves trace, determinant and eigenvalues; probability scales as the square
of the hard-body radius; two independent estimators agree within the sampler's
own error bars -- rather than comparisons against stored output.

```bash
pytest -q
ruff check .
mypy
```
