# Orbital Analysis

Flight dynamics, state estimation, and surrogate-accelerated conjunction
assessment.

Two things are built here. The first is a conjunction-assessment pipeline:
TLE ingestion, SGP4 propagation, a calibrated uncertainty model, three
independent collision-probability estimators, a Gaussian-process surrogate
trained at space-filling design points, and a risk-triage classifier. The
second is a 6-DOF rigid-body propagator (done) and an EKF/UKF estimation
stack built on top of it (in progress).

A short report covering the conjunction work, including its negative
results, is in [`writeup/report.tex`](writeup/report.tex).

## Layout

```
src/orbital/
  conventions.py      frames, time systems, units -- the single source of truth
  paths.py            project-root resolution
  core/               Earth constants (EGM96) for the 6-DOF dynamics
  attitude/           quaternions, direction cosine matrices, Euler angles
  dynamics/           6-DOF equations of motion, force and torque models
  integrators/        fixed-step RK4 and adaptive DOP853 behind one interface
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

## 6-DOF rigid-body dynamics

The state is 13 numbers: position `r` and velocity `v` of the centre of mass
(km, km/s, J2000 ECI), the attitude quaternion `q` (scalar first, rotating
body axes into ECI), and the body angular velocity `ω` (rad/s, body axes).

**Translation.** Newton's law per unit mass, summing the force models:

$$\dot{\mathbf r} = \mathbf v, \qquad
\dot{\mathbf v} = -\frac{\mu}{r^3}\mathbf r + \mathbf a_{J_2} + \dots$$

with the oblateness perturbation

$$\mathbf a_{J_2} = -\frac{3}{2}\frac{J_2 \mu R_E^2}{r^5}
\begin{bmatrix} x\,(1 - 5z^2/r^2) \\ y\,(1 - 5z^2/r^2) \\ z\,(3 - 5z^2/r^2) \end{bmatrix}.$$

**Rotation.** Quaternion kinematics and Euler's equations in body axes:

$$\dot q = \tfrac12\, q \otimes \begin{bmatrix} 0 \\ \boldsymbol\omega \end{bmatrix},
\qquad
I\dot{\boldsymbol\omega} = \boldsymbol\tau - \boldsymbol\omega \times I\boldsymbol\omega,$$

with the gravity-gradient torque, where `û` is the unit radius vector in body axes:

$$\boldsymbol\tau_{gg} = \frac{3\mu}{r^3}\,\hat{\mathbf u} \times I\hat{\mathbf u}.$$

**Quaternion norm drift.** The exact flow keeps `|q| = 1` (since
`q · q̇ = 0`), but a discrete integrator does not. Instead of hiding this
inside the equations, each integrator takes a `Constraint`. RK4 projects `q`
back to unit length after any step that pushes the error past the threshold.
DOP853 watches the error with a terminal `solve_ivp` event, then projects and
restarts. Both report how often they projected, so the drift stays visible.

**Validation** (`tests/test_dynamics.py`, curves from
`scripts/validate_6dof.py`). Every check compares against physics, not
stored output:

| Case | Reference | Result |
|---|---|---|
| Torque-free tumble, 3000 s | energy and inertial **H** vector conserved | DOP853: 2e-13, 6e-12 |
| RK4 step halving | fourth-order convergence | drift falls by more than 16x |
| Axisymmetric body | closed-form precession, `φ̇ = \|H\|/I_t` | DOP853 attitude error 2e-11 rad |
| Quaternion norm, RK4 h = 0.5 s | stays unit | 3e-7 unprojected, 2e-16 projected |
| Two-body, one period | orbit closes, energy and **h** conserved | < 1e-6 km |
| J2, 20 orbits | energy and `h_z` conserved; node regression `-3/2 n J2 (R/a)^2 cos i` | secular rate within 0.42% |
| Gravity gradient | direct sum over point masses | matches to 1e-5 |
| Pitch libration | `ω = n sqrt(3 (I_2 - I_1)/I_3)` | within 0.1% of amplitude |

At matched accuracy DOP853 is far cheaper. Over the 3000 s tumble it uses
27k right-hand-side evaluations for 2e-13 energy drift, against RK4's 48k
for 2e-9.

## Running

```bash
pytest                              # offline suite, no network required
python scripts/validate_iss.py      # propagation sanity checks
python scripts/validate_6dof.py     # 6-DOF conservation and precession curves
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
