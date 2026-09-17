# Orbital Analysis

Flight dynamics, state estimation, and surrogate-accelerated conjunction
assessment.

Two things are built here. The first is a conjunction-assessment pipeline:
TLE ingestion, SGP4 propagation, a calibrated uncertainty model, three
independent collision-probability estimators, a Gaussian-process surrogate
trained at space-filling design points, and a risk-triage classifier. The
second is a 6-DOF rigid-body propagator and an EKF/UKF orbit-determination
stack built on top of it.

A short report covering the conjunction work, including its negative
results, is in [`writeup/report.tex`](writeup/report.tex).

## Layout

```
src/orbital/
  conventions.py      frames, time systems, units -- the single source of truth
  paths.py            project-root resolution
  core/               Earth constants, Julian dates, GMST, station coordinates
  attitude/           quaternions, direction cosine matrices, Euler angles
  dynamics/           6-DOF equations of motion, force and torque models
  integrators/        fixed-step RK4 and adaptive DOP853 behind one interface
  estimation/         EKF and UKF, measurement models, NEES/NIS, simulation
  plotting.py         shared figure style
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

## Orbit determination: EKF vs UKF

The filter state is `x = [r, v]` (km, km/s, J2000 ECI). Both filters use the
same force models as the 6-DOF propagator, which also generates the truth.

**Dynamics and STM.** `ẋ = [v, a(r)]`. The EKF also integrates the
variational equations `Φ̇ = A Φ`, where `A = [[0, I], [∂a/∂r, 0]]`. The
gradient `∂a/∂r` is computed by central differences, so a new force model
needs no hand-derived gradient. Process noise is optional white acceleration
(`Q = q [[dt³/3, dt²/2], [dt²/2, dt]] ⊗ I`).

**EKF.** `P⁻ = Φ P Φᵀ + Q`. The update uses `K = P⁻Hᵀ S⁻¹` with
`S = H P⁻ Hᵀ + R`, and the Joseph form
`P⁺ = (I−KH) P⁻ (I−KH)ᵀ + K R Kᵀ`.

**UKF.** Uses 13 scaled sigma points, each pushed through the full nonlinear
dynamics and measurement model. The defaults are α = 1, β = 2, κ = 0. With
those, all weights are non-negative, so the predicted covariance is positive
semi-definite by construction. The common α = 10⁻³ gives a centre weight of
about −10⁶.

**Measurements.** Range and range-rate from ground stations. Station
positions use WGS-84 coordinates rotated by GMST, with a 10° elevation mask.
A linear position fix is also included. It shows that adding a sensor only
means implementing the `MeasurementModel` protocol, and it serves as the case
where the EKF and UKF must agree exactly. Neither filter needs changes for a
new sensor.

**Consistency.** NEES `eᵀP⁻¹e` is χ² with 6 degrees of freedom, and NIS
`yᵀS⁻¹y` is χ² with m degrees of freedom. Both are averaged over N Monte
Carlo runs and checked against the two-sided χ²(N·dof)/N interval.

### Result (`scripts/run_estimation.py`, 50 runs)

The orbit is 7000 km at 51.6°, tracked for two revolutions by Goldstone,
Canberra and Madrid. Noise is 10 m in range and 1 cm/s in range-rate. The
only thing that changes between cases is the initial uncertainty:

| Prior (per axis) | Filter | NEES at end (95% band 5.1–7.0) | Final position RMSE |
|---|---|---|---|
| 10 m / 1 cm/s | EKF | 6.6 | 3.0 m |
| 10 m / 1 cm/s | UKF | 6.6 | 3.0 m |
| 1 km / 1 m/s (TLE grade) | EKF | **1019** | 22.8 m |
| 1 km / 1 m/s (TLE grade) | UKF | 8.8 | 5.8 m |

**Why the EKF degrades.** The script locates the failure; it isn't
guessed. There are two effects, and the second dominates:

1. **At the first update** (prior σ ≈ 1–4 km), range-rate's second-order
   term has a spread 3.2× its noise; for range the ratio is only 0.36. The
   EKF drops that term and treats the linearization error as information,
   so NEES rises from 5 to about 20. The UKF captures the term.
2. **Across the 73-minute gap after the first pass.** The pass leaves P very
   thin in some directions: its eigenvalues span 10⁻¹¹ to 10⁻¹ in mixed
   km and km/s units. Second-order dynamics move variance from the wide
   directions into the thin ones. A 4000-point Monte Carlo cloud pushed
   through the full dynamics gives a smallest eigenvalue of 1.5×10⁻¹², and
   the UKF predicts 1.7×10⁻¹². Linear propagation (ΦPΦᵀ) cannot create that
   variance, so the EKF predicts 3.6×10⁻¹⁵, about 400× too small. The
   cloud's NEES under the EKF covariance is 1104; under the UKF's it is 5.3.

The per-axis RTN sigmas and the predicted mean are right to within 1% for
both filters. The EKF is wrong **only** in the direction it is most sure
of. That is why the per-axis 3σ plot (`fig6_estimation_error`) looks fine
while NEES (`fig7_estimation_consistency`) is 150× too high. Because the
next pass is then over-trusted, the EKF ends 4× less accurate. **Checking
error against per-axis 3σ bounds is not a consistency test.**

The UKF is not perfect either: it finishes at NEES 8.8, somewhat above the
band, because second-order moment matching is itself approximate. It is
off by a small factor, not by orders of magnitude.

**Built-in checks.** The STM matches finite differences, and it is
symplectic (`ΦᵀJΦ = J`, `det Φ = 1`). Every measurement Jacobian matches
finite differences. The EKF update equals the information-form update.
EKF and UKF agree exactly for a linear measurement. GMST matches Vallado's
worked example (152.578787810°) to 10⁻⁶°.

**Frame model.** Earth rotates about the ECI z-axis by GMST, with no
precession, nutation or polar motion, and UT1 is taken as UTC. That is
self-consistent in simulation but not accurate enough for real tracking
data.

## Running

```bash
pytest                              # offline suite, no network required
python scripts/validate_iss.py      # propagation sanity checks
python scripts/validate_6dof.py     # 6-DOF conservation and precession curves
python scripts/run_estimation.py    # EKF vs UKF Monte Carlo (~2-3 min, cached)
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
