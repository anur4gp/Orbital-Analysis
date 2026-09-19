# Orbital Analysis — consolidated results (context for resume/CV writing)

Paste this whole file into a chat as background. It is a fact sheet, not prose:
every number below is measured by a script in the repository. The last two
sections matter most — they say what may and may not be claimed.

**Project:** solo computational aerospace project, September 2026.
Satellite collision-probability (conjunction) assessment accelerated with a
surrogate model, extended into a flight-dynamics and orbit-determination stack.
**Author:** Anurag Paudel, Applied Mathematical Sciences (computational), UC Merced.
**Purpose:** portfolio for aerospace modeling-and-simulation internships and for
PhD applications in computational/applied math (surrogate modeling, UQ,
stochastic simulation).
**Repo scale:** 60 modules / 6,432 lines in the package, 3,376 lines of tests,
1,678 lines of runnable scripts, 26 commits.

---

## Headline results

| Result | Number |
|---|---|
| Surrogate vs brute-force Monte Carlo, **at matched accuracy** (~7% in Pc) over a 337,789-pair catalog screen | **~5,300× cheaper**; break-even at 64 conjunctions |
| Surrogate accuracy from 64 training runs | RMSE **0.029 orders of magnitude** in log10(Pc) (~7%); one prediction in **8.3 µs** |
| Parameter-space reduction, verified numerically not asserted | **6 → 4 parameters**, exact to ~1e-12; **8–14× fewer** expensive evaluations |
| Triage classifier: catalog discarded at **100% recall** of high-risk events | **99.1% discarded** (keeps 0.92%) vs 1.41% for the analyst baseline |
| 6-DOF propagator vs closed-form torque-free precession | **2.3e-11 rad**; energy 2.3e-13, angular momentum 5.7e-12 |
| EKF from TLE-grade initial uncertainty (1 km) | **NEES 1019** (inconsistent) vs UKF 8.8; 4× worse final error |
| Filter consistency mapped over a 6-parameter sweep | EKF consistent at **28%** of design points, UKF at **72%** |
| Engineering quality | **389 tests, 96% coverage**, strict type checking, CI, containerized |

---

## Part 1 — Conjunction assessment

**Data and propagation.** TLE parser with checksum validation; SGP4 propagation
with error-code checking; CelesTrak and authenticated Space-Track clients with
local caching and a rate limiter (20/min, 200/hr, under the published 30/300).
ISS validation: 419 km mean altitude, 92.96 min period, and an empirical period
measured from radius minima agreeing to 0.05 min.

**Uncertainty model (a constraint discovered, not assumed).** The public
conjunction feed has 16 fields and **no covariance matrices**, so position
uncertainty had to be synthesized and calibrated: diagonal RTN covariance with
anisotropy ratios fixed and only the scale fitted, giving
**σ_R = 95 m, σ_T = 953 m, σ_N = 143 m** from 37 deduplicated events —
consistent with published TLE accuracy and derived from the project's own data.

**Monte Carlo baseline and the cost result that motivates everything.** Three
independent Pc estimators (Monte Carlo, polar quadrature, small-disk closed
form); MC and quadrature agree within MC error bars (max |z| = 1.4). At
Pc ≈ 4.5e-6, **10,000 and 100,000 draws return zero hits**; ~22 million draws
are needed for 10% relative error and ~2.2 billion for 1%.

**Dimensional reduction.** Two exact invariances (rotation symmetry of the
hard-body disk, scale covariance of the integral) reduce the encounter
description from 6 parameters to 4. Verified over 200–300 random cases each,
max relative error 8.3e-13. Matching the 4-D fill distance in 6-D would need
~500–900 points instead of 64.

**Surrogate.** Gaussian process, anisotropic squared-exponential kernel, plain
NumPy Cholesky, hyperparameters by marginal likelihood (L-BFGS-B). Trained at
parallel-tempering MaxPro space-filling design points — a C++/pybind11 port of
the author's prior REU research, reused here in a new domain. Labels came from
quadrature, not Monte Carlo, because **MC labels only 28% of design points even
at 1e8 draws** (19 of 64 points sit below Pc = 1e-12, where no MC ever reaches).

**Triage classifier.** Screened **337,789 conjunctions** from three debris
clouds (Fengyun-1C 1,971 objects, Cosmos-2251 584, Iridium-33 111) over 7 days
each; 352 positives (0.104%). Logistic regression keeps 0.82–0.92% of the
catalog at 100% recall against 1.15–1.41% for a miss-distance cut — and the
baseline drops to 97.2% recall on the object split. Screening optimization: an
analytic linear-motion filter removes 91% of candidates with byte-identical
output, **29.5 s → 7.2 s per catalog-day**.

---

## Part 2 — Flight dynamics and estimation

**Package migration under a behavior lock (risk control worth citing).** Before
restructuring 21 modules into an installable package and moving Python 3.9 →
3.12, a harness pinned **132 computed values across 13 module groups**; the
interpreter move and the migration were each shown **IDENTICAL** separately, so
any difference would have been attributable.

**6-DOF rigid-body propagator.** Translational dynamics plus Euler's equations,
quaternion attitude (Hamilton, scalar-first), two-body + J2 gravity,
gravity-gradient torque; fixed-step RK4 and adaptive DOP853 behind one
interface. Validation, all against identities or closed-form solutions:

| Check | Result |
|---|---|
| Torque-free energy / angular-momentum drift (DOP853, 3000 s) | 2.3e-13 / 5.7e-12 |
| Closed-form axisymmetric precession | 2.3e-11 rad |
| Quaternion norm, RK4 h = 0.5 s | 3.4e-7 unprojected → 2.2e-16 projected |
| J2 nodal regression vs analytic secular rate | within 0.42% |
| Gravity-gradient torque vs direct point-mass sum | 1e-5 |
| Small-angle pitch libration vs analytic frequency | within 0.1% |
| Cost at matched accuracy | DOP853 27,200 RHS evals vs RK4 48,000 |

**EKF vs UKF orbit determination.** Range and range-rate from three ground
stations (10 m, 1 cm/s, 10° mask); state transition matrix by variational
equations; Joseph-form covariance update; scaled sigma points. Evaluated by
chi-square NEES/NIS consistency over 50 Monte Carlo runs, not by error alone.

| Initial uncertainty | EKF | UKF |
|---|---|---|
| 10 m / 1 cm/s | NEES 6.6, 3.0 m | NEES 6.6, 3.0 m |
| 1 km / 1 m/s (TLE-grade) | **NEES 1019**, 22.8 m | NEES 8.8, 5.8 m |

The 95% acceptance band is 5.1–7.0. **The failure was located, not just
reported:** across the 73-minute gap between tracking passes, linear covariance
propagation keeps the thinnest eigenvalue at 3.6e-15 while a 4,000-point
nonlinear cloud gives 1.5e-12 and the UKF gives 1.7e-12 — **400× overconfident
in the one direction the filter trusts most**. Per-axis RTN sigmas and the mean
stay correct to 1%, so conventional per-axis 3σ plots look healthy while NEES is
150× above its band. Methodological finding: *plotting error against per-axis
3σ bounds is not a consistency test.*

**Parameter campaign at scale.** 6 parameters × 64 space-filling design points ×
8 trials × 2 filters, 1,017 s on 8 cores. EKF consistent at 28% of points
(median NEES 70.2, p90 1.4e8), UKF at 72% (median 7.0, p90 12.5). EKF
consistency decays with prior width across decades (50 / 38 / 19 / 6%) while the
UKF stays near 75%. Worst case over the design: UKF NEES 32 and 185 m; EKF 2e17
and 2,570 km. **Reproducibility engineered in:** each design point derives its
stream from (seed, index), so results are independent of worker count, shard
count and completion order — a 2-shard/2-worker container run is **bit-identical**
to a 1-worker unsharded run; cross-platform agreement ~1e-5 (different BLAS).
Results persist to Parquet with the config hash, git commit and library versions.

**Engineering quality.** 389 offline tests, 96% statement coverage (CI floor
93%), strict mypy (`disallow_untyped_defs`), ruff enforcing numpy-style
docstrings, GitHub Actions on Python 3.11 and 3.12 plus a container build.
Writing the tests found three real bugs (a frozen default argument, an
off-by-one sliding window in the rate limiter, a protocol-body error).

---

## Tech stack (keyword list)

Python 3.12, NumPy, SciPy, pandas, PyArrow/Parquet, matplotlib, scikit-learn,
C++/pybind11, Docker, GitHub Actions CI, pytest, ruff, mypy, multiprocessing,
Git, LaTeX. Documented deployment path for AWS Batch and GCP Batch array jobs.

**Domain/technique keywords:** SGP4/TLE orbit propagation, two-body and J2
dynamics, 6-DOF rigid-body dynamics, quaternion attitude kinematics, Euler's
equations, Runge-Kutta and adaptive integrators, Kalman filtering (EKF, UKF),
state transition matrices / variational equations, Monte Carlo simulation,
rare-event estimation, Gaussian process regression, design of experiments
(Latin hypercube, MaxPro, parallel tempering), uncertainty quantification,
chi-square consistency testing (NEES/NIS), imbalanced classification,
reproducible and containerized computing.

---

## Honest limits — do not let a bullet overstate these

1. **Absolute probabilities are not validated.** Computed Pc runs 10–100×
   below the operator's published values, because object size and position
   uncertainty had to be assumed. Brute force would inherit the same
   assumptions. Accuracy claims are *relative to the model*, and the ~7%
   figure is surrogate-vs-expensive-baseline, which is a fair comparison.
2. **Uncertainty is synthesized**, reproducing the aggregate scale of TLE error
   rather than per-event variation (per-event correlation is negative, −0.53).
3. **The UKF is not unconditionally better** — it also misses the acceptance
   band at ~25% of design points, by small factors rather than orders.
4. **The cloud path is documented and container-verified locally, but was never
   run on a real AWS or GCP batch service.**
5. **The two halves are not yet coupled** — no TEME↔J2000 conversion exists, so
   filter-derived covariance does not yet feed the collision probability.
6. **The triage population is debris only**, and no event in it reached the
   operational Pc > 1e-4 threshold, so the label threshold is a stated choice.
7. **Report bibliography details were written from memory** and are unverified.
8. Single-author student project; not peer-reviewed, not operationally deployed.

**Phrases to avoid:** "validated against operational data", "deployed on AWS",
"production system", "improved accuracy by 5,300×" (it is *cost*, at matched
accuracy), "real-time", "published paper".

---

## What the work actually demonstrates (framing for bullets)

- Translating research technique across domains: the space-filling design work
  from a prior REU (Beilis, Paudel & Mireles, *Optimization of Parallel
  Tempering for Latin Hypercube Designs*, RUSIS@IU Technical Report, 2025)
  applied to orbital uncertainty.
- Cost/accuracy engineering: making a comparison fair (matched accuracy, full
  training cost charged) rather than flattering.
- Diagnosing *why* a standard method fails, not just observing that it does.
- Reporting negative results: space-filling designs cannot replace samples in
  the rare-event integral; the design method's 2–3× criterion advantage does
  not become a 2–3× accuracy advantage (its real benefit is 20× lower variance);
  gradient boosting fails where logistic regression succeeds at 0.1% positives.
- Reproducible computing: determinism proven across parallel and sharded
  execution, provenance stored with results, behavior locked across a refactor.

### Two calibrated example bullets (tone reference, not final copy)

- Cut catalog-scale collision-probability screening cost ~5,300× at matched
  accuracy (~7%) by replacing brute-force Monte Carlo with a Gaussian-process
  surrogate trained at 64 parallel-tempering design points, after proving an
  exact 6→4 parameter reduction (verified to 1e-12) that cut training cost a
  further 8–14×.
- Identified and localized a failure mode in extended Kalman filter orbit
  determination — 400× covariance underestimate during tracking gaps, invisible
  to conventional per-axis 3σ checks — by building a 6-DOF propagator and
  chi-square consistency framework across a 64-point parameter sweep.
