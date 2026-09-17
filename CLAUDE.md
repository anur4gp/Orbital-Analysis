# Project: Surrogate-Accelerated Conjunction Assessment

## Setup instructions (run these first)

This file lives at the root of the `Orbital Analysis/` project folder. If the
folder structure below doesn't exist yet, create it before doing anything else:

1. From inside `Orbital Analysis/`, create the subfolders:
   ```
   mkdir -p data src tests notebooks
   ```
2. Set up a Python virtual environment scoped to this project:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install Phase 1 dependencies:
   ```
   pip install sgp4 numpy pandas matplotlib requests
   ```
   Hold off on `astropy`/`poliastro` until reference-frame work is needed later.
4. Freeze dependencies so the environment is reproducible:
   ```
   pip freeze > requirements.txt
   ```
5. Initialize git and commit the skeleton:
   ```
   git init
   git add .
   git commit -m "Initial project skeleton"
   ```
6. Add a `.gitignore` covering at least: `venv/`, `__pycache__/`, `*.pyc`,
   `data/*` (raw pulls shouldn't necessarily be versioned — decide case by
   case), `.env`.

Resulting structure:
```
Orbital Analysis/
  CLAUDE.md
  requirements.txt
  .gitignore
  data/           # cached TLEs, raw pulls
  src/
    propagation.py
    data_fetch.py
  tests/
  notebooks/      # exploration, not final code
```

## What this is

A computational aerospace portfolio project: estimating satellite collision
probability (conjunction assessment) faster than brute-force Monte Carlo by
using a space-filling-design surrogate model. This is meant to double as a
gateway project into aerospace/defense-adjacent industry work and as PhD
application portfolio material (target programs: computational/applied math
with surrogate modeling, UQ, stochastic simulation focus).

**Why it matters:** Full Monte Carlo collision-probability screening across
the whole tracked-object catalog is computationally infeasible at current
scale, and growing megaconstellations are making the problem worse. This
project builds the standard expensive baseline, then replaces most of its
cost with a surrogate trained on efficiently-chosen sample points — directly
reusing space-filling design research from a prior REU, applied to a new
domain.

## Relevant background on me (Anurag)

- Applied Mathematical Sciences undergrad, UC Merced, computational emphasis.
- RUSIS REU: space-filling Latin hypercube design via parallel tempering and
  simulated annealing — this is the core technique this project repurposes
  for sampling orbital uncertainty cheaply instead of brute-force Monte Carlo.
- Compartmental ODE modeling experience (a clinic patient-flow model, fit via
  `scipy.least_squares`) — same propagate-and-fit skill as orbit propagation,
  just a different dynamical system.
- ML classification background (quantum walk thesis work) — reused later for
  the risk-triage classifier phase (flagging high-risk conjunctions).
- Known modeling instinct worth carrying over: when a model has more free
  parameters than the data can support, fix the well-constrained parameters
  to a narrow bound rather than leaving everything free — this showed up in
  the ODE identifiability work and applies equally to surrogate/UQ fitting.

## How I like to work — apply throughout this project

- Understand each phase fully before moving to the next. Don't bundle
  multiple phases together or skip ahead unprompted — walk through steps in
  order and expect follow-up questions before advancing.
- Be concise and direct. No hedging, no filler, no over-explaining.
- Sync the repo immediately after each meaningful set of changes so it always
  reflects current state.
- Favor self-contained, dependency-light solutions (local caching over
  always-hitting-an-API, flat structure over premature abstraction).
- Favor incremental prototyping — get something runnable early, refine after.

## Project architecture (5 phases)

1. **Data & propagation** — TLE/CDM ingestion, SGP4 orbit propagation,
   validation against a known trajectory.
2. **Monte Carlo baseline** — brute-force collision probability via sampling;
   this is the expensive "ground truth" everything else is benchmarked against.
3. **Surrogate model** — space-filling design (Latin hypercube / MaxPro /
   simulated annealing) picks a small set of informative sample points; fit a
   Gaussian process or polynomial-chaos surrogate for collision probability.
4. **Risk triage classifier** — flags which conjunctions need full expensive
   treatment vs. which can be dismissed cheaply.
5. **Writeup** — benchmark plot of accuracy vs. compute cost; short report or
   arXiv-style writeup for the portfolio.

## Current status

Setup done (folders, venv on Python 3.9, deps, requirements.txt, git).
Phase 1 steps 1, 2, 4, 5 done; step 3 (Space-Track registration) still open.

- `src/tle.py` — TLE parser + checksum validation
- `src/data_fetch.py` — CelesTrak GP fetch, cached to `data/tle_cache/` with a
  2 h freshness window
- `src/propagation.py` — SGP4 wrappers, error-code checking, derived quantities
- `src/validate_iss.py` — ISS sanity check, all passing (419 km mean altitude,
  92.96 min period, empirical period from radius minima agrees to 0.05 min)
- `tests/test_tle.py` — 31 offline parser tests, all passing

Space-Track registration is now **instant** -- no manual review, contrary to
the Phase 1 step 3 note above. Account is active; credentials in `.env`.

- `src/spacetrack.py` — authenticated client, rate limiter (20/min, 200/hr,
  under the published 30/300), cached to `data/spacetrack_cache/`
- `src/probe_cdm.py` — capability probe for `cdm_public`

### `cdm_public` findings (probed 2026-09-03) — these shape Phase 2

`cdm_public` has **16 fields and no covariance matrices**. It is a screening
summary, not a full CCSDS CDM: no state vectors, no epochs, no RTN covariance.

Available: `CDM_ID`, `TCA`, `PC`, `MIN_RNG`, `EMERGENCY_REPORTABLE`,
`SAT_1_ID`/`SAT_2_ID`, names, object types, RCS size class, exclusion volumes.

Consequences:
1. The Monte Carlo baseline must **synthesize covariances** (RTN error model
   growing with time since epoch), propagating state from TLEs via SGP4. This
   is a stated modeling assumption, not a data product.
2. `PC` is 18 SDS's own published collision probability — a real benchmark to
   compare against, but *not* reproducible exactly, since it was computed from
   the covariance we don't have. Differences mix covariance-model error with
   method error; say so explicitly rather than claiming validation.
3. `PC` and `EMERGENCY_REPORTABLE` are genuine **labels for the Phase 4 risk
   triage classifier** — this is the strongest use of the feed.
4. The feed is pre-filtered to high-risk events: every sampled row had
   `EMERGENCY_REPORTABLE=Y` and `PC > 1e-4`. There are **no negative examples**,
   so a triage classifier needs low-risk cases generated from screening the
   catalog directly, not drawn from this feed.

### Phase 2 status: Monte Carlo baseline built

Covariance model = **Option B**: diagonal RTN per object,
`diag(sigma_R^2, (k_T sigma_R)^2, (k_N sigma_R)^2)`, ratios FIXED at
k_T = 10, k_N = 1.5, only the scale fitted. Fixes surrogate dimension **k = 6**.

- `src/covariance.py` — RTN basis, covariance rotation, combined relative
  covariance (each object rotated to inertial BEFORE summing)
- `src/calibrate.py` — fits sigma_R to rebuilt-vs-reported miss spread
- `src/montecarlo.py` — short-term encounter model, encounter-plane
  projection, MC Pc, independent polar-quadrature Pc, small-disk closed form
- `src/run_montecarlo.py` — the baseline run
- `tests/test_covariance.py` (22), `tests/test_montecarlo.py` (18)

Calibrated: **sigma_R = 95 m, sigma_T = 953 m, sigma_N = 143 m** — consistent
with published TLE accuracy, and derived from our own data, not assumed.

Validated: MC and quadrature agree within MC error bars (max |z| = 1.4) —
two independent methods, so the implementation is cross-checked.

**Cost result that motivates Phase 3:** at Pc ~ 4e-6, 10k and 100k draws
return ZERO hits. ~22 million draws are needed for 10% relative error,
~2.2 billion for 1%. Error falls as 1/sqrt(N), so 10x accuracy costs 100x.

### Known caveats — carry these into the writeup

1. Our Pc runs **10-100x below 18 SDS's published PC**. Expected, and the
   drivers are separable: Pc scales as HBR^2, and our hard-body radii
   (1-4 m from RCS size class) are almost certainly smaller than the
   operational values; our rebuilt miss distances also differ from theirs by
   ~1 km. Do NOT present agreement with PC as validation.
2. Calibration matches the aggregate error scale but has **negative per-event
   correlation (-0.53)** — the model captures how big TLE error is, not which
   conjunctions are worst.
3. `EXCL_VOL` is a km-scale SCREENING volume keyed to object class
   (debris 1, rocket body 3, payload 5), **not** a hard-body radius. Using it
   as HBR would inflate Pc by ~6 orders of magnitude.
4. Residual-vs-TLE-age correlation is only -0.16, so the data does **not**
   support adding a time-growth term (Option D). Staying with B is empirical.
5. One event showed a 20.8 km rebuild error — far outside plausible TLE
   error. Likely a maneuver or stale elements; identify before it
   contaminates Phase 4.

### Phase 3 status: designs built, parameter space fixed at 4-D

`reference/lhd/pt_maxpro.cpp` (pybind11 port of the RUSIS MaxPro parallel
tempering) builds and runs. Build with:

    cd reference/lhd && ../../venv/bin/python setup.py build_ext --inplace

- `src/designs.py` — extension wrapper, CSV design cache, numpy
  reimplementations of psi and phi_p, random-LHD and uniform baselines
- `src/paramspace.py` — the 4-D primary and 6-D ablation parameter spaces
- `tests/test_paramspace.py` (15)

The numpy psi matches the C++ extension to 6 decimals. Parallel tempering
beats the best of 20 random LHDs by **2.1-3.1x on psi**, at 0.2-1.4s per
design (paid once; designs are static artifacts).

### Why the surrogate is 4-D, not 6-D — verified, not assumed

The obvious reading of "space-filling design instead of Monte Carlo" does
**not** work: Pc is a rare-event indicator integral, and a 256-point design
returns zero hits just as 100k random draws do. Designs accelerate smooth
integrands. The construction that works is a surrogate over the
**encounter-parameter space**, trained on expensive MC evaluations at design
points, amortized across the catalog.

Pc depends on 6 raw parameters (mu 2, C 3, HBR 1) but only **4** after
exploiting two exact invariances: the hard-body disk is rotation symmetric,
and Pc is covariant under uniform scaling of all lengths. Verified
numerically, not asserted:

| claim | cases | max rel err |
|---|---|---|
| rotation invariance | 300 | 8.3e-13 |
| scale invariance | 300 | 9.8e-13 |
| 4-D reduction reproduces Pc | 200 | 8.4e-13 |
| Pc even in each mu component | 200 | 6.2e-16 |
| 6-D -> 4-D -> Pc round trip | 200 | 5.5e-13 |

Measured fill distance: k=4/n=64 gives 0.397; matching that in 6-D needs
**~500-900 points**, i.e. 8-14x more expensive MC runs for identical
accuracy. The 6-D space is kept as an ablation to show this empirically.

Caveat: the identity holds *given the model* — circular hard body, Gaussian
uncertainty, short-term encounter. Slow encounters (vrel < 1 km/s, currently
filtered) fall outside it.

### Phase 3 step 2: surrogate trained and benchmarked

- `src/surrogate.py` — GP with ARD squared-exponential kernel, numpy
  Cholesky, hyperparameters by marginal likelihood (scipy L-BFGS-B, bounded)
- `src/run_surrogate.py` — the accuracy/cost benchmark

**Labels come from quadrature, not Monte Carlo — this was forced.** Pc spans
12.8 orders of magnitude over the box; brute-force MC labels only 28% of
design points at 1e8 draws, and 19/64 points sit below Pc = 1e-12 where no
MC ever reaches. The quadrature is exact here and agrees with MC wherever MC
works at all. This *strengthens* the motivation: brute force is not merely
expensive for rare events, it is unusable across most of the space.

Results (RMSE in log10 Pc, 3000-point held-out test set, 5 replicates each):

| n | MaxPro | random LHD | uniform |
|---|---|---|---|
| 32 | 0.034 +/-0.001 | 0.046 +/-0.024 | 0.043 +/-0.015 |
| 64 | 0.029 +/-0.001 | 0.033 +/-0.009 | 0.045 +/-0.013 |
| 128 | 0.018 +/-0.003 | 0.022 +/-0.004 | 0.033 +/-0.014 |
| 256 | 0.006 +/-0.003 | 0.006 +/-0.001 | 0.005 +/-0.001 |

Honest reading:
1. The surrogate works: 0.029 orders of magnitude (~7% in Pc) from 64
   training points, one prediction in 8.3 us vs ~22M MC draws.
2. **MaxPro's real advantage is variance, not mean accuracy** — spread
   +/-0.001 vs +/-0.024 for random LHD at n=32, 20x more consistent. With a
   one-shot expensive budget, a random design is a lottery.
3. **The 2-3x psi advantage does NOT become a 2-3x accuracy advantage.**
   Mean RMSE gains are only 1.1-1.8x at n<=128, and vanish at n=256.
   State this plainly in the writeup.
4. At n=256 all designs converge — log10 Pc is smooth, so once the box is
   covered the design stops mattering. The interesting regime is small n.

**Methodological trap hit and fixed:** an earlier run compared ONE parallel
tempering design against ONE random draw and showed MaxPro losing at n=128
(0.94x). That was baseline sampling noise. Never compare a deterministic
method against a single random realization.

### Writeup reference sheet

`writeup/project_log.tex` + `writeup/surrogate_results.tex` — LaTeX-safe
running log: what worked, what didn't, problems hit, all measured numbers
with `\label`s for cross-referencing. Regenerate the surrogate table with
`python src/run_surrogate.py`. NOT yet compile-verified (no LaTeX toolchain
on this machine); structurally checked for brace/environment balance and
siunitx column validity.

### Phase 4 status: triage classifier built and evaluated

- `src/screening.py` — catalog conjunction screening: coarse grid, per-pair
  local minima on a rolling window, analytic linear-motion filter, parabolic
  SGP4 refinement
- `src/dataset.py`, `src/build_dataset.py` — cheap pre-Pc features, log10(Pc)
  labels, multi-catalog multi-day build
- `src/triage.py`, `src/run_triage.py` — full-recall evaluation
- `writeup/phase4_results.tex`

**Dataset:** 337,789 conjunctions from Fengyun-1C (1971 obj), Cosmos-2251
(584) and Iridium-33 (111) debris over 7 days each. 352 positives (0.104%) at
log10(Pc) > -10. Regenerate with `python src/build_dataset.py` (~11 min);
`data/triage_dataset.csv` is gitignored.

**Zero conjunctions reached the operational Pc > 1e-4 threshold.** A debris
cloud over one week contains no operationally red events, so the label
threshold is a stated choice, not inherited. Method ordering is unchanged at
1e-8 and 1e-12.

**Results** (kept = fraction passed on for expensive analysis; lower better
at equal recall):

| split | model | kept | recall | missed |
|---|---|---|---|---|
| day | miss-distance cut | 1.41% | 100.0% | 0 |
| day | logistic regression | 0.92% | 100.0% | 0 |
| day | gradient boosting | 100.0% | 100.0% | 0 |
| object | miss-distance cut | 1.15% | **97.2%** | 1 |
| object | logistic regression | 0.82% | 100.0% | 0 |
| object | gradient boosting | 100.0% | 100.0% | 0 |

1. Logistic regression beats the analyst baseline: ~a third fewer expensive
   analyses, and it holds full recall on the object split where the
   miss-distance cut misses one.
2. **Gradient boosting fails outright** — keeps 100%, and its *oracle* is
   also ~100%, so the rare positives are genuinely unrankable by it. At 0.1%
   positive rate, model capacity hurts.
3. Miss distance alone has only +0.60 rank correlation with log10(Pc);
   projected covariance depends on geometry it does not capture.

**Two traps hit and fixed:**
- Thresholds set from *in-sample* training scores made GB's operating point
  collapse and masked the baseline's recall failure. Always out-of-fold.
- Screening's coarse gate must be wide (530 km at 60 s), since objects move
  ~960 km between samples; the analytic linear-motion filter then removes 91%
  of candidates with byte-identical output (29.5s -> 7.2s per catalog-day).

**Evaluation must use full recall, not accuracy or AUC** — at 0.1% positives
a constant "no" scores 99.9%. Splits must be grouped (by day or by object),
never random, or the same pair leaks across the split.

### Phase 5 status: figures and report written

- `src/make_figures.py` — four figures, vector PDF for LaTeX + PNG.
  Expensive pieces cached to `data/figure_data.json` (gitignored);
  `--force` refreshes.
- `writeup/figures/` — fig1 motivation, fig2 surrogate accuracy,
  fig3 cost benchmark, fig4 triage operating curves
- `writeup/report.tex` — the arXiv-style short report (self-contained;
  `project_log.tex` remains the running log it draws from)

**Headline number from fig3:** charging the surrogate its full training cost
and MC its per-query cost at matched accuracy (~7% in Pc), break-even is 64
conjunctions and the surrogate is **~5,300x cheaper** over the 337,789-pair
catalog screen.

**Figure palette:** slots 1-3 of the reference categorical theme
(blue #2a78d6 / orange #eb6834 / aqua #1baf7a) — the documented
all-pairs-safe subset. Marker shape and dash pattern carry identity
alongside hue, so figures survive greyscale printing and color-vision
deficiency. Aqua is below 3:1 on white, so the relief rule applies: the
paper prints the same numbers as tables.

**Figure bugs caught by actually looking at the renders:**
1. fig1(a) first drew a design point with Pc ~ 2e-15, where *every* sample
   budget returns zero hits — no convergence visible at all. The reference
   encounter is now chosen to match the Pc of the real Phase 2 conjunction.
2. fig1(b) budget labels sat on top of the histogram bars; now staggered
   vertically above them.
3. fig2 direct labels collided at n=256 where the three series converge.
   Dropped in favor of legend + markers + dash patterns.

**LaTeX state:** four .tex files, all structurally valid (braces,
environments, refs, siunitx S-columns, citations). `report.tex` cites all 10
bibliography entries. **Still not compile-verified** — no TeX toolchain on
this machine. Overleaf compiled project_log.tex fine. To build locally:
install TinyTeX or Tectonic + the LaTeX Workshop VS Code extension, then
`tlmgr install siunitx booktabs multirow`.

**Bibliographic details in report.tex were written from memory and must be
checked before any submission.**

All five phases are complete. Modules import as flat top-level names, so
scripts add `src/` to `sys.path` rather than using a package.

## Extension roadmap (flight dynamics + estimation), started 2026-09-16

Four new phases, each stopped for review: 1 = 6-DOF dynamics, 2 = EKF/UKF
orbit determination, 3 = LHS Monte Carlo campaign runner + Docker,
4 = engineering quality. Phase 0 (package layout, Python 3.12 in
`.venv312/`, pytest, CI, behaviour lock) is done. Docker Desktop is
installed and verified (`hello-world` runs); it must be open before use.
Phase 3's LHS should reuse the MaxPro tempering code, not scipy.stats.qmc.

### New Phase 1 status: 6-DOF dynamics built

- `orbital/core/constants.py`: EGM96 constants, kept separate from SGP4's WGS-72
- `orbital/attitude/`: quaternion (Hamilton, scalar first, BODY->ECI), DCM
  (Shepherd extraction), Euler 3-2-1 and 3-1-3 with gimbal-lock convention
- `orbital/dynamics/`: state (13-vector, frame-tagged; propagate refuses
  TEME), validated inertia (triangle inequality), two-body + J2 forces,
  gravity-gradient torque, conservation diagnostics
- `orbital/integrators/`: `Integrator` protocol, RK4 and DOP853, with
  `Constraint` projection for the quaternion norm (DOP853 uses a terminal
  event + restart, since solve_ivp has no between-step hook)
- `scripts/validate_6dof.py` -> `writeup/figures/fig5_6dof_validation`
- tests: `test_attitude.py`, `test_integrators.py`, `test_dynamics.py`
  (163 total pass; ruff, mypy clean; behaviour lock still IDENTICAL)

Measured: DOP853 torque-free energy drift 2e-13, closed-form precession
error 2e-11 rad; RK4 h=0.5 s quaternion norm drifts to 3e-7 without
projection and holds at 2e-16 with it; J2 node rate within 0.42% of the
analytic value (the gap is osculating vs mean elements); gravity-gradient
libration within 0.1% of the analytic curve.

J2 assumption: the ECI z-axis is treated as Earth's spin axis, which neglects
precession and nutation (~0.3 deg since J2000).

### New Phase 2 status: EKF/UKF orbit determination built

- `core/timescales.py` (Julian date), `core/frames.py` (IAU-82 GMST,
  WGS-84 geodetic -> ECEF, `EarthRotation`: spin about ECI z, no
  precession/nutation, UT1 = UTC by default). GMST matches Vallado Ex 3-5.
- `estimation/orbit_model.py`: 6-state dynamics reusing Phase 1 forces,
  STM by variational equations with central-difference da/dr, vectorised
  `acceleration_many` path (5x speedup), white-acceleration Q
- `estimation/base.py` (shared run loop, histories), `ekf.py` (Joseph form),
  `ukf.py` (alpha=1, beta=2, kappa=0; redraw after predict)
- `estimation/measurements/`: `MeasurementModel` protocol, `GroundStation`,
  `RangeRangeRate`, `PositionFix` (linear reference case)
- `estimation/consistency.py`: NEES, NIS, chi-square averaged bounds, RMS,
  `measurement_nonlinearity` (second-order spread / noise)
- `estimation/simulation.py`: truth from the 6-DOF propagator, noisy obs,
  `monte_carlo` (same noise for every filter)
- `scripts/run_estimation.py` -> fig6_estimation_error, fig7_estimation_consistency;
  MC cached in `data/estimation_mc.npz` (gitignored), `--force` reruns
- `plotting.py`: shared style + `save()` that strips the PDF CreationDate
- tests: `test_frames.py`, `test_estimation.py`; 204 total pass, ruff/mypy
  clean, behaviour lock IDENTICAL

Result (50 runs, 2 revs, 3 stations, 10 m / 1 cm/s): precise prior (10 m,
1 cm/s) EKF = UKF, NEES 6.6 in band, 3.0 m. TLE-grade prior (1 km, 1 m/s):
EKF NEES 1019, 22.8 m; UKF NEES 8.8 (mildly optimistic), 5.8 m.

**Why, measured rather than assumed.** Two effects: (1) at the first update
the range-rate second-order spread is 3.2x its noise, which pushes NEES to
~20; (2) the dominant one: across the 73 min gap, linear PhiPPhi^T keeps
P's thinnest eigenvalue at 3.6e-15 while a 4000-point MC cloud gives
1.5e-12 (the UKF gives 1.7e-12), so the EKF is ~400x overconfident in the
direction it trusts most. RTN sigmas and the mean are right to 1%, so
per-axis 3-sigma plots look fine while NEES is 150x too high.

**Wrong hypothesis caught:** I first blamed range-rate curvature alone. An
ablation with range-only tracking still broke the EKF (NEES ~1300), at the
second pass, which is what exposed the propagation mechanism.

Test lessons: NEES at successive times shares the same runs, so
"fraction of times inside the band" is far noisier than it looks. Test the
final time with enough runs (40) instead. A 12-run test failed on sampling
noise alone.

Machine note: runs are much slower under heavy background load (Discord,
Chrome); 600 s tool timeouts were hit for that reason, not a code problem.

Not built: TEME<->J2000 (no SGP4 data enters the estimation work yet).

### Phase 1 steps, in order (after setup above is done)

1. Understand the TLE format (epoch, catalog number, six orbital elements)
   before writing code.
2. Pull TLE data from **CelesTrak** — no auth needed.
   Query pattern: `https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=tle`
   (`CATNR` = NORAD ID; `GROUP` for bulk pulls). CelesTrak only refreshes
   every 2 hours and rate-limits/blocks aggressively — cache locally, don't
   refetch on every run. Full param docs:
   `celestrak.org/NORAD/documentation/gp-data-formats.php`.
3. Register for **Space-Track.org** in parallel — needed in Phase 2 for real
   Conjunction Data Messages and higher-fidelity/historical data.
   Registration is manually reviewed and can take days, so start it now.
   The `spacetracktool` PyPI package wraps their API once approved.
4. Implement propagation with `sgp4`:
   ```python
   from sgp4.api import Satrec, jday
   sat = Satrec.twoline2rv(line1, line2)
   jd, fr = jday(2024, 4, 26, 0, 0, 0)
   error, position, velocity = sat.sgp4(jd, fr)  # TEME frame, km, km/s
   ```
   Always check `error == 0` before trusting the output.
5. Validate: propagate the ISS forward a few days, sanity-check altitude
   (~400 km) and period (~92-93 min). Stronger check: compare against a
   documented public close-approach or reentry event.

## Data sources & tools reference

- **CelesTrak** (TLEs, no auth) — celestrak.org
- **Space-Track.org** (CDMs, requires registration + approval wait)
- `sgp4` — propagation
- Later phases: `scipy` (least-squares, optimization), a space-filling design
  tool (pyDOE/SMT, or a custom simulated-annealing approach mirroring the
  RUSIS method), scikit-learn/GPyTorch (surrogate/GP), matplotlib (benchmark
  plots)

## Alternatives considered, not pursuing right now

- Pure trajectory optimization (low-thrust transfers) — the common default
  project; doesn't use the most distinctive skill (surrogate/DOE work).
  Noted as a pivot option if this project doesn't pan out.
- Aerodynamic CFD surrogate modeling — also a strong DOE fit, but needs paid
  CFD software or a steep numerical-solver build from scratch.
- Possible Phase 6 extension later: minimum-fuel collision-avoidance
  maneuver optimization once a conjunction is flagged high-risk.