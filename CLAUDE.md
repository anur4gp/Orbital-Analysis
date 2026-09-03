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

**Next:** Phase 3. Modules import as flat top-level names, so scripts add
`src/` to `sys.path` rather than using a package.

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