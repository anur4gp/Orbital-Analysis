"""Catalog-wide conjunction screening.

Phase 4 needs low-risk conjunctions, which cdm_public does not contain: its
rows are all EMERGENCY_REPORTABLE with PC > 1e-4. Negatives have to be
manufactured by screening the catalog directly.

The screen is a three-stage sieve:

  1. Coarse propagation of every object onto a shared time grid.
  2. Per-pair local minima of separation, detected on a rolling three-step
     window so memory stays O(pairs) rather than O(pairs x timesteps).
  3. An analytic linear-motion estimate of the true minimum, which discards
     the great majority of local minima without any further propagation.
  4. Parabolic refinement of the survivors to a precise TCA.

Stage 2's threshold must account for how far objects move between samples.
At 15 km/s relative speed a 60 s grid steps 900 km, so a pair passing within
1 km can appear no closer than ~450 km at any sampled instant. Screening at
a small threshold on a coarse grid silently misses almost everything, so the
coarse gate is deliberately wide -- which leaves nearly every pair as a
candidate.

Stage 3 is what makes that affordable. Near closest approach relative motion
is very nearly rectilinear, so from the sampled relative position and
velocity the true minimum separation follows in closed form: the component
of `dr` perpendicular to `dv`. That estimate is vectorized over all
candidates at once and is accurate to well within the screening margin, so
only genuine near misses reach the expensive per-pair SGP4 refinement.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from sgp4.api import SatrecArray

from orbital.sgp4tools.propagation import _to_jd, satrec_from_tle
from orbital.sgp4tools.tle import TLE

# Widest plausible closing speed in LEO (head-on, ~7.7 km/s each).
MAX_VREL_KM_S = 16.0


@dataclass
class Conjunction:
    """One refined close approach between two catalog objects."""

    i: int
    j: int
    norad_i: int
    norad_j: int
    tca: datetime
    miss_km: float
    vrel_km_s: float
    r1: np.ndarray
    v1: np.ndarray
    r2: np.ndarray
    v2: np.ndarray


def coarse_gate_km(step_seconds: float, refine_threshold_km: float) -> float:
    """Separation gate that cannot miss an approach closer than the target.

    Between samples the pair closes by at most MAX_VREL * step, so the true
    minimum can sit up to half that below the smallest sampled separation.
    """
    return refine_threshold_km + 0.5 * MAX_VREL_KM_S * step_seconds


def propagate_catalog(tles: list[TLE], start: datetime, minutes: float,
                      step_seconds: float):
    """Propagate every object onto a shared grid. Returns (offsets, r, ok).

    `r` and `v` are (n_objects, n_times, 3) in TEME km and km/s. `ok` masks
    objects whose SGP4 calls all succeeded -- decayed objects raise error
    codes and must be dropped rather than silently producing garbage.
    """
    sats = [satrec_from_tle(t) for t in tles]
    offsets = np.arange(0.0, minutes * 60.0 + step_seconds, step_seconds)
    times = [start + timedelta(seconds=float(s)) for s in offsets]
    jd_fr = np.array([_to_jd(t) for t in times])
    jd = np.ascontiguousarray(jd_fr[:, 0])
    fr = np.ascontiguousarray(jd_fr[:, 1])

    errors, r, v = SatrecArray(sats).sgp4(jd, fr)
    ok = ~np.any(errors != 0, axis=1)
    return offsets, r, v, ok


def find_candidates(r: np.ndarray, v: np.ndarray, gate_km: float,
                    keep_threshold_km: float, step_seconds: float,
                    pair_chunk: int = 2_000_000):
    """Local minima of separation that could plausibly be genuine near misses.

    Scans the time grid keeping only three consecutive separation vectors, so
    memory is O(pairs) rather than O(pairs x timesteps). A pair can have
    several conjunctions in one window, so every local minimum is kept, not
    just the global one.

    Each surviving local minimum is then screened analytically: with relative
    position `dr` and velocity `dv`, rectilinear motion reaches its closest
    approach at `t* = -(dr.dv)/|dv|^2`, giving a minimum separation of
    `|dr + t* dv|`. Only minima whose linear estimate falls inside the keep
    threshold (plus a margin for curvature) go on to SGP4 refinement.
    """
    n, n_times, _ = r.shape
    iu, ju = np.triu_indices(n, k=1)
    n_pairs = iu.size
    # Curvature over half a sampling step is small but not zero; the margin
    # keeps the linear filter conservative rather than exact.
    margin = keep_threshold_km + 0.05 * MAX_VREL_KM_S * step_seconds

    def separations(t: int) -> np.ndarray:
        out = np.empty(n_pairs)
        for s0 in range(0, n_pairs, pair_chunk):
            e = min(s0 + pair_chunk, n_pairs)
            d = r[iu[s0:e], t, :] - r[ju[s0:e], t, :]
            out[s0:e] = np.sqrt(np.einsum("ij,ij->i", d, d))
        return out

    def linear_min(idx: np.ndarray, t: int) -> np.ndarray:
        dr = r[iu[idx], t, :] - r[ju[idx], t, :]
        dv = v[iu[idx], t, :] - v[ju[idx], t, :]
        vv = np.einsum("ij,ij->i", dv, dv)
        vv = np.where(vv > 0, vv, np.inf)
        t_star = -np.einsum("ij,ij->i", dr, dv) / vv
        closest = dr + t_star[:, None] * dv
        return np.sqrt(np.einsum("ij,ij->i", closest, closest))

    prev2 = separations(0)
    prev1 = separations(1) if n_times > 1 else prev2
    hits = []
    for t in range(2, n_times):
        curr = separations(t)
        local = np.flatnonzero((prev1 < prev2) & (prev1 <= curr) & (prev1 < gate_km))
        if local.size:
            est = linear_min(local, t - 1)
            for idx in local[est < margin]:
                hits.append((int(iu[idx]), int(ju[idx]), t - 1))
        prev2, prev1 = prev1, curr
    return hits


def _sep_sq(sat_a, sat_b, when: datetime) -> float:
    jd, fr = _to_jd(when)
    ea, ra, _ = sat_a.sgp4(jd, fr)
    eb, rb, _ = sat_b.sgp4(jd, fr)
    if ea != 0 or eb != 0:
        return np.inf
    d = np.array(ra) - np.array(rb)
    return float(d @ d)


def refine(sat_a, sat_b, t_center: datetime, half_window_s: float,
           iterations: int = 8):
    """Refine a candidate to its true TCA by parabolic interpolation.

    Near closest approach relative motion is nearly rectilinear, so squared
    separation is nearly quadratic in time and a three-point parabola locates
    the vertex directly. The bracket shrinks each iteration, reaching
    sub-millisecond timing -- necessary because at 15 km/s even one second of
    timing error is 15 km of separation error.
    """
    best_t, h = t_center, half_window_s
    for _ in range(iterations):
        t_minus = best_t - timedelta(seconds=h)
        t_plus = best_t + timedelta(seconds=h)
        f_m, f_0, f_p = (_sep_sq(sat_a, sat_b, t) for t in (t_minus, best_t, t_plus))
        if not np.isfinite(f_m + f_0 + f_p):
            return None
        denom = f_m - 2.0 * f_0 + f_p
        if denom > 0:
            shift = 0.5 * h * (f_m - f_p) / denom
            shift = float(np.clip(shift, -h, h))
            candidate = best_t + timedelta(seconds=shift)
            if _sep_sq(sat_a, sat_b, candidate) < f_0:
                best_t = candidate
        else:
            # Not bracketing a minimum; step downhill instead.
            best_t = t_minus if f_m < f_p else t_plus
        h *= 0.5

    jd, fr = _to_jd(best_t)
    ea, r1, v1 = sat_a.sgp4(jd, fr)
    eb, r2, v2 = sat_b.sgp4(jd, fr)
    if ea != 0 or eb != 0:
        return None
    r1, v1, r2, v2 = map(np.array, (r1, v1, r2, v2))
    return best_t, float(np.linalg.norm(r1 - r2)), float(np.linalg.norm(v1 - v2)), r1, v1, r2, v2


def screen(tles: list[TLE], start: datetime, minutes: float = 1440.0,
           step_seconds: float = 60.0, refine_threshold_km: float = 50.0,
           verbose: bool = True) -> list[Conjunction]:
    """Full screening pass over a catalog. Returns refined conjunctions."""
    offsets, r, v, ok = propagate_catalog(tles, start, minutes, step_seconds)
    kept = np.flatnonzero(ok)
    if verbose:
        dropped = len(tles) - kept.size
        print(f"  propagated {kept.size} objects over {minutes/60:.0f} h "
              f"at {step_seconds:.0f} s ({dropped} dropped on SGP4 errors)")

    r, v = r[kept], v[kept]
    sub_tles = [tles[i] for i in kept]
    gate = coarse_gate_km(step_seconds, refine_threshold_km)
    candidates = find_candidates(r, v, gate, refine_threshold_km, step_seconds)
    if verbose:
        n = kept.size
        print(f"  coarse gate {gate:.0f} km, linear filter -> "
              f"{len(candidates):,} candidates from {n*(n-1)//2:,} pairs")

    sats = [satrec_from_tle(t) for t in sub_tles]
    out = []
    for i, j, t_idx in candidates:
        t_center = start + timedelta(seconds=float(offsets[t_idx]))
        result = refine(sats[i], sats[j], t_center, step_seconds)
        if result is None:
            continue
        tca, miss, vrel, r1, v1, r2, v2 = result
        if miss > refine_threshold_km:
            continue
        out.append(Conjunction(
            i=i, j=j,
            norad_i=sub_tles[i].catalog_number, norad_j=sub_tles[j].catalog_number,
            tca=tca, miss_km=miss, vrel_km_s=vrel,
            r1=r1, v1=v1, r2=r2, v2=v2,
        ))
    if verbose:
        print(f"  refined -> {len(out):,} conjunctions within "
              f"{refine_threshold_km:.0f} km")
    return out
