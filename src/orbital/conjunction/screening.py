"""Catalog-wide conjunction screening.

1. Propagate every object onto a shared coarse grid.
2. Per-pair local minima of separation on a rolling 3-step window.
3. Analytic linear-motion miss estimate to discard most minima.
4. Parabolic SGP4 refinement of survivors to a precise TCA.

The coarse gate must be wide: at 16 km/s a 60 s step moves ~960 km.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from sgp4.api import Satrec, SatrecArray

from orbital.sgp4tools.propagation import _to_jd, satrec_from_tle
from orbital.sgp4tools.tle import TLE

# Upper bound on LEO closing speed.
MAX_VREL_KM_S = 16.0


# (TCA, miss km, vrel km/s, r1, v1, r2, v2)
RefinedApproach = tuple[datetime, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


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
    """Sampled-separation gate, km, that cannot miss an approach within ``refine_threshold_km``."""
    return refine_threshold_km + 0.5 * MAX_VREL_KM_S * step_seconds


def propagate_catalog(
    tles: list[TLE], start: datetime, minutes: float, step_seconds: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Propagate every object onto a shared grid.

    Returns
    -------
    offsets
        Seconds since ``start``, shape (n_times,).
    r, v
        TEME position (km) and velocity (km/s), shape (n_objects, n_times, 3).
    ok
        Mask of objects with no SGP4 errors.
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


def find_candidates(
    r: np.ndarray, v: np.ndarray, gate_km: float, keep_threshold_km: float,
    step_seconds: float, pair_chunk: int = 2_000_000,
) -> list[tuple[int, int, int]]:
    """Local separation minima whose linear-motion miss is within ``keep_threshold_km``.

    Memory is O(pairs): only three consecutive time steps are held. Linear
    miss is ``|dr + t* dv|`` with ``t* = -(dr.dv)/|dv|^2``.

    Returns
    -------
    list of tuple
        ``(i, j, time_index)`` per surviving minimum.
    """
    n, n_times, _ = r.shape
    iu, ju = np.triu_indices(n, k=1)
    n_pairs = iu.size
    # Margin for curvature over half a step.
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


def _sep_sq(sat_a: Satrec, sat_b: Satrec, when: datetime) -> float:
    jd, fr = _to_jd(when)
    ea, ra, _ = sat_a.sgp4(jd, fr)
    eb, rb, _ = sat_b.sgp4(jd, fr)
    if ea != 0 or eb != 0:
        return np.inf
    d = np.array(ra) - np.array(rb)
    return float(d @ d)


def refine(
    sat_a: Satrec, sat_b: Satrec, t_center: datetime, half_window_s: float,
    iterations: int = 8,
) -> RefinedApproach | None:
    """Refine a candidate TCA by iterated three-point parabolic fits.

    The bracket halves each iteration. Returns None on an SGP4 error.
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
    """Screen a catalog; returns refined approaches within ``refine_threshold_km``."""
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
