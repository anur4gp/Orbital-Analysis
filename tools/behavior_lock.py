"""Behaviour lock for the pre-package codebase.

Phase 0 moves every module into a package and changes every import. This
harness pins what the code *computes* so that migration can be shown not to
have altered it, rather than merely asserted not to have.

It deliberately does NOT diff script stdout. The on-disk caches are days old
and past their freshness windows, so any script touching CelesTrak or
Space-Track would refetch and produce different output run to run. Every
probe here is offline, seeded, and depends only on fixed inputs or files
already committed to the repo.

Usage:
    python tools/behavior_lock.py capture <out.json>
    python tools/behavior_lock.py compare <a.json> <b.json>
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _mod(new: str, old: str):
    """Import a module by its post-migration name, falling back to the old one.

    The same harness has to run on both sides of the migration, or the
    comparison measures the harness rather than the code.
    """
    import importlib
    try:
        return importlib.import_module(new)
    except ModuleNotFoundError:
        return importlib.import_module(old)

# A frozen ISS element set: check digits recomputed, so this is a fixture and
# not a verbatim archived record.
L1 = "1 25544U 98067A   24117.51782528  .00016717  00000-0  30074-3 0  9992"
L2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49309239448471"
EPOCH = datetime(2024, 4, 26, 12, 25, 0, tzinfo=UTC)


def _j(x):
    """Convert numpy types to plain JSON-serialisable values."""
    if isinstance(x, np.ndarray):
        return [_j(v) for v in x.tolist()]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if np.isnan(v) else (repr(v) if np.isinf(v) else v)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (list, tuple)):
        return [_j(v) for v in x]
    if isinstance(x, dict):
        return {k: _j(v) for k, v in x.items()}
    return x


def capture() -> dict:
    out: dict = {}

    # ---- tle ----
    tle = _mod("orbital.sgp4tools.tle", "tle")
    t = tle.parse_tle(L1, L2, "ISS (ZARYA)")
    out["tle"] = _j({
        "checksum_l1": tle.checksum(L1), "checksum_l2": tle.checksum(L2),
        "check_line": [tle.check_line(L1), tle.check_line(L2)],
        "decimal_point_assumed": [tle._decimal_point_assumed(s)
                                  for s in ("30074-3", "00000-0", "-11606-4")],
        "catalog_number": t.catalog_number, "epoch": t.epoch.isoformat(),
        "inclination": t.inclination, "raan": t.raan, "ecc": t.eccentricity,
        "argp": t.arg_perigee, "ma": t.mean_anomaly, "mm": t.mean_motion,
        "bstar": t.bstar, "revnum": t.revolution_number,
        "period_minutes": t.period_minutes,
        "age_days": t.age_days(EPOCH),
        "n_parsed": len(tle.parse_tle_file(f"X\n{L1}\n{L2}\n{L1}\n{L2}\n")),
    })

    # ---- propagation ----
    prop = _mod("orbital.sgp4tools.propagation", "propagation")
    sat = prop.satrec_from_tle(t)
    r, v = prop.propagate(sat, EPOCH)
    times, rs, vs, errs = prop.propagate_series(sat, EPOCH, 120.0, 10.0)
    out["propagation"] = _j({
        "r": r, "v": v, "radius": prop.radius_km(r), "alt": prop.altitude_km(r),
        "period": prop.period_minutes(sat), "sma": prop.semi_major_axis_km(sat),
        "apsides": prop.apsides_km(sat),
        "series_shape": list(rs.shape), "series_r_last": rs[-1],
        "series_err_sum": int(np.sum(errs)),
    })

    # ---- covariance ----
    cov = _mod("orbital.conjunction.covariance", "covariance")
    c = cov.RTNCovariance(cov.CALIBRATED_SIGMA_R_KM)
    r2 = np.array([4000.0, 5000.0, 2000.0])
    v2 = np.array([-5.0, 3.0, 2.0])
    comb = cov.combined_covariance(r, v, r2, v2, c, c)
    rng = np.random.default_rng(12345)
    out["covariance"] = _j({
        "calibrated_sigma_r": cov.CALIBRATED_SIGMA_R_KM,
        "k_t": cov.DEFAULT_K_T, "k_n": cov.DEFAULT_K_N,
        "sigmas": [c.sigma_r_km, c.sigma_t_km, c.sigma_n_km],
        "basis": cov.rtn_basis(r, v), "rtn": c.matrix_rtn(),
        "eci": c.matrix_eci(r, v), "combined": comb,
        "sample_mean": cov.sample_relative_offsets(comb, 5000, rng).mean(axis=0),
    })

    # ---- montecarlo ----
    mc = _mod("orbital.conjunction.probability", "montecarlo")
    mu2, cov2 = mc.project_encounter(r - r2, v - v2, comb)
    rng = np.random.default_rng(999)
    est = mc.pc_monte_carlo(mu2, cov2, 0.01, 200_000, rng)
    out["montecarlo"] = _j({
        "basis": mc.encounter_plane_basis(v - v2),
        "mu2": mu2, "cov2": cov2,
        "hbr": [mc.hard_body_radius_km("SMALL", "LARGE"), mc.hard_body_radius_km("", "")],
        "pc_analytic": mc.pc_analytic(mu2, cov2, 0.01),
        "pc_small_disk": mc.pc_small_disk(mu2, cov2, 0.01),
        "log10_pc": mc.log10_pc_small_disk(mu2, cov2, 0.01),
        "mc_pc": est.pc, "mc_hits": est.n_hits,
    })

    # ---- paramspace ----
    ps = _mod("orbital.surrogate.paramspace", "paramspace")
    u4 = np.linspace(0.05, 0.95, 4).reshape(1, 4)
    u6 = np.linspace(0.05, 0.95, 6).reshape(1, 6)
    x4, x6 = ps.from_unit_cube_4d(u4)[0], ps.from_unit_cube(u6)[0]
    m4, c4, h4 = ps.unpack_4d(x4)
    m6, c6, h6 = ps.unpack(x6)
    out["paramspace"] = _j({
        "bounds_4d": ps.BOUNDS_4D, "bounds_6d": ps.BOUNDS,
        "x4": x4, "x6": x6,
        "unpack4": [m4, c4, h4], "unpack6": [m6, c6, h6],
        "reduce": ps.reduce_to_4d(m6, c6, h6),
        "roundtrip": ps.to_unit_cube(x6),
    })

    # ---- designs (cached CSVs on disk, deterministic) ----
    dz = _mod("orbital.surrogate.designs", "designs")
    rng = np.random.default_rng(7)
    d = np.loadtxt(dz.design_path("maxpro", 64, 4), delimiter=",")
    out["designs"] = _j({
        "have_extension": dz.HAVE_PT_MAXPRO,
        "maxpro_psi": dz.maxpro_criterion(d),
        "maximin_phi": dz.maximin_criterion(d),
        "min_distance": dz.min_distance(d),
        "is_lhd": dz.is_latin_hypercube(d),
        "temp_schedule": dz.temperature_schedule(6),
        "random_lhd_psi": dz.maxpro_criterion(dz.random_lhd(32, 4, rng)),
        "uniform_psi": dz.maxpro_criterion(dz.uniform_sample(32, 4, rng)),
    })

    # ---- surrogate ----
    _gp = _mod("orbital.surrogate.gp", "surrogate")
    ard_sqexp, fit_gp = _gp.ard_sqexp, _gp.fit_gp
    rng = np.random.default_rng(3)
    xs = rng.random((40, 4))
    ys = np.sin(3 * xs[:, 0]) + 2 * xs[:, 1] ** 2 - xs[:, 2]
    gp = fit_gp(xs, ys, seed=3)
    xt = rng.random((20, 4))
    mean, sd = gp.predict(xt, return_std=True)
    out["surrogate"] = _j({
        "kernel": ard_sqexp(xs[:3], xs[:3], np.array([0.0, -0.7, -0.7, -0.7, -0.7])),
        "pred_mean": mean, "pred_std": sd,
        "rmse": float(np.sqrt(((gp.predict(xs) - ys) ** 2).mean())),
    })

    # ---- conjunctions ----
    cj = _mod("orbital.conjunction.events", "conjunctions")
    row = {"CDM_ID": "1", "TCA": "2026-08-05T05:08:44.527000", "PC": "2.37e-4",
           "MIN_RNG": "181", "SAT_1_ID": "8178", "SAT_2_ID": "9999",
           "SAT_1_NAME": "A", "SAT_2_NAME": "B", "SAT1_OBJECT_TYPE": "DEBRIS",
           "SAT2_OBJECT_TYPE": "PAYLOAD", "SAT1_RCS": "SMALL", "SAT2_RCS": "LARGE",
           "SAT_1_EXCL_VOL": "1.00", "SAT_2_EXCL_VOL": "5.00"}
    ev = cj.parse_event(row)
    row2 = dict(row, CDM_ID="2", TCA="2026-08-05T05:10:00.000000")
    row3 = dict(row, CDM_ID="3", TCA="2026-08-06T09:00:00.000000")
    dedup = cj.deduplicate([ev, cj.parse_event(row2), cj.parse_event(row3)])
    out["conjunctions"] = _j({
        "tca": ev.tca.isoformat(), "pc": ev.pc, "min_rng": ev.min_rng_m,
        "ids": list(ev.object_ids), "excl": ev.combined_excl_vol_km,
        "tractable": len(cj.tractable([ev])),
        "dedup_ids": [e.cdm_id for e in dedup],
    })

    # ---- screening ----
    sc = _mod("orbital.conjunction.screening", "screening")
    out["screening"] = _j({
        "gate_60s": sc.coarse_gate_km(60.0, 50.0),
        "gate_10s": sc.coarse_gate_km(10.0, 50.0),
        "max_vrel": sc.MAX_VREL_KM_S,
    })

    # ---- dataset ----
    ds = _mod("orbital.triage.features", "dataset")
    conj = sc.Conjunction(i=0, j=1, norad_i=25544, norad_j=25544, tca=EPOCH,
                          miss_km=float(np.linalg.norm(r - r2)),
                          vrel_km_s=float(np.linalg.norm(v - v2)),
                          r1=r, v1=v, r2=r2, v2=v2)
    out["dataset"] = _j({
        "features": ds.FEATURES,
        "plane_angle": ds.orbit_plane_angle(r, v, r2, v2),
        "row": ds.features_for(conj, t, t),
        "log10_pc": ds.log10_pc_for(conj),
        "default_hbr": ds.DEFAULT_HBR_KM,
    })

    # ---- triage ----
    tg = _mod("orbital.triage.evaluation", "triage")
    res = tg.full_recall_operating_point(np.array([9., 8., 7., 6., 5.]),
                                         np.array([1, 0, 1, 0, 0]), "t")
    out["triage"] = _j({"kept": res.kept_fraction, "recall": res.recall,
                        "precision": res.precision, "reduction": res.reduction,
                        "n_kept": res.n_kept, "n_pos": res.n_positive})

    # ---- offline helpers from the network modules ----
    df = _mod("orbital.sgp4tools.celestrak", "data_fetch")
    st = _mod("orbital.sgp4tools.spacetrack", "spacetrack")
    lim = st.RateLimiter(per_minute=3, per_hour=5)
    # Paths are recorded RELATIVE TO THE REPO ROOT. Six modules derive these
    # with Path(__file__).parent.parent, so moving a module deeper in the tree
    # silently redirects its cache directory. Capturing only a filename would
    # miss that; capturing the resolved directory catches it.
    def project_root_of(mod):
        """Where this module believes the project root is.

        Before migration each module carried its own ``ROOT``, derived
        positionally. After migration they share ``orbital.paths``. The probe
        compares the resolved location, which is the behaviour that matters,
        rather than the name of the attribute holding it.
        """
        own = getattr(mod, "ROOT", None)
        if own is not None:
            return own
        from orbital.paths import PROJECT_ROOT
        return PROJECT_ROOT

    def rel(path) -> str:
        try:
            return str(Path(path).resolve().relative_to(ROOT))
        except ValueError:
            return f"OUTSIDE_REPO:{path}"

    out["paths"] = _j({
        "celestrak_cache": rel(df.CACHE_DIR),
        "spacetrack_root": rel(project_root_of(st)),
        "spacetrack_env": rel(st.ENV_PATH),
        "spacetrack_cache": rel(st.CACHE_DIR),
        "designs_root": rel(project_root_of(dz)),
        "designs_lhd": rel(dz.LHD_DIR),
        "designs_dir": rel(dz.DESIGN_DIR),
        "celestrak_cache_file": rel(df._cache_path("catnr", 25544)),
        "design_file": rel(dz.design_path("maxpro", 64, 4)),
    })

    out["fetchers"] = _j({
        "gp_url": df.GP_URL, "max_age": df.DEFAULT_MAX_AGE_HOURS,
        "cache_name": df._cache_path("catnr", 25544).name,
        "st_query_url": st.QUERY_URL,
        "st_limits": [st.MAX_PER_MINUTE, st.MAX_PER_HOUR],
        "st_path": st.SpaceTrack()._build_path("gp", {"NORAD_CAT_ID": [1, 2]}, "json"),
        "limiter": [lim.per_minute, lim.per_hour],
    })
    return out


def compare(a: dict, b: dict, tol: float = 1e-9) -> int:
    """Diff two captures. Returns the number of differing leaves."""
    diffs: list[str] = []

    def walk(x, y, path=""):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                if k not in x or k not in y:
                    diffs.append(f"{path}/{k}: present in only one capture")
                else:
                    walk(x[k], y[k], f"{path}/{k}")
        elif isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                diffs.append(f"{path}: length {len(x)} vs {len(y)}")
            else:
                for i, (p, q) in enumerate(zip(x, y, strict=False)):
                    walk(p, q, f"{path}[{i}]")
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)) \
                and not isinstance(x, bool) and not isinstance(y, bool):
            scale = max(abs(float(x)), abs(float(y)), 1.0)
            if abs(float(x) - float(y)) / scale > tol:
                diffs.append(f"{path}: {x!r} vs {y!r}")
        elif x != y:
            diffs.append(f"{path}: {x!r} vs {y!r}")

    walk(a, b)
    for d in diffs[:40]:
        print(f"  DIFF {d}")
    if len(diffs) > 40:
        print(f"  ... and {len(diffs) - 40} more")
    return len(diffs)


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "capture":
        data = capture()
        Path(sys.argv[2]).write_text(json.dumps(data, indent=1, sort_keys=True))
        leaves = json.dumps(data).count(":")
        print(f"captured {len(data)} module groups (~{leaves} values) -> {sys.argv[2]}")
        return 0
    if len(sys.argv) >= 4 and sys.argv[1] == "compare":
        a = json.loads(Path(sys.argv[2]).read_text())
        b = json.loads(Path(sys.argv[3]).read_text())
        n = compare(a, b)
        print("IDENTICAL" if n == 0 else f"{n} DIFFERENCES")
        return 1 if n else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
