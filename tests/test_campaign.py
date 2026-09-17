"""Campaign configuration, design mapping, execution and storage.

The property that matters is determinism: a point's results must not depend
on how the work was divided. That is asserted here by running the same
campaign serially, in parallel, and in shards, and requiring identical rows.

Scenarios are deliberately tiny (few points, few trials, a fraction of an
orbit) -- these test the machinery, while the physics is covered by
tests/test_estimation.py.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from orbital.campaign import (
    PARAMETERS,
    CampaignConfig,
    Provenance,
    design_matrix,
    merge_shards,
    read_runs,
    run_campaign,
    run_point,
    settings_for,
    summarize,
    write_runs,
)
from orbital.campaign.cli import build_parser, default_out_root, default_shard
from orbital.campaign.design import cached_maxpro_sizes
from orbital.campaign.runner import build_filters, point_scenario, point_seed, shard_indices
from orbital.campaign.storage import shard_name
from orbital.surrogate import designs

TINY = CampaignConfig(n_points=4, n_trials=2, revolutions=0.35)


class TestConfig:
    def test_fingerprint_is_stable_and_sensitive(self):
        assert TINY.fingerprint() == CampaignConfig(**TINY.to_dict()).fingerprint()
        assert TINY.fingerprint() != TINY.replace(seed=TINY.seed + 1).fingerprint()
        assert TINY.fingerprint() != TINY.replace(n_trials=TINY.n_trials + 1).fingerprint()

    def test_round_trip_through_json(self):
        restored = CampaignConfig.from_dict(json.loads(json.dumps(TINY.to_dict())))
        assert restored == TINY

    def test_unknown_keys_are_ignored(self):
        d = TINY.to_dict() | {"from_a_future_version": 3}
        assert CampaignConfig.from_dict(d) == TINY

    @pytest.mark.parametrize(
        "changes",
        [{"design": "sobol"}, {"n_points": 0}, {"n_trials": 0}, {"filters": ("EKF", "PF")}],
    )
    def test_rejects_invalid(self, changes):
        with pytest.raises(ValueError):
            TINY.replace(**changes)

    def test_dimension_matches_parameters(self):
        assert TINY.dimension == len(PARAMETERS) == 6

    def test_provenance_is_recorded(self):
        p = Provenance().to_dict()
        assert set(p) == {"git_commit", "python", "numpy", "scipy", "platform", "started_utc"}
        assert p["numpy"] == np.__version__


class TestParameters:
    def test_unit_endpoints_map_to_bounds(self):
        for p in PARAMETERS:
            assert p.from_unit(0.0) == pytest.approx(p.low)
            assert p.from_unit(1.0) == pytest.approx(p.high)

    def test_log_parameters_use_the_geometric_midpoint(self):
        for p in PARAMETERS:
            expected = np.sqrt(p.low * p.high) if p.log else 0.5 * (p.low + p.high)
            assert p.from_unit(0.5) == pytest.approx(expected)

    def test_log_sampling_spreads_decades_evenly(self):
        """Half the draws below the geometric mean -- the point of log scaling."""
        p = PARAMETERS[0]
        assert p.log
        values = p.from_unit(np.linspace(0, 1, 101))
        assert np.mean(values < np.sqrt(p.low * p.high)) == pytest.approx(0.5, abs=0.01)

    def test_settings_carry_every_parameter(self):
        s = settings_for(np.full(len(PARAMETERS), 0.5))
        assert set(s.to_dict()) == {p.name for p in PARAMETERS}

    def test_settings_reject_wrong_length(self):
        with pytest.raises(ValueError):
            settings_for(np.zeros(3))


class TestDesign:
    def test_maxpro_design_is_a_latin_hypercube(self):
        d = design_matrix(TINY)
        assert d.shape == (TINY.n_points, TINY.dimension)
        assert designs.is_latin_hypercube(d)

    @pytest.mark.parametrize("kind", ["maxpro", "random_lhd", "uniform"])
    def test_designs_lie_in_the_unit_cube(self, kind):
        d = design_matrix(TINY.replace(design=kind))
        assert d.shape == (TINY.n_points, TINY.dimension)
        assert np.all((d >= 0.0) & (d <= 1.0))

    def test_random_design_is_reproducible_and_seed_dependent(self):
        cfg = TINY.replace(design="random_lhd")
        assert np.array_equal(design_matrix(cfg), design_matrix(cfg))
        assert not np.array_equal(design_matrix(cfg), design_matrix(cfg.replace(seed=7)))

    def test_campaign_sizes_are_cached_as_committed_artifacts(self):
        """The container has no compiler, so every size it runs must be cached."""
        available = cached_maxpro_sizes(TINY.dimension)
        assert {4, 16, 32, 64} <= set(available)
        assert TINY.n_points in available
        assert CampaignConfig().n_points in available

    def test_uncached_size_fails_with_an_actionable_message(self):
        cfg = TINY.replace(n_points=4099)
        if 4099 in cached_maxpro_sizes(cfg.dimension):
            pytest.skip("size unexpectedly cached")
        if designs.HAVE_PT_MAXPRO:
            pytest.skip("extension present: the design would be generated")
        with pytest.raises(RuntimeError, match="Cached sizes"):
            design_matrix(cfg)

    def test_maxpro_beats_uniform_on_its_own_criterion(self):
        big = TINY.replace(n_points=32)
        maxpro = designs.maxpro_criterion(design_matrix(big))
        uniform = designs.maxpro_criterion(design_matrix(big.replace(design="uniform")))
        assert maxpro < uniform


class TestSharding:
    def test_shards_partition_the_design(self):
        for shards in (1, 3, 5, 7):
            got = [i for s in range(shards) for i in shard_indices(7, s, shards)]
            assert sorted(got) == list(range(7))

    def test_round_robin_balances_sizes(self):
        sizes = [len(shard_indices(10, s, 4)) for s in range(4)]
        assert max(sizes) - min(sizes) <= 1

    def test_rejects_out_of_range_shard(self):
        with pytest.raises(ValueError):
            shard_indices(8, 3, 3)

    def test_shard_file_names(self):
        assert shard_name(0, 1) == "runs.parquet"
        assert shard_name(2, 5) == "runs.shard2of5.parquet"

    def test_array_index_comes_from_the_batch_environment(self, monkeypatch):
        monkeypatch.delenv("AWS_BATCH_JOB_ARRAY_INDEX", raising=False)
        monkeypatch.delenv("BATCH_TASK_INDEX", raising=False)
        assert default_shard() == 0
        monkeypatch.setenv("BATCH_TASK_INDEX", "4")
        assert default_shard() == 4
        monkeypatch.setenv("AWS_BATCH_JOB_ARRAY_INDEX", "9")
        assert default_shard() == 9

    def test_output_root_follows_the_environment(self, monkeypatch, tmp_path):
        """The container sets this so a bare `docker run` writes to its mount."""
        monkeypatch.setenv("ORBITAL_CAMPAIGN_OUT", str(tmp_path))
        assert default_out_root() == tmp_path
        monkeypatch.delenv("ORBITAL_CAMPAIGN_OUT")
        assert default_out_root().name == "campaign"

    def test_cli_defaults_match_the_documented_command(self):
        args = build_parser().parse_args(["--points", "8", "--shards", "2"])
        assert (args.points, args.shards, args.shard, args.design) == (8, 2, None, "maxpro")


class TestPointSetup:
    def test_seed_depends_on_point_and_campaign(self):
        seeds = {point_seed(TINY, i) for i in range(4)}
        assert len(seeds) == 4
        assert point_seed(TINY, 0) == point_seed(TINY, 0)
        assert point_seed(TINY, 0) != point_seed(TINY.replace(seed=1), 0)

    def test_scenario_reflects_the_settings(self):
        s = settings_for(design_matrix(TINY)[0])
        sc = point_scenario(TINY, s)
        assert np.allclose(np.diff(sc.t_s), s.cadence_s)
        assert np.sqrt(sc.p0[0, 0]) == pytest.approx(s.sigma_r0_km)
        assert np.sqrt(sc.p0[3, 3]) == pytest.approx(s.sigma_v0_km_s)
        assert all(m.station.min_elevation_deg == s.min_elevation_deg for m in sc.models)
        assert all(m.sigma_range_km == s.sigma_range_km for m in sc.models)

    def test_arc_length_follows_revolutions(self):
        s = settings_for(design_matrix(TINY)[0])
        one = point_scenario(TINY.replace(revolutions=1.0), s).t_s[-1]
        two = point_scenario(TINY.replace(revolutions=2.0), s).t_s[-1]
        assert two == pytest.approx(2 * one, rel=0.01)

    def test_filters_are_named_by_the_config(self):
        names = [f.name for f in build_filters(TINY.replace(filters=("UKF",)))]
        assert names == ["UKF"]


REQUIRED_COLUMNS = {
    "point", "filter", "n_observations", "nees_end", "nees_max", "nees_in_band",
    "consistent", "nis_mean", "rmse_pos_end_m", "rmse_vel_end_mm_s", "rmse_pos_max_m",
    "nees_band_low", "nees_band_high", "fingerprint",
} | {p.name for p in PARAMETERS}


@pytest.mark.slow
class TestExecution:
    def test_one_point_yields_one_row_per_filter(self):
        rows = run_point(0, TINY)
        assert len(rows) == len(TINY.filters)
        assert REQUIRED_COLUMNS <= set(rows[0])
        assert {r["filter"] for r in rows} == set(TINY.filters)
        assert rows[0]["fingerprint"] == TINY.fingerprint()

    def test_parallel_matches_serial(self):
        """The core reproducibility claim: worker count cannot change results."""
        serial = run_campaign(TINY, workers=1)
        parallel = run_campaign(TINY, workers=4)
        pd.testing.assert_frame_equal(serial, parallel)

    def test_sharded_matches_unsharded(self):
        whole = run_campaign(TINY, workers=1)
        shards = pd.concat(
            [run_campaign(TINY, workers=1, shard=s, shards=3) for s in range(3)],
            ignore_index=True,
        ).sort_values(["point", "filter"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(whole, shards)

    def test_explicit_indices_run_only_those_points(self):
        rows = run_campaign(TINY, workers=1, indices=[2])
        assert set(rows["point"]) == {2}


@pytest.mark.slow
class TestStorage:
    def test_round_trip(self, tmp_path):
        rows = run_campaign(TINY, workers=1)
        path = write_runs(rows, TINY, tmp_path)
        assert path.name == "runs.parquet"

        back, config, meta = read_runs(tmp_path)
        pd.testing.assert_frame_equal(rows, back)
        assert config == TINY
        assert meta["fingerprint"] == TINY.fingerprint()
        assert meta["provenance"]["numpy"] == np.__version__

    def test_merge_orders_shards_like_a_whole_run(self, tmp_path):
        for shard in range(2):
            rows = run_campaign(TINY, workers=1, shard=shard, shards=2)
            write_runs(rows, TINY, tmp_path, shard, 2)
        merged = merge_shards(tmp_path)
        pd.testing.assert_frame_equal(merged, run_campaign(TINY, workers=1))

    def test_missing_results_are_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            merge_shards(tmp_path)

    def test_summary_has_one_row_per_filter(self):
        summary = summarize(run_campaign(TINY, workers=1))
        assert list(summary.index) == sorted(TINY.filters)
        assert summary.loc["EKF", "points"] == TINY.n_points
        assert 0.0 <= summary.loc["UKF", "consistent"] <= 1.0
