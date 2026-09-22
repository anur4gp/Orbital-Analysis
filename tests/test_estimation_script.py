"""Estimation entry point: cache integrity and plotting without tracking passes."""
from __future__ import annotations

import importlib
import subprocess
import sys
from dataclasses import fields

import matplotlib.pyplot as plt
import numpy as np
import pytest

from orbital.estimation.simulation import MonteCarloResult
from orbital.paths import PROJECT_ROOT


@pytest.fixture
def experiment(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / "scripts"))
    module = importlib.import_module("run_estimation")
    monkeypatch.setattr(module, "CACHE", tmp_path / "estimation_mc.npz")
    monkeypatch.setattr(module, "T_GRID", np.array([0.0, 60.0, 120.0]))
    return module


def test_cache_round_trip_preserves_histories(experiment, monkeypatch):
    original = experiment.run_or_load(2, force=False)

    def unexpected_run(*args, **kwargs):
        pytest.fail("matching cache should avoid Monte Carlo")

    monkeypatch.setattr(experiment, "monte_carlo", unexpected_run)
    restored = experiment.run_or_load(2, force=False)
    assert restored.n_runs == 2
    for case, results in original.cases.items():
        for name, result in results.items():
            for field in fields(MonteCarloResult):
                np.testing.assert_array_equal(
                    getattr(result, field.name),
                    getattr(restored.cases[case][name], field.name),
                )


@pytest.mark.parametrize("reason", ["source", "runs", "force", "legacy"])
def test_stale_cache_is_recomputed(experiment, monkeypatch, reason):
    experiment.run_or_load(1, force=False)
    if reason == "source":
        monkeypatch.setattr(experiment, "cache_fingerprint", lambda: "changed-source")
    elif reason == "legacy":
        np.savez_compressed(experiment.CACHE, n_runs=np.array(1))
    calls = []
    original = experiment.monte_carlo

    def record_run(*args, **kwargs):
        calls.append(kwargs["seed"])
        return original(*args, **kwargs)

    monkeypatch.setattr(experiment, "monte_carlo", record_run)
    result = experiment.run_or_load(2 if reason == "runs" else 1, force=reason == "force")
    assert calls == [experiment.MC_SEED] * len(experiment.CASES)
    assert result.n_runs == (2 if reason == "runs" else 1)


def test_failed_cache_write_preserves_previous_archive(experiment, monkeypatch):
    experiment.run_or_load(1, force=False)
    previous = experiment.CACHE.read_bytes()

    def interrupted_write(path, **arrays):
        path.write_bytes(b"incomplete archive")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(experiment.np, "savez_compressed", interrupted_write)
    with pytest.raises(OSError, match="disk failure"):
        experiment.run_or_load(1, force=True)
    assert experiment.CACHE.read_bytes() == previous
    assert list(experiment.CACHE.parent.iterdir()) == [experiment.CACHE]


def test_short_arc_without_observations_can_be_reported(experiment, monkeypatch, capsys):
    data = experiment.run_or_load(1, force=False)
    saved = []
    monkeypatch.setattr(experiment, "save", lambda fig, name: saved.append(name))
    experiment.summary(data)
    experiment.diagnose_gap()
    experiment.figures(data)
    assert "fewer than two observations" in capsys.readouterr().out
    assert saved == ["fig6_estimation_error", "fig7_estimation_consistency"]
    assert not plt.get_fignums()


def test_cli_rejects_zero_runs():
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/run_estimation.py"), "--runs", "0"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 2
    assert "--runs must be positive" in result.stderr
