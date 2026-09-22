"""Project-root resolution and the campaign command-line entry point."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from orbital import paths


class TestProjectRoot:
    def test_finds_the_root_by_pyproject(self):
        assert (paths.PROJECT_ROOT / "pyproject.toml").is_file()
        assert paths.DATA_DIR == paths.PROJECT_ROOT / "data"

    def test_environment_override_wins(self, tmp_path, monkeypatch):
        """Needed when the package is installed outside a checkout."""
        monkeypatch.setenv("ORBITAL_PROJECT_ROOT", str(tmp_path))
        assert paths._find_project_root() == tmp_path.resolve()

    def test_override_expands_the_home_shortcut(self, monkeypatch):
        monkeypatch.setenv("ORBITAL_PROJECT_ROOT", "~/somewhere")
        assert paths._find_project_root() == (Path.home() / "somewhere").resolve()

    def test_falls_back_to_the_working_directory(self, tmp_path, monkeypatch):
        """No pyproject.toml anywhere above: an installed wheel, not a repo."""
        monkeypatch.delenv("ORBITAL_PROJECT_ROOT", raising=False)
        monkeypatch.setattr(paths, "__file__", str(tmp_path / "pkg" / "paths.py"))
        monkeypatch.chdir(tmp_path)
        assert paths._find_project_root() == Path.cwd()


class TestCampaignEntryPoint:
    """``python -m orbital.campaign`` is what the container runs."""

    def run(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "orbital.campaign", *args],
            capture_output=True, text=True, timeout=120,
            cwd=paths.PROJECT_ROOT,
        )

    def test_help_describes_the_sharding_flags(self):
        result = self.run("--help")
        assert result.returncode == 0
        for flag in ("--points", "--trials", "--shard", "--shards", "--workers", "--out"):
            assert flag in result.stdout

    def test_rejects_an_unknown_design(self):
        result = self.run("--design", "sobol")
        assert result.returncode != 0
        assert "invalid choice" in result.stderr

    @pytest.mark.slow
    def test_runs_a_tiny_campaign_and_writes_parquet(self, tmp_path):
        result = self.run("--points", "4", "--trials", "2", "--revolutions", "0.3",
                          "--workers", "1", "--out", str(tmp_path / "out"))
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "out" / "runs.parquet").is_file()
        assert (tmp_path / "out" / "config.json").is_file()
        assert "EKF" in result.stdout and "UKF" in result.stdout
