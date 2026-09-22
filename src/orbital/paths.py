"""Project filesystem locations.

The root is the nearest parent containing ``pyproject.toml``; set
``ORBITAL_PROJECT_ROOT`` to override.
"""
from __future__ import annotations

import os
from pathlib import Path


def _find_project_root() -> Path:
    override = os.environ.get("ORBITAL_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


PROJECT_ROOT: Path = _find_project_root()
DATA_DIR: Path = PROJECT_ROOT / "data"
REFERENCE_DIR: Path = PROJECT_ROOT / "reference"
WRITEUP_DIR: Path = PROJECT_ROOT / "writeup"
