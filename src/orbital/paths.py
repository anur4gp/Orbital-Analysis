"""Filesystem locations, resolved once.

Before the package layout, six modules derived data directories with
``Path(__file__).resolve().parent.parent``. That expression encodes how deep
a module sits in the tree, so moving a file silently redirected its cache.
Resolving the project root once, by searching upward for ``pyproject.toml``,
removes that coupling.

Set ``ORBITAL_PROJECT_ROOT`` to override -- needed when the package is
installed outside a checkout, where no repository root exists.
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
    # Installed outside a checkout: fall back to the current directory.
    return Path.cwd()


PROJECT_ROOT: Path = _find_project_root()
DATA_DIR: Path = PROJECT_ROOT / "data"
REFERENCE_DIR: Path = PROJECT_ROOT / "reference"
WRITEUP_DIR: Path = PROJECT_ROOT / "writeup"
