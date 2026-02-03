"""Pytest configuration."""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import pytest


def _pytest_base_dir() -> Path:
    root = os.environ.get("PYTEST_BASEDIR")
    base = Path(root) if root else Path(tempfile.gettempdir()) / "biosnn_pytest"
    return _ensure_writable_base(base)


def _ensure_writable_base(base: Path) -> Path:
    try:
        base.mkdir(parents=True, exist_ok=True)
        probe = base / "__write_probe__"
        probe.mkdir(parents=True, exist_ok=True)
        probe.rmdir()
        return base
    except OSError:
        raise RuntimeError(
            f"Pytest base directory '{base}' is not writable. "
            "Set PYTEST_BASEDIR to a writable path."
        )


def pytest_configure(config) -> None:
    """Ensure pytest uses a writable base temp directory outside the repo."""
    if config.option.basetemp is None:
        base = _pytest_base_dir() / "tmp" / uuid.uuid4().hex
        try:
            base.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            base = Path(tempfile.gettempdir()) / "biosnn_pytest" / "tmp"
            base.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(base)


@pytest.fixture
def artifact_dir() -> Path:
    base = _ensure_writable_base(_pytest_base_dir() / "artifacts")
    run_dir = base / uuid.uuid4().hex
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@pytest.fixture
def tmp_path() -> Path:
    base = _ensure_writable_base(_pytest_base_dir() / "tmp_path")
    run_dir = base / uuid.uuid4().hex
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
