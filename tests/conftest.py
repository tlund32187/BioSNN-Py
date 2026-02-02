"""Pytest configuration."""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import pytest


def _pytest_base_dir() -> Path:
    root = os.environ.get("PYTEST_BASEDIR")
    if root:
        base = Path(root)
    else:
        local = os.environ.get("LOCALAPPDATA")
        base = Path(local) / "biosnn_pytest" if local else Path.home() / ".biosnn_pytest"
    base = _ensure_writable_base(base)
    return base


def _ensure_writable_base(base: Path) -> Path:
    try:
        base.mkdir(parents=True, exist_ok=True)
        probe = base / "__write_probe__"
        probe.mkdir(parents=True, exist_ok=True)
        probe.rmdir()
        return base
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "biosnn_pytest"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def pytest_configure(config) -> None:
    """Ensure pytest uses a writable base temp directory outside the repo."""
    if config.option.basetemp is None:
        base = _pytest_base_dir() / "tmp"
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
