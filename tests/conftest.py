"""Pytest configuration."""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import pytest


def _pytest_base_dir() -> Path:
    root = os.environ.get("PYTEST_BASEDIR")
    repo_root = Path(__file__).resolve().parents[1]
    base = Path(root) if root else repo_root / ".pytest_tmp"
    base = _ensure_writable_base(base, fallback=repo_root / ".pytest_tmp")
    return base


def _ensure_writable_base(base: Path, *, fallback: Path) -> Path:
    try:
        base.mkdir(parents=True, exist_ok=True)
        probe = base / "__write_probe__"
        probe.mkdir(parents=True, exist_ok=True)
        probe.rmdir()
        return base
    except OSError:
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


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
    if not getattr(config.option, "keep_tmpdir", False):
        config.option.keep_tmpdir = True


@pytest.fixture
def artifact_dir() -> Path:
    base = _ensure_writable_base(_pytest_base_dir() / "artifacts", fallback=_pytest_base_dir())
    run_dir = base / uuid.uuid4().hex
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@pytest.fixture
def tmp_path() -> Path:
    base = _ensure_writable_base(_pytest_base_dir() / "tmp_path", fallback=_pytest_base_dir())
    run_dir = base / uuid.uuid4().hex
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
