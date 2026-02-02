"""Pytest configuration."""

from __future__ import annotations

from pathlib import Path


def pytest_configure(config) -> None:
    """Ensure pytest uses a writable base temp directory."""
    if config.option.basetemp is None:
        base = Path.cwd() / ".pytest_tmp"
        base.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(base)
