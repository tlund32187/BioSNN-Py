"""Utilities for writing per-run manifest/status artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(f"{text}\n", encoding="utf-8")
    return path


def write_run_config(run_dir: Path, config: dict[str, Any]) -> Path:
    return _write_json(run_dir / "run_config.json", dict(config))


def write_run_features(run_dir: Path, features: dict[str, Any]) -> Path:
    return _write_json(run_dir / "run_features.json", dict(features))


def write_run_status(run_dir: Path, status: dict[str, Any]) -> Path:
    return _write_json(run_dir / "run_status.json", dict(status))


__all__ = [
    "write_run_config",
    "write_run_features",
    "write_run_status",
]
