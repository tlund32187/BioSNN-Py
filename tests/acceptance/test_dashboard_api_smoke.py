from __future__ import annotations

import json
import socket
import time
from pathlib import Path
from typing import cast
from urllib.request import Request, urlopen

import pytest

from biosnn.runners.dashboard_server import start_dashboard_server

pytestmark = pytest.mark.acceptance


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _get_json(url: str) -> dict[str, object]:
    with urlopen(url, timeout=5) as response:  # noqa: S310 - local test server
        payload = response.read().decode("utf-8")
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise AssertionError(f"Expected JSON object, got {type(decoded)!r}")
    return cast(dict[str, object], decoded)


def _post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    request = Request(
        url=url,
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=5) as response:  # noqa: S310 - local test server
        body = response.read().decode("utf-8")
    decoded = json.loads(body)
    if not isinstance(decoded, dict):
        raise AssertionError(f"Expected JSON object, got {type(decoded)!r}")
    return cast(dict[str, object], decoded)


def _resolve_run_dir(repo_root: Path, run_dir_value: object) -> Path:
    text = str(run_dir_value or "").strip()
    if not text:
        raise AssertionError("run_dir missing from /api/run/status")
    if ":" in text[:3]:
        return Path(text)
    if text.startswith("/") and not text.startswith("/artifacts/"):
        return Path(text)
    return repo_root / text.lstrip("/")


def test_dashboard_api_can_start_run_and_emit_manifests(tmp_path: Path) -> None:
    pytest.importorskip("torch")

    repo_root = Path(__file__).resolve().parents[2]
    artifacts_dir = tmp_path / "dashboard_api_artifacts"
    port = _free_port()
    server = start_dashboard_server(
        repo_root=repo_root,
        port=port,
        artifacts_dir=artifacts_dir,
    )

    try:
        base_url = f"http://localhost:{port}"
        demos_payload = _get_json(f"{base_url}/api/demos")
        demos = demos_payload.get("demos")
        assert isinstance(demos, list)
        demo_ids = {str(entry.get("id")) for entry in demos if isinstance(entry, dict)}
        assert {
            "network",
            "propagation_impulse",
            "delay_impulse",
            "learning_gate",
            "dopamine_plasticity",
            "logic_curriculum",
            "logic_and",
            "logic_or",
            "logic_xor",
        }.issubset(demo_ids)

        run_payload = _post_json(
            f"{base_url}/api/run",
            {
                "demo_id": "propagation_impulse",
                "steps": 20,
                "device": "cpu",
                "monitor_mode": "dashboard",
            },
        )
        assert run_payload.get("state") == "running"

        deadline = time.time() + 90.0
        status_payload: dict[str, object] = {}
        while time.time() < deadline:
            status_payload = _get_json(f"{base_url}/api/run/status")
            state = str(status_payload.get("state", ""))
            if state in {"done", "error", "stopped"}:
                break
            time.sleep(0.4)

        assert status_payload.get("state") == "done", status_payload
        run_dir = _resolve_run_dir(repo_root, status_payload.get("run_dir"))
        required_files = [
            run_dir / "run_config.json",
            run_dir / "run_features.json",
            run_dir / "topology.json",
            run_dir / "neuron.csv",
            run_dir / "spikes.csv",
            run_dir / "metrics.csv",
        ]
        for path in required_files:
            assert path.exists(), f"Missing artifact: {path}"
            assert path.stat().st_size > 0, f"Empty artifact: {path}"
    finally:
        controller = getattr(server, "controller", None)
        if controller is not None and hasattr(controller, "shutdown"):
            controller.shutdown()
        server.shutdown()
        server.server_close()


def test_dashboard_api_stop_keeps_state_stopped(tmp_path: Path) -> None:
    pytest.importorskip("torch")

    repo_root = Path(__file__).resolve().parents[2]
    artifacts_dir = tmp_path / "dashboard_api_stop_artifacts"
    port = _free_port()
    server = start_dashboard_server(
        repo_root=repo_root,
        port=port,
        artifacts_dir=artifacts_dir,
    )

    try:
        base_url = f"http://localhost:{port}"
        run_payload = _post_json(
            f"{base_url}/api/run",
            {
                "demo_id": "logic_xor",
                "steps": 20000,
                "device": "cpu",
                "monitor_mode": "dashboard",
            },
        )
        run_id = str(run_payload.get("run_id", ""))
        assert run_payload.get("state") == "running"
        assert run_id

        # Wait until controller reports running (or terminal if it finished too quickly).
        deadline = time.time() + 30.0
        status_payload: dict[str, object] = {}
        while time.time() < deadline:
            status_payload = _get_json(f"{base_url}/api/run/status")
            state = str(status_payload.get("state", ""))
            if state in {"running", "done", "error", "stopped"}:
                break
            time.sleep(0.2)

        # Request stop regardless; server should never oscillate to error for this run.
        _post_json(f"{base_url}/api/run/stop", {})

        deadline = time.time() + 30.0
        final_status: dict[str, object] = {}
        while time.time() < deadline:
            final_status = _get_json(f"{base_url}/api/run/status")
            state = str(final_status.get("state", ""))
            if state in {"stopped", "done"}:
                break
            time.sleep(0.2)
        assert str(final_status.get("state", "")) in {"stopped", "done"}, final_status

        run_dir = _resolve_run_dir(repo_root, final_status.get("run_dir"))
        run_status_path = run_dir / "run_status.json"
        assert run_status_path.exists()
        status_doc = json.loads(run_status_path.read_text(encoding="utf-8"))
        assert isinstance(status_doc, dict)
        assert str(status_doc.get("state", "")) in {"stopped", "done"}, status_doc
        if str(final_status.get("state", "")) == "stopped":
            assert str(status_doc.get("state", "")) == "stopped", status_doc
    finally:
        controller = getattr(server, "controller", None)
        if controller is not None and hasattr(controller, "shutdown"):
            controller.shutdown()
        server.shutdown()
        server.server_close()
