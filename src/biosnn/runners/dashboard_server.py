"""Dashboard static server + local run-control API."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

from biosnn.experiments.demo_registry import (
    feature_flags_for_run_spec,
    list_demo_definitions,
    resolve_run_spec,
    run_spec_to_cli_args,
)
from biosnn.io.export.run_manifest import write_run_config, write_run_features, write_run_status


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _tail_text(text: str, *, max_lines: int = 40) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


@dataclass(slots=True)
class _RunSnapshot:
    run_id: str | None
    run_dir: str | None
    state: str
    last_error: str | None
    started_at: str | None
    finished_at: str | None
    pid: int | None

    def to_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "state": self.state,
            "last_error": self.last_error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "pid": self.pid,
        }


class DashboardRunController:
    """Thread-safe run lifecycle manager for dashboard API endpoints."""

    def __init__(
        self,
        *,
        repo_root: Path,
        artifacts_dir: Path,
        initial_run_dir: Path | None = None,
        initial_state: str = "idle",
    ) -> None:
        self._repo_root = repo_root
        self._artifacts_dir = artifacts_dir
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._run_id: str | None = None
        self._run_dir: Path | None = None
        self._state: str = initial_state
        self._last_error: str | None = None
        self._started_at: str | None = None
        self._finished_at: str | None = None
        self._stop_requested_run_ids: set[str] = set()
        if initial_run_dir is not None:
            self._run_id = initial_run_dir.name
            self._run_dir = initial_run_dir

    def demos_payload(self) -> dict[str, Any]:
        return {"demos": list_demo_definitions()}

    def status_payload(self) -> dict[str, Any]:
        snapshot = self._snapshot()
        payload = snapshot.to_payload()
        payload["artifacts_dir"] = self._to_web_path(self._artifacts_dir)
        return payload

    def start_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        run_spec = resolve_run_spec(payload)
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("A run is already active. Stop it before starting a new run.")

            run_id = self._make_run_id()
            run_dir = self._artifacts_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            features = feature_flags_for_run_spec(run_spec)
            write_run_config(
                run_dir,
                {
                    **run_spec,
                    "run_id": run_id,
                    "run_dir": self._to_web_path(run_dir),
                },
            )
            write_run_features(run_dir, features)
            write_run_status(
                run_dir,
                {
                    "run_id": run_id,
                    "state": "running",
                    "started_at": _now_iso(),
                    "last_error": None,
                },
            )

            cli_args = run_spec_to_cli_args(
                run_spec=run_spec,
                run_id=run_id,
                artifacts_dir=self._artifacts_dir,
            )
            cmd = [sys.executable, "-m", "biosnn.runners.cli", *cli_args]
            process = subprocess.Popen(
                cmd,
                cwd=str(self._repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
            self._process = process
            self._run_id = run_id
            self._run_dir = run_dir
            self._state = "running"
            self._last_error = None
            self._started_at = _now_iso()
            self._finished_at = None
            self._stop_requested_run_ids.discard(run_id)

            watcher = threading.Thread(
                target=self._watch_process,
                args=(process, run_id, run_dir),
                daemon=True,
            )
            watcher.start()

            return {
                "run_id": run_id,
                "run_dir": self._to_web_path(run_dir),
                "state": "running",
            }

    def stop_run(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
            run_id = self._run_id
            run_dir = self._run_dir
            if process is None or process.poll() is not None:
                self._process = None
                return {"stopped": False, "reason": "no-active-run"}
            if run_id is not None:
                self._stop_requested_run_ids.add(run_id)

            process.terminate()
            terminated = True
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
                terminated = True

            self._process = None
            self._state = "stopped"
            self._finished_at = _now_iso()
            self._last_error = "Run stopped by user."
            if run_dir is not None and run_id is not None:
                write_run_status(
                    run_dir,
                    {
                        "run_id": run_id,
                        "state": "stopped",
                        "started_at": self._started_at,
                        "finished_at": self._finished_at,
                        "last_error": self._last_error,
                    },
                )
            return {"stopped": terminated, "run_id": run_id}

    def shutdown(self) -> None:
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=3)
            except Exception:
                with suppress(Exception):
                    process.kill()

    def _watch_process(
        self,
        process: subprocess.Popen[str],
        run_id: str,
        run_dir: Path,
    ) -> None:
        stdout, stderr = process.communicate()
        return_code = int(process.returncode or 0)
        stop_requested = False
        with self._lock:
            if run_id in self._stop_requested_run_ids:
                self._stop_requested_run_ids.remove(run_id)
                stop_requested = True
        last_error: str | None
        if stop_requested:
            finished_state = "stopped"
            last_error = "Run stopped by user."
        else:
            last_error = _tail_text(stderr) if return_code != 0 else None
            finished_state = "done" if return_code == 0 else "error"
        finished_at = _now_iso()
        with self._lock:
            if self._process is process:
                self._process = None
                self._state = finished_state
                self._finished_at = finished_at
                self._last_error = last_error
        status_payload: dict[str, Any] = {
            "run_id": run_id,
            "state": finished_state,
            "started_at": self._started_at,
            "finished_at": finished_at,
            "return_code": return_code,
            "last_error": last_error,
        }
        out_tail = _tail_text(stdout)
        err_tail = _tail_text(stderr)
        if out_tail:
            status_payload["stdout_tail"] = out_tail
        if err_tail:
            status_payload["stderr_tail"] = err_tail
        write_run_status(run_dir, status_payload)

    def _snapshot(self) -> _RunSnapshot:
        with self._lock:
            process = self._process
            pid = process.pid if process is not None and process.poll() is None else None
            run_dir = self._to_web_path(self._run_dir) if self._run_dir is not None else None
            return _RunSnapshot(
                run_id=self._run_id,
                run_dir=run_dir,
                state=self._state,
                last_error=self._last_error,
                started_at=self._started_at,
                finished_at=self._finished_at,
                pid=pid,
            )

    def _make_run_id(self) -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"{time.time_ns() % 1_000_000:06d}"
        return f"run_{stamp}_{suffix}"

    def _to_web_path(self, path: Path | None) -> str:
        if path is None:
            return ""
        try:
            rel = path.resolve().relative_to(self._repo_root.resolve())
            return (Path("/") / rel).as_posix()
        except Exception:
            return path.as_posix()


class DashboardServerHandler(SimpleHTTPRequestHandler):
    """Static file handler with /api/* run-control endpoints."""

    def __init__(
        self,
        *args: Any,
        directory: str,
        controller: DashboardRunController,
        **kwargs: Any,
    ) -> None:
        self._controller = controller
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/demos":
            self._send_json(self._controller.demos_payload(), status=HTTPStatus.OK)
            return
        if parsed.path == "/api/run/status":
            self._send_json(self._controller.status_payload(), status=HTTPStatus.OK)
            return
        # Browser fetches (especially for large CSVs) can be aborted if the user reloads,
        # the tab is closed, or the client times out. On Windows this frequently raises
        # WinError 10053 (ConnectionAbortedError) and produces noisy stack traces.
        try:
            super().do_GET()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/run":
            payload = self._read_json_body()
            if payload is None:
                self._send_json({"error": "Invalid JSON body."}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                result = self._controller.start_run(payload)
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.CONFLICT)
                return
            self._send_json(result, status=HTTPStatus.ACCEPTED)
            return
        if parsed.path == "/api/run/stop":
            result = self._controller.stop_run()
            self._send_json(result, status=HTTPStatus.OK)
            return
        self._send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args: Any) -> None:
        _ = (fmt, args)
        # Keep dashboard server quiet in tests and CLI output.

    def _read_json_body(self) -> dict[str, Any] | None:
        length_raw = self.headers.get("Content-Length")
        if not length_raw:
            return {}
        try:
            length = int(length_raw)
        except ValueError:
            return None
        body = self.rfile.read(max(0, length))
        if not body:
            return {}
        try:
            parsed = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return cast(dict[str, Any], parsed)
        return None

    def _send_json(self, payload: dict[str, Any], *, status: HTTPStatus) -> None:
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        try:
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # Browser/client disconnected before the response could be written.
            return


class _ExclusiveThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = False

    def server_bind(self) -> None:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            with suppress(OSError):
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def start_dashboard_server(
    *,
    repo_root: Path,
    port: int,
    artifacts_dir: Path | None = None,
    initial_run_dir: Path | None = None,
    initial_state: str = "idle",
) -> ThreadingHTTPServer:
    """Start dashboard HTTP server in a daemon thread and return the server object."""

    repo_root = repo_root.resolve()
    controller = DashboardRunController(
        repo_root=repo_root,
        artifacts_dir=(artifacts_dir or (repo_root / "artifacts")).resolve(),
        initial_run_dir=initial_run_dir.resolve() if initial_run_dir is not None else None,
        initial_state=initial_state,
    )

    def handler(*args: Any, **kwargs: Any) -> DashboardServerHandler:
        return DashboardServerHandler(
            *args,
            directory=str(repo_root),
            controller=controller,
            **kwargs,
        )

    server = _ExclusiveThreadingHTTPServer(("localhost", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server.controller = controller  # type: ignore[attr-defined]
    server._dashboard_thread = thread  # type: ignore[attr-defined]
    return server


__all__ = ["DashboardRunController", "DashboardServerHandler", "start_dashboard_server"]
