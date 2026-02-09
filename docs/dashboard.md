# Dashboard Run Control

The dashboard now supports local run orchestration through HTTP endpoints exposed by the CLI dashboard server.

## Run Selection

When the dashboard is opened without `?run=...`, it queries the local API and shows:

- available demos (`/api/demos`)
- current run status (`/api/run/status`)
- run controls to start/stop runs

If the URL has `?run=/artifacts/run_...`, the dashboard loads that run folder directly and does not require the API.

## API Endpoints

- `GET /api/demos`
  - returns demo IDs, names, and default parameters
- `POST /api/run`
  - accepts a RunSpec-like JSON payload and starts a new run subprocess
- `GET /api/run/status`
  - returns current run state (`idle`, `running`, `done`, `error`, `stopped`)
- `POST /api/run/stop`
  - best-effort stop for the active run

## Run Manifests

Each run folder writes:

- `run_config.json` resolved run parameters
- `run_features.json` feature checklist payload
- `run_status.json` lifecycle status

The dashboard checklist panel reads `run_features.json` to show enabled subsystems for the active run.
