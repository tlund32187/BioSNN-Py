# Public API (biosnn.api)

**Policy:** Only symbols re-exported from `biosnn.api` are considered public and semver-stable.

This document is a living log of the intended public façade. For now it’s a scaffold with placeholders.

## Stable façade modules (planned)
- `biosnn.api.version` — package version and compatibility markers
- `biosnn.api.deprecations` — standardized deprecation helpers (central policy)

## Planned “first-class” public concepts (to be refined before coding)
- Simulation engine entrypoints (build/run/step)
- Contract types (interfaces/protocols) for neurons/synapses/learning/modulators/monitors
- Network construction (builders) and configuration DTOs
- Monitoring and export APIs (CSV/graph/trace)

## Stability levels
- **Stable:** can only change with semver MAJOR
- **Provisional:** may change with semver MINOR (explicitly labeled)
- **Experimental:** may change any time (not exported from `biosnn.api`)
