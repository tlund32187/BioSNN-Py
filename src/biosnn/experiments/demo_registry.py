"""Demo registry and run-spec helpers for dashboard-driven run selection."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

DemoId = Literal[
    "network",
    "propagation_impulse",
    "delay_impulse",
    "learning_gate",
    "dopamine_plasticity",
]

RunMonitorMode = Literal["fast", "dashboard"]
RunDevice = Literal["cpu", "cuda"]
RunDtype = Literal["float32", "float64", "bfloat16", "float16"]
RunFusedLayout = Literal["auto", "coo", "csr"]
RunSynapseBackend = Literal["spmm_fused", "event_driven"]
RunRingStrategy = Literal["dense", "event_bucketed"]
RunRingDtype = Literal["none", "float32", "float16", "bfloat16"]

ALLOWED_DEMOS: tuple[DemoId, ...] = (
    "network",
    "propagation_impulse",
    "delay_impulse",
    "learning_gate",
    "dopamine_plasticity",
)
ALLOWED_DEVICE = {"cpu", "cuda"}
ALLOWED_DTYPE = {"float32", "float64", "bfloat16", "float16"}
ALLOWED_FUSED_LAYOUT = {"auto", "coo", "csr"}
ALLOWED_SYNAPSE_BACKEND = {"spmm_fused", "event_driven"}
ALLOWED_RING_STRATEGY = {"dense", "event_bucketed"}
ALLOWED_RING_DTYPE = {"none", "float32", "float16", "bfloat16"}
ALLOWED_MONITOR_MODE = {"fast", "dashboard"}

_BASE_RUN_SPEC_DEFAULTS: dict[str, Any] = {
    "demo_id": "network",
    "steps": 200,
    "dt": 1e-3,
    "seed": 123,
    "device": "cpu",
    "dtype": "float32",
    "fused_layout": "auto",
    "synapse_backend": "spmm_fused",
    "ring_strategy": "dense",
    "ring_dtype": "none",
    "max_ring_mib": 2048.0,
    "store_sparse_by_delay": None,
    "delay_steps": 3,
    "monitor_mode": "dashboard",
    "learning": {
        "enabled": False,
        "rule": "three_factor_hebbian",
        "lr": 0.1,
    },
    "modulators": {
        "enabled": False,
        "kinds": [],
        "pulse_step": 50,
        "amount": 1.0,
    },
}

_DEMO_NAMES: dict[DemoId, str] = {
    "network": "Network",
    "propagation_impulse": "Propagation Impulse",
    "delay_impulse": "Delay Impulse",
    "learning_gate": "Learning Gate",
    "dopamine_plasticity": "Dopamine Plasticity",
}

_DEMO_DEFAULT_OVERRIDES: dict[DemoId, dict[str, Any]] = {
    "network": {
        "steps": 500,
        "learning": {"enabled": False},
        "modulators": {"enabled": False, "kinds": []},
    },
    "propagation_impulse": {
        "steps": 120,
        "delay_steps": 0,
        "learning": {"enabled": False},
        "modulators": {"enabled": False, "kinds": []},
    },
    "delay_impulse": {
        "steps": 120,
        "delay_steps": 3,
        "learning": {"enabled": False},
        "modulators": {"enabled": False, "kinds": []},
    },
    "learning_gate": {
        "steps": 200,
        "delay_steps": 0,
        "learning": {
            "enabled": True,
            "rule": "three_factor_hebbian",
            "lr": 0.1,
        },
        "modulators": {"enabled": False, "kinds": []},
    },
    "dopamine_plasticity": {
        "steps": 220,
        "delay_steps": 0,
        "learning": {
            "enabled": True,
            "rule": "three_factor_hebbian",
            "lr": 0.1,
        },
        "modulators": {
            "enabled": True,
            "kinds": ["dopamine"],
            "pulse_step": 50,
            "amount": 1.0,
        },
    },
}


def _deep_merge_dict(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge_dict(cast(Mapping[str, Any], merged[key]), value)
        else:
            merged[key] = value
    return merged


def _coerce_choice(value: Any, *, allowed: set[str], default: str) -> str:
    if value is None:
        return default
    text = str(value).strip().lower()
    return text if text in allowed else default


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _coerce_nonnegative_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_optional_int(value: Any, default: int | None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    return None


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip().lower()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def default_run_spec(*, demo_id: DemoId = "network") -> dict[str, Any]:
    base = _deep_merge_dict(_BASE_RUN_SPEC_DEFAULTS, {})
    override = _DEMO_DEFAULT_OVERRIDES.get(demo_id, {})
    merged = _deep_merge_dict(base, override)
    merged["demo_id"] = demo_id
    return merged


def resolve_run_spec(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(raw) if raw else {}
    demo_raw = payload.get("demo_id")
    demo_id = cast(
        DemoId,
        _coerce_choice(demo_raw, allowed=set(ALLOWED_DEMOS), default="network"),
    )
    merged = _deep_merge_dict(default_run_spec(demo_id=demo_id), payload)

    learning_raw = merged.get("learning")
    learning_map = cast(Mapping[str, Any], learning_raw) if isinstance(learning_raw, Mapping) else {}
    modulators_raw = merged.get("modulators")
    modulators_map = (
        cast(Mapping[str, Any], modulators_raw) if isinstance(modulators_raw, Mapping) else {}
    )

    resolved: dict[str, Any] = {
        "demo_id": demo_id,
        "steps": _coerce_positive_int(merged.get("steps"), int(default_run_spec(demo_id=demo_id)["steps"])),
        "dt": _coerce_float(merged.get("dt"), float(default_run_spec(demo_id=demo_id)["dt"])),
        "seed": _coerce_optional_int(merged.get("seed"), 123),
        "device": _coerce_choice(merged.get("device"), allowed=ALLOWED_DEVICE, default="cpu"),
        "dtype": _coerce_choice(merged.get("dtype"), allowed=ALLOWED_DTYPE, default="float32"),
        "fused_layout": _coerce_choice(
            merged.get("fused_layout"), allowed=ALLOWED_FUSED_LAYOUT, default="auto"
        ),
        "synapse_backend": _coerce_choice(
            merged.get("synapse_backend"),
            allowed=ALLOWED_SYNAPSE_BACKEND,
            default="spmm_fused",
        ),
        "ring_strategy": _coerce_choice(
            merged.get("ring_strategy"), allowed=ALLOWED_RING_STRATEGY, default="dense"
        ),
        "ring_dtype": _coerce_choice(
            merged.get("ring_dtype"), allowed=ALLOWED_RING_DTYPE, default="none"
        ),
        "max_ring_mib": _coerce_float(merged.get("max_ring_mib"), 2048.0),
        "store_sparse_by_delay": _coerce_optional_bool(merged.get("store_sparse_by_delay")),
        "delay_steps": _coerce_nonnegative_int(
            merged.get("delay_steps"),
            int(default_run_spec(demo_id=demo_id)["delay_steps"]),
        ),
        "monitor_mode": _coerce_choice(
            merged.get("monitor_mode"), allowed=ALLOWED_MONITOR_MODE, default="dashboard"
        ),
        "learning": {
            "enabled": bool(learning_map.get("enabled", False)),
            "rule": str(learning_map.get("rule", "three_factor_hebbian")).strip()
            or "three_factor_hebbian",
            "lr": _coerce_float(learning_map.get("lr"), 0.1),
        },
        "modulators": {
            "enabled": bool(modulators_map.get("enabled", False)),
            "kinds": _coerce_string_list(modulators_map.get("kinds")),
            "pulse_step": _coerce_nonnegative_int(modulators_map.get("pulse_step"), 50),
            "amount": _coerce_float(modulators_map.get("amount"), 1.0),
        },
    }

    if resolved["dt"] <= 0:
        resolved["dt"] = 1e-3
    if resolved["max_ring_mib"] <= 0:
        resolved["max_ring_mib"] = 0.0
    if resolved["learning"]["lr"] <= 0:
        resolved["learning"]["lr"] = 0.1
    if resolved["modulators"]["amount"] < 0:
        resolved["modulators"]["amount"] = 0.0

    # Demo-specific behavior defaults.
    if demo_id in {"learning_gate", "dopamine_plasticity"}:
        resolved["learning"]["enabled"] = True
    if demo_id == "dopamine_plasticity":
        resolved["modulators"]["enabled"] = True
        if not resolved["modulators"]["kinds"]:
            resolved["modulators"]["kinds"] = ["dopamine"]
    return resolved


def run_spec_from_cli_args(
    *,
    args: Any,
    device: str,
) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "demo_id": getattr(args, "demo", "network"),
        "steps": getattr(args, "steps", 200),
        "dt": getattr(args, "dt", 1e-3),
        "seed": getattr(args, "seed", 123),
        "device": device,
        "dtype": "float32",
        "fused_layout": getattr(args, "fused_layout", "auto"),
        "synapse_backend": getattr(args, "synapse_backend", "spmm_fused"),
        "ring_strategy": getattr(args, "ring_strategy", "dense"),
        "ring_dtype": getattr(args, "ring_dtype", None) or "none",
        "max_ring_mib": getattr(args, "max_ring_mib", 2048.0),
        "store_sparse_by_delay": getattr(args, "store_sparse_by_delay", None),
        "delay_steps": getattr(args, "delay_steps", 3),
        "monitor_mode": getattr(args, "mode", "dashboard"),
        "learning": {
            "enabled": False,
            "rule": "three_factor_hebbian",
            "lr": getattr(args, "learning_lr", 0.1),
        },
        "modulators": {
            "enabled": False,
            "kinds": [],
            "pulse_step": getattr(args, "da_step", 50),
            "amount": getattr(args, "da_amount", 1.0),
        },
    }
    demo_name = str(getattr(args, "demo", "")).strip().lower()
    if demo_name in {"learning_gate", "dopamine_plasticity"}:
        raw["learning"]["enabled"] = True
    if demo_name == "dopamine_plasticity":
        raw["modulators"]["enabled"] = True
        raw["modulators"]["kinds"] = ["dopamine"]
    return resolve_run_spec(raw)


def run_spec_to_cli_args(
    *,
    run_spec: Mapping[str, Any],
    run_id: str,
    artifacts_dir: Path,
) -> list[str]:
    spec = resolve_run_spec(run_spec)
    args: list[str] = [
        "--demo",
        str(spec["demo_id"]),
        "--steps",
        str(int(spec["steps"])),
        "--dt",
        str(float(spec["dt"])),
        "--device",
        str(spec["device"]),
        "--mode",
        str(spec["monitor_mode"]),
        "--fused-layout",
        str(spec["fused_layout"]),
        "--synapse-backend",
        str(spec["synapse_backend"]),
        "--ring-strategy",
        str(spec["ring_strategy"]),
        "--max-ring-mib",
        str(float(spec["max_ring_mib"])),
        "--delay_steps",
        str(int(spec["delay_steps"])),
        "--learning_lr",
        str(float(spec["learning"]["lr"])),
        "--da_amount",
        str(float(spec["modulators"]["amount"])),
        "--da_step",
        str(int(spec["modulators"]["pulse_step"])),
        "--run-id",
        run_id,
        "--artifacts-dir",
        str(artifacts_dir),
        "--no-open",
        "--no-server",
    ]
    seed = spec.get("seed")
    if seed is not None:
        args.extend(["--seed", str(int(seed))])
    ring_dtype = str(spec.get("ring_dtype", "none")).strip().lower()
    if ring_dtype in {"float32", "float16", "bfloat16"}:
        args.extend(["--ring-dtype", ring_dtype])
    store_sparse = spec.get("store_sparse_by_delay")
    if store_sparse is not None:
        args.extend(["--store-sparse-by-delay", "true" if bool(store_sparse) else "false"])
    return args


def feature_flags_for_run_spec(
    run_spec: Mapping[str, Any],
) -> dict[str, Any]:
    spec = resolve_run_spec(run_spec)
    demo_id = cast(DemoId, spec["demo_id"])
    known_max_delay: int | None
    if demo_id == "propagation_impulse":
        known_max_delay = 0
    elif demo_id == "delay_impulse":
        known_max_delay = int(spec["delay_steps"])
    elif demo_id in {"learning_gate", "dopamine_plasticity"}:
        known_max_delay = 0
    else:
        known_max_delay = None

    ring_len = (known_max_delay + 1) if known_max_delay is not None else None
    monitor_mode = cast(RunMonitorMode, spec["monitor_mode"])
    monitor_policy = "sync_opt_in" if monitor_mode == "dashboard" else "cuda_safe"
    return {
        "demo_id": demo_id,
        "learning": {
            "enabled": bool(spec["learning"]["enabled"]),
            "rule": spec["learning"]["rule"] if spec["learning"]["enabled"] else None,
            "lr": float(spec["learning"]["lr"]) if spec["learning"]["enabled"] else None,
        },
        "delays": {
            "enabled": known_max_delay is None or known_max_delay > 0,
            "max_delay_steps": known_max_delay,
            "ring_len": ring_len,
        },
        "modulators": {
            "enabled": bool(spec["modulators"]["enabled"]),
            "kinds": list(spec["modulators"]["kinds"]),
            "pulse_step": int(spec["modulators"]["pulse_step"]),
            "amount": float(spec["modulators"]["amount"]),
        },
        "synapse": {
            "backend": spec["synapse_backend"],
            "fused_layout": spec["fused_layout"],
            "ring_strategy": spec["ring_strategy"],
            "ring_dtype": spec["ring_dtype"],
            "store_sparse_by_delay": spec["store_sparse_by_delay"],
        },
        "monitor": {
            "mode": monitor_mode,
            "sync_policy": monitor_policy,
        },
    }


def list_demo_definitions() -> list[dict[str, Any]]:
    definitions: list[dict[str, Any]] = []
    for demo_id in ALLOWED_DEMOS:
        defaults = resolve_run_spec(default_run_spec(demo_id=demo_id))
        definitions.append(
            {
                "id": demo_id,
                "name": _DEMO_NAMES[demo_id],
                "defaults": defaults,
            }
        )
    return definitions


__all__ = [
    "ALLOWED_DEMOS",
    "DemoId",
    "default_run_spec",
    "feature_flags_for_run_spec",
    "list_demo_definitions",
    "resolve_run_spec",
    "run_spec_from_cli_args",
    "run_spec_to_cli_args",
]
