"""Demo registry and run-spec helpers for dashboard-driven run selection."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

DemoId = Literal[
    "network",
    "pruning_sparse",
    "neurogenesis_sparse",
    "propagation_impulse",
    "delay_impulse",
    "learning_gate",
    "dopamine_plasticity",
    "logic_curriculum",
    "logic_and",
    "logic_or",
    "logic_xor",
    "logic_nand",
    "logic_nor",
    "logic_xnor",
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
    "pruning_sparse",
    "neurogenesis_sparse",
    "propagation_impulse",
    "delay_impulse",
    "learning_gate",
    "dopamine_plasticity",
    "logic_curriculum",
    "logic_and",
    "logic_or",
    "logic_xor",
    "logic_nand",
    "logic_nor",
    "logic_xnor",
)
ALLOWED_DEVICE = {"cpu", "cuda"}
ALLOWED_DTYPE = {"float32", "float64", "bfloat16", "float16"}
ALLOWED_FUSED_LAYOUT = {"auto", "coo", "csr"}
ALLOWED_SYNAPSE_BACKEND = {"spmm_fused", "event_driven"}
ALLOWED_RING_STRATEGY = {"dense", "event_bucketed"}
ALLOWED_RING_DTYPE = {"none", "float32", "float16", "bfloat16"}
ALLOWED_MONITOR_MODE = {"fast", "dashboard"}

_LOGIC_DEMO_TO_GATE: dict[DemoId, str] = {
    "logic_and": "and",
    "logic_or": "or",
    "logic_xor": "xor",
    "logic_nand": "nand",
    "logic_nor": "nor",
    "logic_xnor": "xnor",
}
_LOGIC_GATE_VALUES = {"and", "or", "xor", "nand", "nor", "xnor"}
_LOGIC_LEARNING_MODES = {"rstdp", "surrogate", "none"}
_LOGIC_SAMPLING_METHODS = {"sequential", "random_balanced"}
_LOGIC_CURRICULUM_DEFAULT = "or,and,nor,nand,xor,xnor"

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
    "pruning_sparse": "Pruning Sparse",
    "neurogenesis_sparse": "Neurogenesis Sparse",
    "propagation_impulse": "Propagation Impulse",
    "delay_impulse": "Delay Impulse",
    "learning_gate": "Learning Gate",
    "dopamine_plasticity": "Dopamine Plasticity",
    "logic_curriculum": "Logic Curriculum",
    "logic_and": "Logic AND",
    "logic_or": "Logic OR",
    "logic_xor": "Logic XOR",
    "logic_nand": "Logic NAND",
    "logic_nor": "Logic NOR",
    "logic_xnor": "Logic XNOR",
}

_DEMO_DEFAULT_OVERRIDES: dict[DemoId, dict[str, Any]] = {
    "network": {
        "steps": 500,
        "learning": {"enabled": False},
        "modulators": {"enabled": False, "kinds": []},
    },
    "pruning_sparse": {
        "steps": 5000,
        "learning": {"enabled": False},
        "modulators": {"enabled": False, "kinds": []},
    },
    "neurogenesis_sparse": {
        "steps": 5000,
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
    "logic_and": {
        "steps": 5000,
        "learning": {"enabled": True, "rule": "rstdp", "lr": 0.1},
        "modulators": {"enabled": False, "kinds": []},
        "logic_gate": "and",
        "logic_learning_mode": "rstdp",
        "logic_sim_steps_per_trial": 10,
        "logic_sampling_method": "sequential",
        "logic_debug": False,
        "logic_debug_every": 25,
    },
    "logic_curriculum": {
        "steps": 2500,
        "learning": {"enabled": True, "rule": "rstdp", "lr": 0.1},
        "modulators": {"enabled": False, "kinds": []},
        "logic_gate": "xor",
        "logic_learning_mode": "rstdp",
        "logic_sim_steps_per_trial": 10,
        "logic_sampling_method": "sequential",
        "logic_curriculum_gates": _LOGIC_CURRICULUM_DEFAULT,
        "logic_curriculum_replay_ratio": 0.35,
        "logic_debug": False,
        "logic_debug_every": 25,
    },
    "logic_or": {
        "steps": 5000,
        "learning": {"enabled": True, "rule": "rstdp", "lr": 0.1},
        "modulators": {"enabled": False, "kinds": []},
        "logic_gate": "or",
        "logic_learning_mode": "rstdp",
        "logic_sim_steps_per_trial": 10,
        "logic_sampling_method": "sequential",
        "logic_debug": False,
        "logic_debug_every": 25,
    },
    "logic_xor": {
        "steps": 20000,
        "learning": {"enabled": True, "rule": "rstdp", "lr": 0.1},
        "modulators": {"enabled": False, "kinds": []},
        "logic_gate": "xor",
        "logic_learning_mode": "rstdp",
        "logic_sim_steps_per_trial": 10,
        "logic_sampling_method": "sequential",
        "logic_debug": False,
        "logic_debug_every": 25,
    },
    "logic_nand": {
        "steps": 5000,
        "learning": {"enabled": True, "rule": "rstdp", "lr": 0.1},
        "modulators": {"enabled": False, "kinds": []},
        "logic_gate": "nand",
        "logic_learning_mode": "rstdp",
        "logic_sim_steps_per_trial": 10,
        "logic_sampling_method": "sequential",
        "logic_debug": False,
        "logic_debug_every": 25,
    },
    "logic_nor": {
        "steps": 5000,
        "learning": {"enabled": True, "rule": "rstdp", "lr": 0.1},
        "modulators": {"enabled": False, "kinds": []},
        "logic_gate": "nor",
        "logic_learning_mode": "rstdp",
        "logic_sim_steps_per_trial": 10,
        "logic_sampling_method": "sequential",
        "logic_debug": False,
        "logic_debug_every": 25,
    },
    "logic_xnor": {
        "steps": 5000,
        "learning": {"enabled": True, "rule": "rstdp", "lr": 0.1},
        "modulators": {"enabled": False, "kinds": []},
        "logic_gate": "xnor",
        "logic_learning_mode": "rstdp",
        "logic_sim_steps_per_trial": 10,
        "logic_sampling_method": "sequential",
        "logic_debug": False,
        "logic_debug_every": 25,
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
    if demo_id in _LOGIC_DEMO_TO_GATE:
        resolved["logic_gate"] = _coerce_choice(
            merged.get("logic_gate"),
            allowed=_LOGIC_GATE_VALUES,
            default=_LOGIC_DEMO_TO_GATE[demo_id],
        )
        resolved["logic_learning_mode"] = _coerce_choice(
            merged.get("logic_learning_mode"),
            allowed=_LOGIC_LEARNING_MODES,
            default="rstdp",
        )
        resolved["logic_sim_steps_per_trial"] = _coerce_positive_int(
            merged.get("logic_sim_steps_per_trial"),
            10,
        )
        resolved["logic_sampling_method"] = _coerce_choice(
            merged.get("logic_sampling_method"),
            allowed=_LOGIC_SAMPLING_METHODS,
            default="sequential",
        )
        resolved["logic_debug"] = bool(merged.get("logic_debug", False))
        resolved["logic_debug_every"] = _coerce_positive_int(
            merged.get("logic_debug_every"),
            25,
        )
    if demo_id == "logic_curriculum":
        curriculum_raw = str(merged.get("logic_curriculum_gates", _LOGIC_CURRICULUM_DEFAULT))
        curriculum_items = [
            part.strip().lower() for part in curriculum_raw.split(",") if part.strip()
        ]
        curriculum_items = [part for part in curriculum_items if part in _LOGIC_GATE_VALUES]
        if not curriculum_items:
            curriculum_items = _LOGIC_CURRICULUM_DEFAULT.split(",")
        replay_ratio = _coerce_float(merged.get("logic_curriculum_replay_ratio"), 0.35)
        if replay_ratio < 0.0:
            replay_ratio = 0.0
        if replay_ratio > 1.0:
            replay_ratio = 1.0
        resolved["logic_curriculum_gates"] = ",".join(curriculum_items)
        resolved["logic_curriculum_replay_ratio"] = replay_ratio
        resolved["logic_gate"] = curriculum_items[-1]
        resolved["logic_learning_mode"] = "rstdp"
        resolved["learning"]["enabled"] = True
        resolved["learning"]["rule"] = "rstdp"
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
    if demo_name in _LOGIC_DEMO_TO_GATE:
        demo_key = cast(DemoId, demo_name)
        logic_mode = str(getattr(args, "logic_learning_mode", "rstdp")).strip().lower()
        if logic_mode not in _LOGIC_LEARNING_MODES:
            logic_mode = "rstdp"
        raw["learning"]["enabled"] = logic_mode != "none"
        raw["learning"]["rule"] = logic_mode
        raw["logic_gate"] = _coerce_choice(
            getattr(args, "logic_gate", None),
            allowed=_LOGIC_GATE_VALUES,
            default=_LOGIC_DEMO_TO_GATE[demo_key],
        )
        raw["logic_learning_mode"] = logic_mode
        raw["logic_sim_steps_per_trial"] = max(
            1,
            int(getattr(args, "logic_sim_steps_per_trial", 10)),
        )
        raw["logic_sampling_method"] = _coerce_choice(
            getattr(args, "logic_sampling_method", None),
            allowed=_LOGIC_SAMPLING_METHODS,
            default="sequential",
        )
        raw["logic_debug"] = bool(getattr(args, "logic_debug", False))
        raw["logic_debug_every"] = max(1, int(getattr(args, "logic_debug_every", 25)))
    if demo_name == "logic_curriculum":
        raw["learning"]["enabled"] = True
        raw["learning"]["rule"] = "rstdp"
        raw["logic_learning_mode"] = "rstdp"
        raw["logic_sim_steps_per_trial"] = max(
            1,
            int(getattr(args, "logic_sim_steps_per_trial", 10)),
        )
        raw["logic_sampling_method"] = _coerce_choice(
            getattr(args, "logic_sampling_method", None),
            allowed=_LOGIC_SAMPLING_METHODS,
            default="sequential",
        )
        raw["logic_curriculum_gates"] = str(
            getattr(args, "logic_curriculum_gates", _LOGIC_CURRICULUM_DEFAULT)
        )
        raw["logic_curriculum_replay_ratio"] = float(
            getattr(args, "logic_curriculum_replay_ratio", 0.35)
        )
        raw["logic_debug"] = bool(getattr(args, "logic_debug", False))
        raw["logic_debug_every"] = max(1, int(getattr(args, "logic_debug_every", 25)))
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
    demo_id = cast(DemoId, spec["demo_id"])
    if demo_id in _LOGIC_DEMO_TO_GATE:
        gate = str(spec.get("logic_gate", _LOGIC_DEMO_TO_GATE[demo_id])).strip().lower()
        if gate not in _LOGIC_GATE_VALUES:
            gate = _LOGIC_DEMO_TO_GATE[demo_id]
        if not bool(spec["learning"]["enabled"]):
            logic_mode = "none"
        else:
            learning_rule = str(spec["learning"]["rule"]).strip().lower()
            logic_mode = "surrogate" if learning_rule == "surrogate" else "rstdp"
        args.extend(
            [
                "--logic-gate",
                gate,
                "--logic-learning-mode",
                logic_mode,
                "--logic-sim-steps-per-trial",
                str(int(spec.get("logic_sim_steps_per_trial", 10))),
                "--logic-sampling-method",
                str(spec.get("logic_sampling_method", "sequential")),
                "--logic-debug-every",
                str(int(spec.get("logic_debug_every", 25))),
            ]
        )
        if bool(spec.get("logic_debug", False)):
            args.append("--logic-debug")
    if demo_id == "logic_curriculum":
        args.extend(
            [
                "--logic-curriculum-gates",
                str(spec.get("logic_curriculum_gates", _LOGIC_CURRICULUM_DEFAULT)),
                "--logic-curriculum-replay-ratio",
                str(float(spec.get("logic_curriculum_replay_ratio", 0.35))),
                "--logic-learning-mode",
                "rstdp",
                "--logic-sim-steps-per-trial",
                str(int(spec.get("logic_sim_steps_per_trial", 10))),
                "--logic-sampling-method",
                str(spec.get("logic_sampling_method", "sequential")),
                "--logic-debug-every",
                str(int(spec.get("logic_debug_every", 25))),
            ]
        )
        if bool(spec.get("logic_debug", False)):
            args.append("--logic-debug")
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
    elif demo_id in {"learning_gate", "dopamine_plasticity", "logic_curriculum"} or demo_id in _LOGIC_DEMO_TO_GATE:
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
        "pruning": {
            "enabled": demo_id == "pruning_sparse",
        },
        "neurogenesis": {
            "enabled": demo_id == "neurogenesis_sparse",
        },
        "logic_gate": spec.get("logic_gate") if demo_id in _LOGIC_DEMO_TO_GATE else None,
        "logic_curriculum_gates": spec.get("logic_curriculum_gates")
        if demo_id == "logic_curriculum"
        else None,
        "logic_curriculum_replay_ratio": spec.get("logic_curriculum_replay_ratio")
        if demo_id == "logic_curriculum"
        else None,
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
