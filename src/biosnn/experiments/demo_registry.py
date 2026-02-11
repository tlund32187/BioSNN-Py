"""Demo registry and run-spec helpers for dashboard-driven run selection."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

DemoId = Literal[
    "network",
    "vision",
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
RunLogicBackend = Literal["harness", "engine"]
RunHomeostasisRule = Literal["rate_ema_threshold"]
RunHomeostasisScope = Literal["per_population", "per_neuron"]
RunModulatorFieldType = Literal["global_scalar", "grid_diffusion_2d"]
RunSynapseReceptorMode = Literal["exc_only", "ei_ampa_nmda_gabaa", "ei_ampa_nmda_gabaa_gabab"]

ALLOWED_DEMOS: tuple[DemoId, ...] = (
    "network",
    "vision",
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
ALLOWED_LOGIC_BACKEND = {"harness", "engine"}
ALLOWED_HOMEOSTASIS_RULE = {"rate_ema_threshold"}
ALLOWED_HOMEOSTASIS_SCOPE = {"per_population", "per_neuron"}
ALLOWED_MODULATOR_FIELD_TYPE = {"global_scalar", "grid_diffusion_2d"}
ALLOWED_SYNAPSE_RECEPTOR_MODE = {"exc_only", "ei_ampa_nmda_gabaa", "ei_ampa_nmda_gabaa_gabab"}
ALLOWED_EXPLORATION_MODE = {"epsilon_greedy"}
ALLOWED_EXPLORATION_TIE_BREAK = {"random_among_max", "alternate", "prefer_last"}

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
_LOGIC_NEURON_MODELS = {"adex_3c", "lif_3c"}
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
    "monitors_enabled": True,
    "learning": {
        "enabled": False,
        "rule": "three_factor_hebbian",
        "lr": 0.1,
    },
    "synapse": {
        "backend": "spmm_fused",
        "fused_layout": "auto",
        "ring_strategy": "dense",
        "ring_dtype": "none",
        "store_sparse_by_delay": None,
        "receptor_mode": "exc_only",
    },
    "advanced_synapse": {
        "enabled": False,
        "conductance_mode": False,
        "nmda_voltage_block": False,
        "stp_enabled": False,
    },
    "modulators": {
        "enabled": False,
        "kinds": [],
        "pulse_step": 50,
        "amount": 1.0,
        "field_type": "global_scalar",
        "grid_size": [16, 16],
        "world_extent": [1.0, 1.0],
        "diffusion": 0.0,
        "decay_tau": 1.0,
        "deposit_sigma": 0.0,
    },
    "wrapper": {
        "enabled": False,
        "ach_lr_gain": 0.0,
        "ne_lr_gain": 0.0,
        "ht_extra_weight_decay": 0.0,
    },
    "excitability_modulation": {
        "enabled": False,
        "targets": ["hidden", "out"],
        "compartment": "soma",
        "ach_gain": 0.0,
        "ne_gain": 0.0,
        "ht_gain": 0.0,
        "clamp_abs": 1.0,
    },
    "homeostasis": {
        "enabled": False,
        "rule": "rate_ema_threshold",
        "alpha": 0.01,
        "eta": 1e-3,
        "r_target": 0.05,
        "clamp_min": 0.0,
        "clamp_max": 0.05,
        "scope": "per_neuron",
    },
    "pruning": {
        "enabled": False,
        "prune_interval_steps": 250,
        "usage_alpha": 0.01,
        "w_min": 0.05,
        "usage_min": 0.01,
        "k_min_out": 1,
        "k_min_in": 1,
        "max_prune_fraction_per_interval": 0.10,
        "verbose": False,
    },
    "neurogenesis": {
        "enabled": False,
        "growth_interval_steps": 500,
        "add_neurons_per_event": 4,
        "newborn_plasticity_multiplier": 1.5,
        "newborn_duration_steps": 250,
        "max_total_neurons": 20000,
        "verbose": False,
    },
    "logic": {
        "learn_every": 1,
        "reward_delivery_steps": 2,
        "reward_delivery_clamp_input": True,
        "exploration": {
            "enabled": True,
            "mode": "epsilon_greedy",
            "epsilon_start": 0.20,
            "epsilon_end": 0.01,
            "epsilon_decay_trials": 3000,
            "tie_break": "random_among_max",
            "seed": 123,
        },
    },
}

_DEMO_NAMES: dict[DemoId, str] = {
    "network": "Network",
    "vision": "Vision",
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
    "vision": {
        "steps": 500,
        "learning": {"enabled": False},
        "modulators": {"enabled": False, "kinds": []},
    },
    "pruning_sparse": {
        "steps": 5000,
        "learning": {"enabled": False},
        "modulators": {"enabled": False, "kinds": []},
        "pruning": {"enabled": True, "verbose": True},
    },
    "neurogenesis_sparse": {
        "steps": 5000,
        "learning": {"enabled": False},
        "modulators": {"enabled": False, "kinds": []},
        "neurogenesis": {"enabled": True, "verbose": True},
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
        "logic_backend": "harness",
        "logic_gate": "and",
        "logic_learning_mode": "rstdp",
        "logic_neuron_model": "adex_3c",
        "logic_sim_steps_per_trial": 10,
        "logic_sampling_method": "sequential",
        "logic_debug": False,
        "logic_debug_every": 25,
    },
    "logic_curriculum": {
        "steps": 2500,
        "learning": {"enabled": True, "rule": "rstdp", "lr": 0.1},
        "modulators": {"enabled": False, "kinds": []},
        "logic_backend": "harness",
        "logic_gate": "xor",
        "logic_learning_mode": "rstdp",
        "logic_neuron_model": "adex_3c",
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
        "logic_backend": "harness",
        "logic_gate": "or",
        "logic_learning_mode": "rstdp",
        "logic_neuron_model": "adex_3c",
        "logic_sim_steps_per_trial": 10,
        "logic_sampling_method": "sequential",
        "logic_debug": False,
        "logic_debug_every": 25,
    },
    "logic_xor": {
        "steps": 20000,
        "learning": {"enabled": True, "rule": "rstdp", "lr": 0.1},
        "modulators": {"enabled": False, "kinds": []},
        "logic_backend": "harness",
        "logic_gate": "xor",
        "logic_learning_mode": "rstdp",
        "logic_neuron_model": "adex_3c",
        "logic_sim_steps_per_trial": 10,
        "logic_sampling_method": "sequential",
        "logic_debug": False,
        "logic_debug_every": 25,
    },
    "logic_nand": {
        "steps": 5000,
        "learning": {"enabled": True, "rule": "rstdp", "lr": 0.1},
        "modulators": {"enabled": False, "kinds": []},
        "logic_backend": "harness",
        "logic_gate": "nand",
        "logic_learning_mode": "rstdp",
        "logic_neuron_model": "adex_3c",
        "logic_sim_steps_per_trial": 10,
        "logic_sampling_method": "sequential",
        "logic_debug": False,
        "logic_debug_every": 25,
    },
    "logic_nor": {
        "steps": 5000,
        "learning": {"enabled": True, "rule": "rstdp", "lr": 0.1},
        "modulators": {"enabled": False, "kinds": []},
        "logic_backend": "harness",
        "logic_gate": "nor",
        "logic_learning_mode": "rstdp",
        "logic_neuron_model": "adex_3c",
        "logic_sim_steps_per_trial": 10,
        "logic_sampling_method": "sequential",
        "logic_debug": False,
        "logic_debug_every": 25,
    },
    "logic_xnor": {
        "steps": 5000,
        "learning": {"enabled": True, "rule": "rstdp", "lr": 0.1},
        "modulators": {"enabled": False, "kinds": []},
        "logic_backend": "harness",
        "logic_gate": "xnor",
        "logic_learning_mode": "rstdp",
        "logic_neuron_model": "adex_3c",
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


def _coerce_float_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, float] = {}
    for key, raw in value.items():
        token = str(key).strip().lower()
        if not token:
            continue
        out[token] = _coerce_float(raw, 0.0)
    return out


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _coerce_grid_size(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    parts: tuple[Any, Any] | None = None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        parts = (value[0], value[1])
    elif isinstance(value, str):
        text = value.strip().lower()
        if "x" in text:
            items = [part.strip() for part in text.split("x", maxsplit=1)]
            if len(items) == 2:
                parts = (items[0], items[1])
        elif "," in text:
            items = [part.strip() for part in text.split(",", maxsplit=1)]
            if len(items) == 2:
                parts = (items[0], items[1])
    if parts is None:
        return default
    try:
        h = int(parts[0])
        w = int(parts[1])
    except (TypeError, ValueError):
        return default
    if h <= 0 or w <= 0:
        return default
    return (h, w)


def _coerce_float_pair(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    parts: tuple[Any, Any] | None = None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        parts = (value[0], value[1])
    elif isinstance(value, str):
        text = value.strip()
        if "," in text:
            items = [part.strip() for part in text.split(",", maxsplit=1)]
            if len(items) == 2:
                parts = (items[0], items[1])
    if parts is None:
        return default
    try:
        x = float(parts[0])
        y = float(parts[1])
    except (TypeError, ValueError):
        return default
    if x <= 0.0 or y <= 0.0:
        return default
    return (x, y)


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
    default_spec = default_run_spec(demo_id=demo_id)

    learning_raw = merged.get("learning")
    learning_map = cast(Mapping[str, Any], learning_raw) if isinstance(learning_raw, Mapping) else {}
    modulators_raw = merged.get("modulators")
    modulators_map = (
        cast(Mapping[str, Any], modulators_raw) if isinstance(modulators_raw, Mapping) else {}
    )
    synapse_raw = merged.get("synapse")
    synapse_map = cast(Mapping[str, Any], synapse_raw) if isinstance(synapse_raw, Mapping) else {}
    homeostasis_raw = merged.get("homeostasis")
    homeostasis_map = (
        cast(Mapping[str, Any], homeostasis_raw) if isinstance(homeostasis_raw, Mapping) else {}
    )
    pruning_raw = merged.get("pruning")
    pruning_map = cast(Mapping[str, Any], pruning_raw) if isinstance(pruning_raw, Mapping) else {}
    neurogenesis_raw = merged.get("neurogenesis")
    neurogenesis_map = (
        cast(Mapping[str, Any], neurogenesis_raw) if isinstance(neurogenesis_raw, Mapping) else {}
    )
    advanced_synapse_raw = merged.get("advanced_synapse")
    advanced_synapse_map = (
        cast(Mapping[str, Any], advanced_synapse_raw)
        if isinstance(advanced_synapse_raw, Mapping)
        else {}
    )
    wrapper_raw = merged.get("wrapper")
    wrapper_map = cast(Mapping[str, Any], wrapper_raw) if isinstance(wrapper_raw, Mapping) else {}
    excitability_raw = merged.get("excitability_modulation")
    excitability_map = (
        cast(Mapping[str, Any], excitability_raw) if isinstance(excitability_raw, Mapping) else {}
    )
    logic_raw = merged.get("logic")
    logic_map = cast(Mapping[str, Any], logic_raw) if isinstance(logic_raw, Mapping) else {}
    exploration_raw = _first_non_none(logic_map.get("exploration"), merged.get("exploration"))
    exploration_map = (
        cast(Mapping[str, Any], exploration_raw) if isinstance(exploration_raw, Mapping) else {}
    )

    default_learning = cast(Mapping[str, Any], default_spec.get("learning", {}))
    default_modulators = cast(Mapping[str, Any], default_spec.get("modulators", {}))
    default_synapse = cast(Mapping[str, Any], default_spec.get("synapse", {}))
    default_advanced_synapse = cast(
        Mapping[str, Any], default_spec.get("advanced_synapse", {})
    )
    default_wrapper = cast(Mapping[str, Any], default_spec.get("wrapper", {}))
    default_excitability = cast(
        Mapping[str, Any], default_spec.get("excitability_modulation", {})
    )
    default_homeostasis = cast(Mapping[str, Any], default_spec.get("homeostasis", {}))
    default_pruning = cast(Mapping[str, Any], default_spec.get("pruning", {}))
    default_neurogenesis = cast(Mapping[str, Any], default_spec.get("neurogenesis", {}))
    default_logic = cast(Mapping[str, Any], default_spec.get("logic", {}))
    default_exploration = cast(Mapping[str, Any], default_logic.get("exploration", {}))

    synapse_backend = _coerce_choice(
        _first_non_none(synapse_map.get("backend"), merged.get("synapse_backend")),
        allowed=ALLOWED_SYNAPSE_BACKEND,
        default=str(default_synapse.get("backend", "spmm_fused")),
    )
    fused_layout = _coerce_choice(
        _first_non_none(synapse_map.get("fused_layout"), merged.get("fused_layout")),
        allowed=ALLOWED_FUSED_LAYOUT,
        default=str(default_synapse.get("fused_layout", "auto")),
    )
    ring_strategy = _coerce_choice(
        _first_non_none(synapse_map.get("ring_strategy"), merged.get("ring_strategy")),
        allowed=ALLOWED_RING_STRATEGY,
        default=str(default_synapse.get("ring_strategy", "dense")),
    )
    ring_dtype = _coerce_choice(
        _first_non_none(synapse_map.get("ring_dtype"), merged.get("ring_dtype")),
        allowed=ALLOWED_RING_DTYPE,
        default=str(default_synapse.get("ring_dtype", "none")),
    )
    store_sparse_by_delay = _coerce_optional_bool(
        _first_non_none(synapse_map.get("store_sparse_by_delay"), merged.get("store_sparse_by_delay"))
    )
    receptor_mode = _coerce_choice(
        _first_non_none(synapse_map.get("receptor_mode"), merged.get("receptor_mode")),
        allowed=ALLOWED_SYNAPSE_RECEPTOR_MODE,
        default=str(default_synapse.get("receptor_mode", "exc_only")),
    )

    default_grid_size = _coerce_grid_size(default_modulators.get("grid_size"), (16, 16))
    default_world_extent = _coerce_float_pair(default_modulators.get("world_extent"), (1.0, 1.0))
    grid_size = _coerce_grid_size(modulators_map.get("grid_size"), default_grid_size)
    world_extent = _coerce_float_pair(modulators_map.get("world_extent"), default_world_extent)
    mod_field_type = _coerce_choice(
        modulators_map.get("field_type"),
        allowed=ALLOWED_MODULATOR_FIELD_TYPE,
        default=str(default_modulators.get("field_type", "global_scalar")),
    )
    wrapper_combine_mode = _coerce_choice(
        wrapper_map.get("combine_mode"),
        allowed={"exp", "linear"},
        default=str(default_wrapper.get("combine_mode", "exp")),
    )
    wrapper_missing_policy = _coerce_choice(
        wrapper_map.get("missing_modulators_policy"),
        allowed={"zero"},
        default=str(default_wrapper.get("missing_modulators_policy", "zero")),
    )
    excitability_targets = _coerce_string_list(
        _first_non_none(excitability_map.get("targets"), default_excitability.get("targets"))
    )
    if not excitability_targets:
        excitability_targets = ["hidden", "out"]
    excitability_compartment = _coerce_choice(
        _first_non_none(excitability_map.get("compartment"), default_excitability.get("compartment")),
        allowed={"soma", "dendrite", "ais", "axon"},
        default="soma",
    )
    adv_reversal = _coerce_float_mapping(
        _first_non_none(
            advanced_synapse_map.get("reversal_potential_v"),
            default_advanced_synapse.get("reversal_potential_v"),
        )
    )
    post_voltage_source = _coerce_choice(
        _first_non_none(
            advanced_synapse_map.get("post_voltage_source"),
            default_advanced_synapse.get("post_voltage_source"),
        ),
        allowed={"auto", "soma", "dendrite"},
        default="auto",
    )
    exploration_mode = _coerce_choice(
        _first_non_none(exploration_map.get("mode"), default_exploration.get("mode")),
        allowed=ALLOWED_EXPLORATION_MODE,
        default=str(default_exploration.get("mode", "epsilon_greedy")),
    )
    exploration_tie_break = _coerce_choice(
        _first_non_none(exploration_map.get("tie_break"), default_exploration.get("tie_break")),
        allowed=ALLOWED_EXPLORATION_TIE_BREAK,
        default=str(default_exploration.get("tie_break", "random_among_max")),
    )
    reward_clamp_input = _coerce_optional_bool(
        _first_non_none(
            logic_map.get("reward_delivery_clamp_input"),
            merged.get("reward_delivery_clamp_input"),
            default_logic.get("reward_delivery_clamp_input", True),
        )
    )
    if reward_clamp_input is None:
        reward_clamp_input = bool(default_logic.get("reward_delivery_clamp_input", True))
    monitors_enabled = _coerce_optional_bool(merged.get("monitors_enabled"))
    if monitors_enabled is None:
        monitors_enabled = bool(default_spec.get("monitors_enabled", True))

    resolved: dict[str, Any] = {
        "demo_id": demo_id,
        "steps": _coerce_positive_int(merged.get("steps"), int(default_spec["steps"])),
        "dt": _coerce_float(merged.get("dt"), float(default_spec["dt"])),
        "seed": _coerce_optional_int(merged.get("seed"), 123),
        "device": _coerce_choice(merged.get("device"), allowed=ALLOWED_DEVICE, default="cpu"),
        "dtype": _coerce_choice(merged.get("dtype"), allowed=ALLOWED_DTYPE, default="float32"),
        "fused_layout": fused_layout,
        "synapse_backend": synapse_backend,
        "ring_strategy": ring_strategy,
        "ring_dtype": ring_dtype,
        "max_ring_mib": _coerce_float(merged.get("max_ring_mib"), 2048.0),
        "store_sparse_by_delay": store_sparse_by_delay,
        "delay_steps": _coerce_nonnegative_int(
            merged.get("delay_steps"),
            int(default_spec["delay_steps"]),
        ),
        "monitor_mode": _coerce_choice(
            merged.get("monitor_mode"), allowed=ALLOWED_MONITOR_MODE, default="dashboard"
        ),
        "monitors_enabled": bool(monitors_enabled),
        "learning": {
            "enabled": bool(learning_map.get("enabled", default_learning.get("enabled", False))),
            "rule": str(
                learning_map.get("rule", default_learning.get("rule", "three_factor_hebbian"))
            ).strip()
            or "three_factor_hebbian",
            "lr": _coerce_float(learning_map.get("lr"), float(default_learning.get("lr", 0.1))),
        },
        "modulators": {
            "enabled": bool(modulators_map.get("enabled", default_modulators.get("enabled", False))),
            "kinds": _coerce_string_list(modulators_map.get("kinds")),
            "pulse_step": _coerce_nonnegative_int(
                modulators_map.get("pulse_step"),
                int(default_modulators.get("pulse_step", 50)),
            ),
            "amount": _coerce_float(modulators_map.get("amount"), float(default_modulators.get("amount", 1.0))),
            "field_type": mod_field_type,
            "grid_size": [int(grid_size[0]), int(grid_size[1])],
            "world_extent": [float(world_extent[0]), float(world_extent[1])],
            "diffusion": _coerce_float(
                modulators_map.get("diffusion"),
                float(default_modulators.get("diffusion", 0.0)),
            ),
            "decay_tau": _coerce_float(
                modulators_map.get("decay_tau"),
                float(default_modulators.get("decay_tau", 1.0)),
            ),
            "deposit_sigma": _coerce_float(
                modulators_map.get("deposit_sigma"),
                float(default_modulators.get("deposit_sigma", 0.0)),
            ),
        },
        "synapse": {
            "backend": synapse_backend,
            "fused_layout": fused_layout,
            "ring_strategy": ring_strategy,
            "ring_dtype": ring_dtype,
            "store_sparse_by_delay": store_sparse_by_delay,
            "receptor_mode": receptor_mode,
        },
        "advanced_synapse": {
            "enabled": bool(
                advanced_synapse_map.get(
                    "enabled", default_advanced_synapse.get("enabled", False)
                )
            ),
            "conductance_mode": bool(
                advanced_synapse_map.get(
                    "conductance_mode",
                    default_advanced_synapse.get("conductance_mode", False),
                )
            ),
            "bio_synapse": bool(
                advanced_synapse_map.get(
                    "bio_synapse", default_advanced_synapse.get("bio_synapse", False)
                )
            ),
            "bio_nmda_block": bool(
                advanced_synapse_map.get(
                    "bio_nmda_block", default_advanced_synapse.get("bio_nmda_block", False)
                )
            ),
            "bio_stp": bool(
                advanced_synapse_map.get(
                    "bio_stp", default_advanced_synapse.get("bio_stp", False)
                )
            ),
            "nmda_voltage_block": bool(
                advanced_synapse_map.get(
                    "nmda_voltage_block",
                    default_advanced_synapse.get("nmda_voltage_block", False),
                )
            ),
            "nmda_mg_mM": _coerce_float(
                advanced_synapse_map.get("nmda_mg_mM"),
                float(default_advanced_synapse.get("nmda_mg_mM", 1.0)),
            ),
            "nmda_v_half_v": _coerce_float(
                advanced_synapse_map.get("nmda_v_half_v"),
                float(default_advanced_synapse.get("nmda_v_half_v", -0.020)),
            ),
            "nmda_slope_v": _coerce_float(
                advanced_synapse_map.get("nmda_slope_v"),
                float(default_advanced_synapse.get("nmda_slope_v", 0.016)),
            ),
            "stp_enabled": bool(
                advanced_synapse_map.get(
                    "stp_enabled", default_advanced_synapse.get("stp_enabled", False)
                )
            ),
            "stp_u": _coerce_float(
                advanced_synapse_map.get("stp_u"),
                float(default_advanced_synapse.get("stp_u", 0.2)),
            ),
            "stp_tau_rec_s": _coerce_float(
                advanced_synapse_map.get("stp_tau_rec_s"),
                float(default_advanced_synapse.get("stp_tau_rec_s", 0.2)),
            ),
            "stp_tau_facil_s": _coerce_float(
                advanced_synapse_map.get("stp_tau_facil_s"),
                float(default_advanced_synapse.get("stp_tau_facil_s", 0.0)),
            ),
            "stp_state_dtype": _first_non_none(
                advanced_synapse_map.get("stp_state_dtype"),
                default_advanced_synapse.get("stp_state_dtype", "float16"),
            ),
            "post_voltage_source": post_voltage_source,
            "reversal_potential_v": adv_reversal,
        },
        "wrapper": {
            "enabled": bool(wrapper_map.get("enabled", default_wrapper.get("enabled", False))),
            "ach_lr_gain": _coerce_float(
                wrapper_map.get("ach_lr_gain"),
                float(default_wrapper.get("ach_lr_gain", 0.0)),
            ),
            "ne_lr_gain": _coerce_float(
                wrapper_map.get("ne_lr_gain"),
                float(default_wrapper.get("ne_lr_gain", 0.0)),
            ),
            "ht_lr_gain": _coerce_float(
                wrapper_map.get("ht_lr_gain"),
                float(default_wrapper.get("ht_lr_gain", 0.0)),
            ),
            "ht_extra_weight_decay": _coerce_float(
                wrapper_map.get("ht_extra_weight_decay"),
                float(default_wrapper.get("ht_extra_weight_decay", 0.0)),
            ),
            "lr_clip_min": _coerce_float(
                wrapper_map.get("lr_clip_min"),
                float(default_wrapper.get("lr_clip_min", 0.1)),
            ),
            "lr_clip_max": _coerce_float(
                wrapper_map.get("lr_clip_max"),
                float(default_wrapper.get("lr_clip_max", 10.0)),
            ),
            "dopamine_baseline": _coerce_float(
                wrapper_map.get("dopamine_baseline"),
                float(default_wrapper.get("dopamine_baseline", 0.0)),
            ),
            "ach_baseline": _coerce_float(
                wrapper_map.get("ach_baseline"),
                float(default_wrapper.get("ach_baseline", 0.0)),
            ),
            "ne_baseline": _coerce_float(
                wrapper_map.get("ne_baseline"),
                float(default_wrapper.get("ne_baseline", 0.0)),
            ),
            "ht_baseline": _coerce_float(
                wrapper_map.get("ht_baseline"),
                float(default_wrapper.get("ht_baseline", 0.0)),
            ),
            "combine_mode": wrapper_combine_mode,
            "missing_modulators_policy": wrapper_missing_policy,
        },
        "excitability_modulation": {
            "enabled": bool(
                excitability_map.get("enabled", default_excitability.get("enabled", False))
            ),
            "targets": excitability_targets,
            "compartment": excitability_compartment,
            "ach_gain": _coerce_float(
                excitability_map.get("ach_gain"),
                float(default_excitability.get("ach_gain", 0.0)),
            ),
            "ne_gain": _coerce_float(
                excitability_map.get("ne_gain"),
                float(default_excitability.get("ne_gain", 0.0)),
            ),
            "ht_gain": _coerce_float(
                excitability_map.get("ht_gain"),
                float(default_excitability.get("ht_gain", 0.0)),
            ),
            "clamp_abs": _coerce_float(
                excitability_map.get("clamp_abs"),
                float(default_excitability.get("clamp_abs", 1.0)),
            ),
        },
        "homeostasis": {
            "enabled": bool(homeostasis_map.get("enabled", default_homeostasis.get("enabled", False))),
            "rule": _coerce_choice(
                homeostasis_map.get("rule"),
                allowed=ALLOWED_HOMEOSTASIS_RULE,
                default=str(default_homeostasis.get("rule", "rate_ema_threshold")),
            ),
            "alpha": _coerce_float(homeostasis_map.get("alpha"), float(default_homeostasis.get("alpha", 0.01))),
            "eta": _coerce_float(homeostasis_map.get("eta"), float(default_homeostasis.get("eta", 1e-3))),
            "r_target": _coerce_float(
                homeostasis_map.get("r_target"),
                float(default_homeostasis.get("r_target", 0.05)),
            ),
            "clamp_min": _coerce_float(
                homeostasis_map.get("clamp_min"),
                float(default_homeostasis.get("clamp_min", 0.0)),
            ),
            "clamp_max": _coerce_float(
                homeostasis_map.get("clamp_max"),
                float(default_homeostasis.get("clamp_max", 0.05)),
            ),
            "scope": _coerce_choice(
                homeostasis_map.get("scope"),
                allowed=ALLOWED_HOMEOSTASIS_SCOPE,
                default=str(default_homeostasis.get("scope", "per_neuron")),
            ),
        },
        "pruning": {
            "enabled": bool(pruning_map.get("enabled", default_pruning.get("enabled", False))),
            "prune_interval_steps": _coerce_positive_int(
                pruning_map.get("prune_interval_steps"),
                int(default_pruning.get("prune_interval_steps", 250)),
            ),
            "usage_alpha": _coerce_float(
                pruning_map.get("usage_alpha"),
                float(default_pruning.get("usage_alpha", 0.01)),
            ),
            "w_min": _coerce_float(pruning_map.get("w_min"), float(default_pruning.get("w_min", 0.05))),
            "usage_min": _coerce_float(
                pruning_map.get("usage_min"),
                float(default_pruning.get("usage_min", 0.01)),
            ),
            "k_min_out": _coerce_nonnegative_int(
                pruning_map.get("k_min_out"),
                int(default_pruning.get("k_min_out", 1)),
            ),
            "k_min_in": _coerce_nonnegative_int(
                pruning_map.get("k_min_in"),
                int(default_pruning.get("k_min_in", 1)),
            ),
            "max_prune_fraction_per_interval": _coerce_float(
                pruning_map.get("max_prune_fraction_per_interval"),
                float(default_pruning.get("max_prune_fraction_per_interval", 0.10)),
            ),
            "verbose": bool(pruning_map.get("verbose", default_pruning.get("verbose", False))),
        },
        "neurogenesis": {
            "enabled": bool(neurogenesis_map.get("enabled", default_neurogenesis.get("enabled", False))),
            "growth_interval_steps": _coerce_positive_int(
                neurogenesis_map.get("growth_interval_steps"),
                int(default_neurogenesis.get("growth_interval_steps", 500)),
            ),
            "add_neurons_per_event": _coerce_positive_int(
                neurogenesis_map.get("add_neurons_per_event"),
                int(default_neurogenesis.get("add_neurons_per_event", 4)),
            ),
            "newborn_plasticity_multiplier": _coerce_float(
                neurogenesis_map.get("newborn_plasticity_multiplier"),
                float(default_neurogenesis.get("newborn_plasticity_multiplier", 1.5)),
            ),
            "newborn_duration_steps": _coerce_positive_int(
                neurogenesis_map.get("newborn_duration_steps"),
                int(default_neurogenesis.get("newborn_duration_steps", 250)),
            ),
            "max_total_neurons": _coerce_positive_int(
                neurogenesis_map.get("max_total_neurons"),
                int(default_neurogenesis.get("max_total_neurons", 20000)),
            ),
            "verbose": bool(neurogenesis_map.get("verbose", default_neurogenesis.get("verbose", False))),
        },
        "logic": {
            "learn_every": _coerce_positive_int(
                _first_non_none(logic_map.get("learn_every"), merged.get("logic_learn_every")),
                _coerce_positive_int(default_logic.get("learn_every"), 1),
            ),
            "reward_delivery_steps": _coerce_nonnegative_int(
                _first_non_none(
                    logic_map.get("reward_delivery_steps"),
                    merged.get("reward_delivery_steps"),
                ),
                _coerce_nonnegative_int(default_logic.get("reward_delivery_steps"), 2),
            ),
            "reward_delivery_clamp_input": bool(reward_clamp_input),
            "exploration": {
                "enabled": bool(
                    _first_non_none(
                        exploration_map.get("enabled"),
                        default_exploration.get("enabled", True),
                    )
                ),
                "mode": exploration_mode,
                "epsilon_start": _coerce_float(
                    _first_non_none(
                        exploration_map.get("epsilon_start"),
                        default_exploration.get("epsilon_start", 0.20),
                    ),
                    float(default_exploration.get("epsilon_start", 0.20)),
                ),
                "epsilon_end": _coerce_float(
                    _first_non_none(
                        exploration_map.get("epsilon_end"),
                        default_exploration.get("epsilon_end", 0.01),
                    ),
                    float(default_exploration.get("epsilon_end", 0.01)),
                ),
                "epsilon_decay_trials": _coerce_positive_int(
                    _first_non_none(
                        exploration_map.get("epsilon_decay_trials"),
                        default_exploration.get("epsilon_decay_trials", 3000),
                    ),
                    int(default_exploration.get("epsilon_decay_trials", 3000)),
                ),
                "tie_break": exploration_tie_break,
                "seed": _coerce_nonnegative_int(
                    _first_non_none(
                        exploration_map.get("seed"),
                        default_exploration.get("seed", 123),
                    ),
                    int(default_exploration.get("seed", 123)),
                ),
            },
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
    if resolved["modulators"]["diffusion"] < 0:
        resolved["modulators"]["diffusion"] = 0.0
    if resolved["modulators"]["decay_tau"] <= 0:
        resolved["modulators"]["decay_tau"] = 1.0
    if resolved["modulators"]["deposit_sigma"] < 0:
        resolved["modulators"]["deposit_sigma"] = 0.0
    if resolved["advanced_synapse"]["stp_tau_rec_s"] < 0:
        resolved["advanced_synapse"]["stp_tau_rec_s"] = 0.0
    if resolved["advanced_synapse"]["stp_tau_facil_s"] < 0:
        resolved["advanced_synapse"]["stp_tau_facil_s"] = 0.0
    if resolved["wrapper"]["lr_clip_max"] < resolved["wrapper"]["lr_clip_min"]:
        resolved["wrapper"]["lr_clip_max"] = resolved["wrapper"]["lr_clip_min"]
    if resolved["excitability_modulation"]["clamp_abs"] < 0:
        resolved["excitability_modulation"]["clamp_abs"] = abs(
            float(resolved["excitability_modulation"]["clamp_abs"])
        )
    if resolved["homeostasis"]["alpha"] <= 0 or resolved["homeostasis"]["alpha"] > 1.0:
        resolved["homeostasis"]["alpha"] = float(default_homeostasis.get("alpha", 0.01))
    if resolved["homeostasis"]["eta"] < 0:
        resolved["homeostasis"]["eta"] = float(default_homeostasis.get("eta", 1e-3))
    if resolved["homeostasis"]["clamp_max"] < resolved["homeostasis"]["clamp_min"]:
        resolved["homeostasis"]["clamp_max"] = resolved["homeostasis"]["clamp_min"]
    if resolved["pruning"]["usage_alpha"] < 0:
        resolved["pruning"]["usage_alpha"] = float(default_pruning.get("usage_alpha", 0.01))
    if resolved["pruning"]["w_min"] < 0:
        resolved["pruning"]["w_min"] = float(default_pruning.get("w_min", 0.05))
    if resolved["pruning"]["usage_min"] < 0:
        resolved["pruning"]["usage_min"] = float(default_pruning.get("usage_min", 0.01))
    if resolved["pruning"]["max_prune_fraction_per_interval"] < 0:
        resolved["pruning"]["max_prune_fraction_per_interval"] = 0.0
    if resolved["pruning"]["max_prune_fraction_per_interval"] > 1.0:
        resolved["pruning"]["max_prune_fraction_per_interval"] = 1.0
    if resolved["neurogenesis"]["newborn_plasticity_multiplier"] <= 0:
        resolved["neurogenesis"]["newborn_plasticity_multiplier"] = float(
            default_neurogenesis.get("newborn_plasticity_multiplier", 1.5)
        )
    logic_exploration = cast(Mapping[str, Any], resolved.get("logic", {}).get("exploration", {}))
    if float(logic_exploration.get("epsilon_start", 0.0)) < 0.0:
        resolved["logic"]["exploration"]["epsilon_start"] = 0.0
    if float(logic_exploration.get("epsilon_end", 0.0)) < 0.0:
        resolved["logic"]["exploration"]["epsilon_end"] = 0.0

    # Demo-specific behavior defaults.
    if demo_id in {"learning_gate", "dopamine_plasticity"}:
        resolved["learning"]["enabled"] = True
    if demo_id == "dopamine_plasticity":
        resolved["modulators"]["enabled"] = True
        if not resolved["modulators"]["kinds"]:
            resolved["modulators"]["kinds"] = ["dopamine"]
    if demo_id == "pruning_sparse":
        resolved["pruning"]["enabled"] = True
    if demo_id == "neurogenesis_sparse":
        resolved["neurogenesis"]["enabled"] = True
    if demo_id == "logic_curriculum" or demo_id in _LOGIC_DEMO_TO_GATE:
        resolved["logic_backend"] = _coerce_choice(
            merged.get("logic_backend"),
            allowed=ALLOWED_LOGIC_BACKEND,
            default="harness",
        )
    if demo_id in _LOGIC_DEMO_TO_GATE:
        resolved["logic_gate"] = _coerce_choice(
            merged.get("logic_gate"),
            allowed=_LOGIC_GATE_VALUES,
            default=_LOGIC_DEMO_TO_GATE[demo_id],
        )
        learning_rule = str(resolved["learning"]["rule"]).strip().lower()
        inferred_mode = "none"
        if bool(resolved["learning"]["enabled"]):
            inferred_mode = "surrogate" if learning_rule == "surrogate" else "rstdp"
        resolved["logic_learning_mode"] = _coerce_choice(
            payload.get("logic_learning_mode"),
            allowed=_LOGIC_LEARNING_MODES,
            default=inferred_mode,
        )
        if resolved["logic_learning_mode"] == "none":
            resolved["learning"]["enabled"] = False
        elif resolved["logic_learning_mode"] == "surrogate":
            resolved["learning"]["enabled"] = True
            resolved["learning"]["rule"] = "surrogate"
        else:
            resolved["learning"]["enabled"] = True
            if str(resolved["learning"]["rule"]).strip().lower() == "surrogate":
                resolved["learning"]["rule"] = "rstdp"
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
        resolved["logic_neuron_model"] = _coerce_choice(
            merged.get("logic_neuron_model"),
            allowed=_LOGIC_NEURON_MODELS,
            default="adex_3c",
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
        learning_rule = str(resolved["learning"]["rule"]).strip().lower()
        inferred_mode = "none"
        if bool(resolved["learning"]["enabled"]):
            inferred_mode = "surrogate" if learning_rule == "surrogate" else "rstdp"
        resolved["logic_learning_mode"] = _coerce_choice(
            payload.get("logic_learning_mode"),
            allowed=_LOGIC_LEARNING_MODES,
            default=inferred_mode,
        )
        if resolved["logic_learning_mode"] == "none":
            resolved["learning"]["enabled"] = False
        elif resolved["logic_learning_mode"] == "surrogate":
            resolved["learning"]["enabled"] = True
            resolved["learning"]["rule"] = "surrogate"
        else:
            resolved["learning"]["enabled"] = True
            if str(resolved["learning"]["rule"]).strip().lower() == "surrogate":
                resolved["learning"]["rule"] = "rstdp"
        resolved["logic_neuron_model"] = _coerce_choice(
            merged.get("logic_neuron_model"),
            allowed=_LOGIC_NEURON_MODELS,
            default="adex_3c",
        )
        if resolved.get("logic_backend") == "engine":
            payload_learning = cast(
                Mapping[str, Any], payload.get("learning", {})
            ) if isinstance(payload.get("learning"), Mapping) else {}
            payload_modulators = cast(
                Mapping[str, Any], payload.get("modulators", {})
            ) if isinstance(payload.get("modulators"), Mapping) else {}
            if "lr" not in payload_learning:
                resolved["learning"]["lr"] = 1e-3
            if "amount" not in payload_modulators:
                resolved["modulators"]["amount"] = 0.10
            if "decay_tau" not in payload_modulators:
                resolved["modulators"]["decay_tau"] = 0.05
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
        "receptor_mode": getattr(args, "receptor_mode", "exc_only"),
        "delay_steps": getattr(args, "delay_steps", 3),
        "monitor_mode": getattr(args, "mode", "dashboard"),
        "monitors_enabled": bool(getattr(args, "monitors", True)),
        "learning": {
            "enabled": False,
            "rule": "three_factor_hebbian",
            "lr": getattr(args, "learning_lr", 0.1),
        },
        "synapse": {
            "backend": getattr(args, "synapse_backend", "spmm_fused"),
            "fused_layout": getattr(args, "fused_layout", "auto"),
            "ring_strategy": getattr(args, "ring_strategy", "dense"),
            "ring_dtype": getattr(args, "ring_dtype", None) or "none",
            "store_sparse_by_delay": getattr(args, "store_sparse_by_delay", None),
            "receptor_mode": getattr(args, "receptor_mode", "exc_only"),
        },
        "advanced_synapse": {
            "enabled": bool(getattr(args, "logic_adv_synapse_enabled", False)),
            "conductance_mode": bool(getattr(args, "logic_adv_synapse_conductance_mode", False)),
            "nmda_voltage_block": bool(getattr(args, "logic_adv_synapse_nmda_block", False)),
            "stp_enabled": bool(getattr(args, "logic_adv_synapse_stp_enabled", False)),
        },
        "modulators": {
            "enabled": bool(getattr(args, "modulators_enabled", False)),
            "kinds": _coerce_string_list(getattr(args, "modulator_kinds", "")),
            "pulse_step": getattr(args, "da_step", 50),
            "amount": getattr(args, "da_amount", 1.0),
            "field_type": getattr(args, "modulator_field_type", "global_scalar"),
            "grid_size": getattr(args, "modulator_grid_size", "16x16"),
            "world_extent": getattr(args, "modulator_world_extent", "1.0,1.0"),
            "diffusion": getattr(args, "modulator_diffusion", 0.0),
            "decay_tau": getattr(args, "modulator_decay_tau", 1.0),
            "deposit_sigma": getattr(args, "modulator_deposit_sigma", 0.0),
        },
        "wrapper": {
            "enabled": bool(getattr(args, "logic_wrapper_enabled", False)),
            "ach_lr_gain": getattr(args, "logic_wrapper_ach_lr_gain", 0.0),
            "ne_lr_gain": getattr(args, "logic_wrapper_ne_lr_gain", 0.0),
            "ht_extra_weight_decay": getattr(args, "logic_wrapper_ht_extra_weight_decay", 0.0),
        },
        "excitability_modulation": {
            "enabled": bool(getattr(args, "logic_excitability_enabled", False)),
            "ach_gain": getattr(args, "logic_excitability_ach_gain", 0.0),
            "ne_gain": getattr(args, "logic_excitability_ne_gain", 0.0),
            "ht_gain": getattr(args, "logic_excitability_ht_gain", 0.0),
        },
        "homeostasis": {
            "enabled": bool(getattr(args, "enable_homeostasis", False)),
            "rule": "rate_ema_threshold",
            "alpha": getattr(args, "homeostasis_alpha", 0.01),
            "eta": getattr(args, "homeostasis_eta", 1e-3),
            "r_target": getattr(args, "homeostasis_r_target", 0.05),
            "clamp_min": getattr(args, "homeostasis_clamp_min", 0.0),
            "clamp_max": getattr(args, "homeostasis_clamp_max", 0.05),
            "scope": getattr(args, "homeostasis_scope", "per_neuron"),
        },
        "pruning": {
            "enabled": bool(getattr(args, "enable_pruning", False)),
            "prune_interval_steps": getattr(args, "prune_interval_steps", 250),
            "usage_alpha": getattr(args, "prune_usage_alpha", 0.01),
            "w_min": getattr(args, "prune_w_min", 0.05),
            "usage_min": getattr(args, "prune_usage_min", 0.01),
            "k_min_out": getattr(args, "prune_k_min_out", 1),
            "k_min_in": getattr(args, "prune_k_min_in", 1),
            "max_prune_fraction_per_interval": getattr(args, "prune_max_fraction", 0.10),
            "verbose": bool(getattr(args, "prune_verbose", False)),
        },
        "neurogenesis": {
            "enabled": bool(getattr(args, "enable_neurogenesis", False)),
            "growth_interval_steps": getattr(args, "growth_interval_steps", 500),
            "add_neurons_per_event": getattr(args, "add_neurons_per_event", 4),
            "newborn_plasticity_multiplier": getattr(args, "newborn_plasticity_multiplier", 1.5),
            "newborn_duration_steps": getattr(args, "newborn_duration_steps", 250),
            "max_total_neurons": getattr(args, "max_total_neurons", 20000),
            "verbose": bool(getattr(args, "neurogenesis_verbose", False)),
        },
        "logic": {
            "learn_every": getattr(args, "logic_learn_every", 1),
            "reward_delivery_steps": getattr(args, "logic_reward_delivery_steps", 2),
            "reward_delivery_clamp_input": bool(
                getattr(args, "logic_reward_delivery_clamp_input", True)
            ),
            "exploration": {
                "enabled": bool(getattr(args, "logic_exploration_enabled", True)),
                "mode": "epsilon_greedy",
                "epsilon_start": getattr(args, "logic_epsilon_start", 0.20),
                "epsilon_end": getattr(args, "logic_epsilon_end", 0.01),
                "epsilon_decay_trials": getattr(args, "logic_epsilon_decay_trials", 3000),
                "tie_break": getattr(args, "logic_tie_break", "random_among_max"),
                "seed": getattr(args, "logic_exploration_seed", 123),
            },
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
        raw["logic_neuron_model"] = _coerce_choice(
            getattr(args, "logic_neuron_model", None),
            allowed=_LOGIC_NEURON_MODELS,
            default="adex_3c",
        )
        raw["logic_debug"] = bool(getattr(args, "logic_debug", False))
        raw["logic_debug_every"] = max(1, int(getattr(args, "logic_debug_every", 25)))
        raw["logic_backend"] = _coerce_choice(
            getattr(args, "logic_backend", None),
            allowed=ALLOWED_LOGIC_BACKEND,
            default="harness",
        )
    if demo_name == "logic_curriculum":
        logic_mode = str(getattr(args, "logic_learning_mode", "rstdp")).strip().lower()
        if logic_mode not in _LOGIC_LEARNING_MODES:
            logic_mode = "rstdp"
        raw["learning"]["enabled"] = logic_mode != "none"
        raw["learning"]["rule"] = "surrogate" if logic_mode == "surrogate" else "rstdp"
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
        raw["logic_neuron_model"] = _coerce_choice(
            getattr(args, "logic_neuron_model", None),
            allowed=_LOGIC_NEURON_MODELS,
            default="adex_3c",
        )
        raw["logic_curriculum_gates"] = str(
            getattr(args, "logic_curriculum_gates", _LOGIC_CURRICULUM_DEFAULT)
        )
        raw["logic_curriculum_replay_ratio"] = float(
            getattr(args, "logic_curriculum_replay_ratio", 0.35)
        )
        raw["logic_debug"] = bool(getattr(args, "logic_debug", False))
        raw["logic_debug_every"] = max(1, int(getattr(args, "logic_debug_every", 25)))
        raw["logic_backend"] = _coerce_choice(
            getattr(args, "logic_backend", None),
            allowed=ALLOWED_LOGIC_BACKEND,
            default="harness",
        )
    return resolve_run_spec(raw)


def run_spec_to_cli_args(
    *,
    run_spec: Mapping[str, Any],
    run_id: str,
    artifacts_dir: Path,
) -> list[str]:
    spec = resolve_run_spec(run_spec)
    synapse_cfg = cast(Mapping[str, Any], spec.get("synapse", {}))
    advanced_synapse_cfg = cast(Mapping[str, Any], spec.get("advanced_synapse", {}))
    modulators_cfg = cast(Mapping[str, Any], spec.get("modulators", {}))
    wrapper_cfg = cast(Mapping[str, Any], spec.get("wrapper", {}))
    excitability_cfg = cast(Mapping[str, Any], spec.get("excitability_modulation", {}))
    homeostasis_cfg = cast(Mapping[str, Any], spec.get("homeostasis", {}))
    pruning_cfg = cast(Mapping[str, Any], spec.get("pruning", {}))
    neurogenesis_cfg = cast(Mapping[str, Any], spec.get("neurogenesis", {}))
    logic_cfg = cast(Mapping[str, Any], spec.get("logic", {}))
    exploration_cfg = cast(Mapping[str, Any], logic_cfg.get("exploration", {}))

    grid_size = _coerce_grid_size(modulators_cfg.get("grid_size"), (16, 16))
    world_extent = _coerce_float_pair(modulators_cfg.get("world_extent"), (1.0, 1.0))
    ring_dtype = _coerce_choice(
        _first_non_none(synapse_cfg.get("ring_dtype"), spec.get("ring_dtype")),
        allowed=ALLOWED_RING_DTYPE,
        default="none",
    )
    store_sparse = _coerce_optional_bool(
        _first_non_none(synapse_cfg.get("store_sparse_by_delay"), spec.get("store_sparse_by_delay"))
    )

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
        str(synapse_cfg.get("fused_layout", spec["fused_layout"])),
        "--synapse-backend",
        str(synapse_cfg.get("backend", spec["synapse_backend"])),
        "--ring-strategy",
        str(synapse_cfg.get("ring_strategy", spec["ring_strategy"])),
        "--receptor-mode",
        str(synapse_cfg.get("receptor_mode", "exc_only")),
        "--max-ring-mib",
        str(float(spec["max_ring_mib"])),
        "--delay_steps",
        str(int(spec["delay_steps"])),
        "--learning_lr",
        str(float(spec["learning"]["lr"])),
        "--da_amount",
        str(float(modulators_cfg.get("amount", 1.0))),
        "--da_step",
        str(int(modulators_cfg.get("pulse_step", 50))),
        "--modulator-kinds",
        ",".join(_coerce_string_list(modulators_cfg.get("kinds"))),
        "--modulator-field-type",
        str(modulators_cfg.get("field_type", "global_scalar")),
        "--modulator-grid-size",
        f"{int(grid_size[0])}x{int(grid_size[1])}",
        "--modulator-world-extent",
        f"{float(world_extent[0])},{float(world_extent[1])}",
        "--modulator-diffusion",
        str(float(modulators_cfg.get("diffusion", 0.0))),
        "--modulator-decay-tau",
        str(float(modulators_cfg.get("decay_tau", 1.0))),
        "--modulator-deposit-sigma",
        str(float(modulators_cfg.get("deposit_sigma", 0.0))),
        "--logic-wrapper-ach-lr-gain",
        str(float(wrapper_cfg.get("ach_lr_gain", 0.0))),
        "--logic-wrapper-ne-lr-gain",
        str(float(wrapper_cfg.get("ne_lr_gain", 0.0))),
        "--logic-wrapper-ht-extra-weight-decay",
        str(float(wrapper_cfg.get("ht_extra_weight_decay", 0.0))),
        "--logic-excitability-ach-gain",
        str(float(excitability_cfg.get("ach_gain", 0.0))),
        "--logic-excitability-ne-gain",
        str(float(excitability_cfg.get("ne_gain", 0.0))),
        "--logic-excitability-ht-gain",
        str(float(excitability_cfg.get("ht_gain", 0.0))),
        "--homeostasis-alpha",
        str(float(homeostasis_cfg.get("alpha", 0.01))),
        "--homeostasis-eta",
        str(float(homeostasis_cfg.get("eta", 1e-3))),
        "--homeostasis-r-target",
        str(float(homeostasis_cfg.get("r_target", 0.05))),
        "--homeostasis-clamp-min",
        str(float(homeostasis_cfg.get("clamp_min", 0.0))),
        "--homeostasis-clamp-max",
        str(float(homeostasis_cfg.get("clamp_max", 0.05))),
        "--homeostasis-scope",
        str(homeostasis_cfg.get("scope", "per_neuron")),
        "--prune-interval-steps",
        str(int(pruning_cfg.get("prune_interval_steps", 250))),
        "--prune-usage-alpha",
        str(float(pruning_cfg.get("usage_alpha", 0.01))),
        "--prune-w-min",
        str(float(pruning_cfg.get("w_min", 0.05))),
        "--prune-usage-min",
        str(float(pruning_cfg.get("usage_min", 0.01))),
        "--prune-k-min-out",
        str(int(pruning_cfg.get("k_min_out", 1))),
        "--prune-k-min-in",
        str(int(pruning_cfg.get("k_min_in", 1))),
        "--prune-max-fraction",
        str(float(pruning_cfg.get("max_prune_fraction_per_interval", 0.10))),
        "--growth-interval-steps",
        str(int(neurogenesis_cfg.get("growth_interval_steps", 500))),
        "--add-neurons-per-event",
        str(int(neurogenesis_cfg.get("add_neurons_per_event", 4))),
        "--newborn-plasticity-multiplier",
        str(float(neurogenesis_cfg.get("newborn_plasticity_multiplier", 1.5))),
        "--newborn-duration-steps",
        str(int(neurogenesis_cfg.get("newborn_duration_steps", 250))),
        "--max-total-neurons",
        str(int(neurogenesis_cfg.get("max_total_neurons", 20000))),
        "--logic-learn-every",
        str(int(logic_cfg.get("learn_every", 1))),
        "--logic-reward-delivery-steps",
        str(int(logic_cfg.get("reward_delivery_steps", 2))),
        "--logic-epsilon-start",
        str(float(exploration_cfg.get("epsilon_start", 0.20))),
        "--logic-epsilon-end",
        str(float(exploration_cfg.get("epsilon_end", 0.01))),
        "--logic-epsilon-decay-trials",
        str(int(exploration_cfg.get("epsilon_decay_trials", 3000))),
        "--logic-tie-break",
        str(exploration_cfg.get("tie_break", "random_among_max")),
        "--logic-exploration-seed",
        str(int(exploration_cfg.get("seed", 123))),
        "--run-id",
        run_id,
        "--artifacts-dir",
        str(artifacts_dir),
        "--no-open",
        "--no-server",
    ]
    if not bool(spec.get("monitors_enabled", True)):
        args.append("--no-monitors")
    seed = spec.get("seed")
    if seed is not None:
        args.extend(["--seed", str(int(seed))])
    if ring_dtype in {"float32", "float16", "bfloat16"}:
        args.extend(["--ring-dtype", ring_dtype])
    if store_sparse is not None:
        args.extend(["--store-sparse-by-delay", "true" if bool(store_sparse) else "false"])
    args.append(
        "--modulators-enabled"
        if bool(modulators_cfg.get("enabled", False))
        else "--no-modulators-enabled"
    )
    args.append(
        "--logic-adv-synapse-enabled"
        if bool(advanced_synapse_cfg.get("enabled", False))
        else "--no-logic-adv-synapse-enabled"
    )
    args.append(
        "--logic-adv-synapse-conductance-mode"
        if bool(advanced_synapse_cfg.get("conductance_mode", False))
        else "--no-logic-adv-synapse-conductance-mode"
    )
    args.append(
        "--logic-adv-synapse-nmda-block"
        if bool(advanced_synapse_cfg.get("nmda_voltage_block", False))
        else "--no-logic-adv-synapse-nmda-block"
    )
    args.append(
        "--logic-adv-synapse-stp-enabled"
        if bool(advanced_synapse_cfg.get("stp_enabled", False))
        else "--no-logic-adv-synapse-stp-enabled"
    )
    args.append(
        "--logic-wrapper-enabled"
        if bool(wrapper_cfg.get("enabled", False))
        else "--no-logic-wrapper-enabled"
    )
    args.append(
        "--logic-excitability-enabled"
        if bool(excitability_cfg.get("enabled", False))
        else "--no-logic-excitability-enabled"
    )
    args.append("--enable-homeostasis" if bool(homeostasis_cfg.get("enabled", False)) else "--no-enable-homeostasis")
    args.append("--enable-pruning" if bool(pruning_cfg.get("enabled", False)) else "--no-enable-pruning")
    args.append("--prune-verbose" if bool(pruning_cfg.get("verbose", False)) else "--no-prune-verbose")
    args.append(
        "--enable-neurogenesis"
        if bool(neurogenesis_cfg.get("enabled", False))
        else "--no-enable-neurogenesis"
    )
    args.append(
        "--neurogenesis-verbose"
        if bool(neurogenesis_cfg.get("verbose", False))
        else "--no-neurogenesis-verbose"
    )
    args.append(
        "--logic-reward-delivery-clamp-input"
        if bool(logic_cfg.get("reward_delivery_clamp_input", True))
        else "--no-logic-reward-delivery-clamp-input"
    )
    args.append(
        "--logic-exploration-enabled"
        if bool(exploration_cfg.get("enabled", True))
        else "--no-logic-exploration-enabled"
    )
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
                "--logic-backend",
                str(spec.get("logic_backend", "harness")),
                "--logic-learning-mode",
                logic_mode,
                "--logic-sim-steps-per-trial",
                str(int(spec.get("logic_sim_steps_per_trial", 10))),
                "--logic-sampling-method",
                str(spec.get("logic_sampling_method", "sequential")),
                "--logic-neuron-model",
                str(spec.get("logic_neuron_model", "adex_3c")),
                "--logic-debug-every",
                str(int(spec.get("logic_debug_every", 25))),
            ]
        )
        if bool(spec.get("logic_debug", False)):
            args.append("--logic-debug")
    if demo_id == "logic_curriculum":
        if not bool(spec["learning"]["enabled"]):
            logic_mode = "none"
        else:
            learning_rule = str(spec["learning"]["rule"]).strip().lower()
            logic_mode = "surrogate" if learning_rule == "surrogate" else "rstdp"
        args.extend(
            [
                "--logic-curriculum-gates",
                str(spec.get("logic_curriculum_gates", _LOGIC_CURRICULUM_DEFAULT)),
                "--logic-curriculum-replay-ratio",
                str(float(spec.get("logic_curriculum_replay_ratio", 0.35))),
                "--logic-backend",
                str(spec.get("logic_backend", "harness")),
                "--logic-learning-mode",
                logic_mode,
                "--logic-sim-steps-per-trial",
                str(int(spec.get("logic_sim_steps_per_trial", 10))),
                "--logic-sampling-method",
                str(spec.get("logic_sampling_method", "sequential")),
                "--logic-neuron-model",
                str(spec.get("logic_neuron_model", "adex_3c")),
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
    is_logic_demo = demo_id == "logic_curriculum" or demo_id in _LOGIC_DEMO_TO_GATE
    logic_backend: RunLogicBackend | None = None
    if is_logic_demo:
        logic_backend = cast(
            RunLogicBackend,
            _coerce_choice(
                spec.get("logic_backend"),
                allowed=ALLOWED_LOGIC_BACKEND,
                default="harness",
            ),
        )

    known_max_delay: int | None
    if demo_id == "propagation_impulse":
        known_max_delay = 0
    elif demo_id == "delay_impulse":
        known_max_delay = int(spec["delay_steps"])
    elif demo_id in {"learning_gate", "dopamine_plasticity"}:
        known_max_delay = 0
    elif is_logic_demo and logic_backend == "engine":
        known_max_delay = int(spec["delay_steps"])
    elif is_logic_demo:
        known_max_delay = None
    else:
        known_max_delay = None

    delays_enabled = known_max_delay is None or known_max_delay > 0
    if is_logic_demo and logic_backend == "harness":
        delays_enabled = False

    logic_uses_internal_dopamine = (
        is_logic_demo
        and logic_backend == "harness"
        and bool(spec["learning"]["enabled"])
        and str(spec["learning"]["rule"]).strip().lower() == "rstdp"
    )
    if is_logic_demo and logic_backend == "harness":
        mod_kinds: list[str] = []
    else:
        mod_kinds = [str(kind).strip() for kind in spec["modulators"]["kinds"] if str(kind).strip()]
    if logic_uses_internal_dopamine and "dopamine" not in mod_kinds:
        mod_kinds.append("dopamine")
    if is_logic_demo and logic_backend == "harness":
        mod_enabled = logic_uses_internal_dopamine
        mod_pulse_step: int | None = None
        mod_amount: float | None = None
        mod_field_type: RunModulatorFieldType | None = None
        mod_grid_size: list[int] | None = None
        mod_world_extent: list[float] | None = None
        mod_diffusion: float | None = None
        mod_decay_tau: float | None = None
        mod_deposit_sigma: float | None = None
    else:
        mod_enabled = bool(spec["modulators"]["enabled"]) or logic_uses_internal_dopamine
        mod_pulse_step = int(spec["modulators"]["pulse_step"])
        mod_amount = float(spec["modulators"]["amount"])
        mod_field_type = cast(
            RunModulatorFieldType,
            _coerce_choice(
                spec["modulators"].get("field_type"),
                allowed=ALLOWED_MODULATOR_FIELD_TYPE,
                default="global_scalar",
            ),
        )
        grid_size = _coerce_grid_size(spec["modulators"].get("grid_size"), (16, 16))
        world_extent = _coerce_float_pair(spec["modulators"].get("world_extent"), (1.0, 1.0))
        mod_grid_size = [int(grid_size[0]), int(grid_size[1])]
        mod_world_extent = [float(world_extent[0]), float(world_extent[1])]
        mod_diffusion = float(spec["modulators"].get("diffusion", 0.0))
        mod_decay_tau = float(spec["modulators"].get("decay_tau", 1.0))
        mod_deposit_sigma = float(spec["modulators"].get("deposit_sigma", 0.0))

    if is_logic_demo and logic_backend == "harness":
        ring_len: int | None = None
    else:
        ring_len = (known_max_delay + 1) if known_max_delay is not None else None

    if is_logic_demo and logic_backend == "harness":
        synapse_backend: RunSynapseBackend | None = None
        fused_layout: RunFusedLayout | None = None
        ring_strategy: RunRingStrategy | None = None
        ring_dtype: RunRingDtype | None = None
        store_sparse_by_delay: bool | None = None
        receptor_mode: RunSynapseReceptorMode | None = None
    else:
        synapse_cfg = cast(Mapping[str, Any], spec.get("synapse", {}))
        synapse_backend = cast(
            RunSynapseBackend,
            _coerce_choice(
                _first_non_none(synapse_cfg.get("backend"), spec["synapse_backend"]),
                allowed=ALLOWED_SYNAPSE_BACKEND,
                default="spmm_fused",
            ),
        )
        fused_layout = cast(
            RunFusedLayout,
            _coerce_choice(
                _first_non_none(synapse_cfg.get("fused_layout"), spec["fused_layout"]),
                allowed=ALLOWED_FUSED_LAYOUT,
                default="auto",
            ),
        )
        ring_strategy = cast(
            RunRingStrategy,
            _coerce_choice(
                _first_non_none(synapse_cfg.get("ring_strategy"), spec["ring_strategy"]),
                allowed=ALLOWED_RING_STRATEGY,
                default="dense",
            ),
        )
        ring_dtype = cast(
            RunRingDtype,
            _coerce_choice(
                _first_non_none(synapse_cfg.get("ring_dtype"), spec["ring_dtype"]),
                allowed=ALLOWED_RING_DTYPE,
                default="none",
            ),
        )
        store_sparse_by_delay = _coerce_optional_bool(
            _first_non_none(synapse_cfg.get("store_sparse_by_delay"), spec["store_sparse_by_delay"])
        )
        receptor_mode = cast(
            RunSynapseReceptorMode,
            _coerce_choice(
                synapse_cfg.get("receptor_mode"),
                allowed=ALLOWED_SYNAPSE_RECEPTOR_MODE,
                default="exc_only",
            ),
        )

    monitor_mode = cast(RunMonitorMode, spec["monitor_mode"])
    monitors_enabled = bool(spec.get("monitors_enabled", True))
    monitor_policy = "disabled"
    if monitors_enabled:
        monitor_policy = "sync_opt_in" if monitor_mode == "dashboard" else "cuda_safe"
    logic_cfg = cast(Mapping[str, Any], spec.get("logic", {}))
    exploration_cfg = cast(Mapping[str, Any], logic_cfg.get("exploration", {}))
    exploration_enabled = bool(exploration_cfg.get("enabled", False)) if is_logic_demo and logic_backend == "engine" else False
    reward_delivery_steps = int(logic_cfg.get("reward_delivery_steps", 0)) if is_logic_demo and logic_backend == "engine" else 0
    return {
        "demo_id": demo_id,
        "learning": {
            "enabled": bool(spec["learning"]["enabled"]),
            "rule": spec["learning"]["rule"] if spec["learning"]["enabled"] else None,
            "lr": float(spec["learning"]["lr"]) if spec["learning"]["enabled"] else None,
        },
        "delays": {
            "enabled": delays_enabled,
            "max_delay_steps": known_max_delay,
            "ring_len": ring_len,
        },
        "modulators": {
            "enabled": mod_enabled,
            "kinds": mod_kinds,
            "pulse_step": mod_pulse_step,
            "amount": mod_amount,
            "field_type": mod_field_type,
            "grid_size": mod_grid_size,
            "world_extent": mod_world_extent,
            "diffusion": mod_diffusion,
            "decay_tau": mod_decay_tau,
            "deposit_sigma": mod_deposit_sigma,
        },
        "synapse": {
            "backend": synapse_backend,
            "fused_layout": fused_layout,
            "ring_strategy": ring_strategy,
            "ring_dtype": ring_dtype,
            "store_sparse_by_delay": store_sparse_by_delay,
            "receptor_mode": receptor_mode,
        },
        "advanced_synapse": {
            "enabled": bool(spec.get("advanced_synapse", {}).get("enabled", False)),
            "conductance_mode": bool(
                spec.get("advanced_synapse", {}).get("conductance_mode", False)
            ),
            "nmda_voltage_block": bool(
                spec.get("advanced_synapse", {}).get("nmda_voltage_block", False)
            ),
            "stp_enabled": bool(spec.get("advanced_synapse", {}).get("stp_enabled", False)),
        },
        "wrapper": {
            "enabled": bool(spec.get("wrapper", {}).get("enabled", False)),
            "ach_lr_gain": float(spec.get("wrapper", {}).get("ach_lr_gain", 0.0)),
            "ne_lr_gain": float(spec.get("wrapper", {}).get("ne_lr_gain", 0.0)),
            "ht_extra_weight_decay": float(
                spec.get("wrapper", {}).get("ht_extra_weight_decay", 0.0)
            ),
        },
        "excitability_modulation": {
            "enabled": bool(spec.get("excitability_modulation", {}).get("enabled", False)),
            "ach_gain": float(spec.get("excitability_modulation", {}).get("ach_gain", 0.0)),
            "ne_gain": float(spec.get("excitability_modulation", {}).get("ne_gain", 0.0)),
            "ht_gain": float(spec.get("excitability_modulation", {}).get("ht_gain", 0.0)),
        },
        "monitor": {
            "enabled": monitors_enabled,
            "mode": monitor_mode,
            "sync_policy": monitor_policy,
        },
        "exploration": {
            "enabled": exploration_enabled,
            "mode": str(exploration_cfg.get("mode", "epsilon_greedy")),
            "epsilon_start": float(exploration_cfg.get("epsilon_start", 0.20)),
            "epsilon_end": float(exploration_cfg.get("epsilon_end", 0.01)),
            "epsilon_decay_trials": int(exploration_cfg.get("epsilon_decay_trials", 3000)),
            "tie_break": str(exploration_cfg.get("tie_break", "random_among_max")),
        },
        "reward_window": {
            "steps": reward_delivery_steps,
            "clamp_input": bool(logic_cfg.get("reward_delivery_clamp_input", True))
            if reward_delivery_steps > 0
            else False,
        },
        "homeostasis": {
            "enabled": bool(spec["homeostasis"]["enabled"]),
            "rule": spec["homeostasis"]["rule"],
            "alpha": float(spec["homeostasis"]["alpha"]),
            "eta": float(spec["homeostasis"]["eta"]),
            "r_target": float(spec["homeostasis"]["r_target"]),
            "clamp_min": float(spec["homeostasis"]["clamp_min"]),
            "clamp_max": float(spec["homeostasis"]["clamp_max"]),
            "scope": spec["homeostasis"]["scope"],
        },
        "pruning": {
            "enabled": bool(spec["pruning"]["enabled"]),
            "prune_interval_steps": int(spec["pruning"]["prune_interval_steps"]),
            "usage_alpha": float(spec["pruning"]["usage_alpha"]),
            "w_min": float(spec["pruning"]["w_min"]),
            "usage_min": float(spec["pruning"]["usage_min"]),
            "k_min_out": int(spec["pruning"]["k_min_out"]),
            "k_min_in": int(spec["pruning"]["k_min_in"]),
            "max_prune_fraction_per_interval": float(
                spec["pruning"]["max_prune_fraction_per_interval"]
            ),
            "verbose": bool(spec["pruning"]["verbose"]),
        },
        "neurogenesis": {
            "enabled": bool(spec["neurogenesis"]["enabled"]),
            "growth_interval_steps": int(spec["neurogenesis"]["growth_interval_steps"]),
            "add_neurons_per_event": int(spec["neurogenesis"]["add_neurons_per_event"]),
            "newborn_plasticity_multiplier": float(
                spec["neurogenesis"]["newborn_plasticity_multiplier"]
            ),
            "newborn_duration_steps": int(spec["neurogenesis"]["newborn_duration_steps"]),
            "max_total_neurons": int(spec["neurogenesis"]["max_total_neurons"]),
            "verbose": bool(spec["neurogenesis"]["verbose"]),
        },
        "logic_backend": logic_backend if is_logic_demo else None,
        "logic_gate": spec.get("logic_gate") if demo_id in _LOGIC_DEMO_TO_GATE else None,
        "logic_neuron_model": spec.get("logic_neuron_model")
        if is_logic_demo
        else None,
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
