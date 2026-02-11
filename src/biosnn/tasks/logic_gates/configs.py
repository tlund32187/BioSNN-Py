"""Configuration for logic-gate harness runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

from .datasets import LogicGate, SamplingMethod, coerce_gate

LearningMode = Literal["rstdp", "surrogate", "none"]
LogicGateNeuronModel = Literal["lif_3c", "adex_3c"]
AdvancedPostVoltageSource = Literal["auto", "soma", "dendrite"]
LearningRuleChoice = Literal["three_factor_elig_stdp", "rstdp_elig", "none"]
ModulatorFieldType = Literal["global_scalar", "grid_diffusion_2d"]
WrapperCombineMode = Literal["exp", "linear"]
LearningModulatorKind = Literal["dopamine", "acetylcholine", "noradrenaline", "serotonin"]
ExplorationMode = Literal["epsilon_greedy"]
ExplorationTieBreak = Literal["random_among_max", "alternate", "prefer_last"]


def _default_reversal_potential_v() -> dict[str, float]:
    return {
        "ampa": 0.0,
        "nmda": 0.0,
        "gaba_a": -0.070,
        "gaba_b": -0.090,
    }


@dataclass(slots=True)
class AdvancedSynapseConfig:
    enabled: bool = False
    conductance_mode: bool = False
    reversal_potential_v: dict[str, float] = field(default_factory=_default_reversal_potential_v)
    bio_synapse: bool = False
    bio_nmda_block: bool = False
    bio_stp: bool = False
    nmda_voltage_block: bool = False
    nmda_mg_mM: float = 1.0
    nmda_v_half_v: float = -0.020
    nmda_slope_v: float = 0.016
    stp_enabled: bool = False
    stp_u: float = 0.2
    stp_tau_rec_s: float = 0.2
    stp_tau_facil_s: float = 0.0
    stp_state_dtype: str | None = "float16"
    post_voltage_source: AdvancedPostVoltageSource = "auto"

    def __post_init__(self) -> None:
        self.post_voltage_source = cast(
            AdvancedPostVoltageSource,
            str(self.post_voltage_source).strip().lower()
            if str(self.post_voltage_source).strip().lower() in {"auto", "soma", "dendrite"}
            else "auto",
        )
        if bool(self.bio_synapse):
            self.enabled = True
            self.conductance_mode = True
        if bool(self.bio_nmda_block):
            self.enabled = True
            self.conductance_mode = True
            self.nmda_voltage_block = True
        if bool(self.bio_stp):
            self.enabled = True
            self.stp_enabled = True


@dataclass(slots=True)
class LogicModulatorConfig:
    enabled: bool = False
    field_type: ModulatorFieldType = "global_scalar"
    kinds: tuple[str, ...] = ("dopamine",)
    pulse_step: int = 50
    amount: float = 1.0
    grid_size: tuple[int, int] = (16, 16)
    world_extent: tuple[float, float] = (1.0, 1.0)
    diffusion: float = 0.0
    decay_tau: float = 1.0
    deposit_sigma: float = 0.0

    def __post_init__(self) -> None:
        self.field_type = cast_modulator_field_type(self.field_type)
        self.kinds = tuple(
            cast_learning_modulator_kind(kind) for kind in _coerce_kind_sequence(self.kinds)
        )
        if not self.kinds:
            self.kinds = ("dopamine",)
        self.pulse_step = max(1, int(self.pulse_step))
        self.grid_size = (max(1, int(self.grid_size[0])), max(1, int(self.grid_size[1])))
        self.world_extent = (float(self.world_extent[0]), float(self.world_extent[1]))
        self.diffusion = max(0.0, float(self.diffusion))
        self.decay_tau = max(1e-9, float(self.decay_tau))
        self.deposit_sigma = max(0.0, float(self.deposit_sigma))


@dataclass(slots=True)
class LearningWrapperConfig:
    enabled: bool = False
    ach_lr_gain: float = 0.0
    ne_lr_gain: float = 0.0
    ht_lr_gain: float = 0.0
    ht_extra_weight_decay: float = 0.0
    lr_clip_min: float = 0.1
    lr_clip_max: float = 10.0
    dopamine_baseline: float = 0.0
    ach_baseline: float = 0.0
    ne_baseline: float = 0.0
    ht_baseline: float = 0.0
    combine_mode: WrapperCombineMode = "exp"
    missing_modulators_policy: Literal["zero"] = "zero"

    def __post_init__(self) -> None:
        self.combine_mode = cast_wrapper_combine_mode(self.combine_mode)
        self.lr_clip_min = float(self.lr_clip_min)
        self.lr_clip_max = float(self.lr_clip_max)
        if self.lr_clip_max < self.lr_clip_min:
            self.lr_clip_max = self.lr_clip_min


@dataclass(slots=True)
class ExcitabilityModulationRunConfig:
    enabled: bool = False
    targets: tuple[str, ...] = ("hidden", "out")
    compartment: str = "soma"
    ach_gain: float = 0.0
    ne_gain: float = 0.0
    ht_gain: float = 0.0
    clamp_abs: float = 1.0

    def __post_init__(self) -> None:
        self.targets = tuple(str(target).strip().lower() for target in self.targets if str(target).strip())
        if not self.targets:
            self.targets = ("hidden", "out")
        comp = str(self.compartment).strip().lower()
        self.compartment = comp if comp in {"soma", "dendrite", "ais", "axon"} else "soma"
        self.clamp_abs = abs(float(self.clamp_abs))


@dataclass(slots=True)
class ExplorationConfig:
    enabled: bool = True
    mode: ExplorationMode = "epsilon_greedy"
    epsilon_start: float = 0.20
    epsilon_end: float = 0.01
    epsilon_decay_trials: int = 3000
    tie_break: ExplorationTieBreak = "random_among_max"
    seed: int = 123

    def __post_init__(self) -> None:
        self.mode = cast(
            ExplorationMode,
            "epsilon_greedy",
        )
        self.tie_break = cast_exploration_tie_break(self.tie_break)
        self.epsilon_start = float(self.epsilon_start)
        self.epsilon_end = float(self.epsilon_end)
        self.epsilon_decay_trials = max(1, int(self.epsilon_decay_trials))
        self.seed = int(self.seed)


@dataclass(slots=True)
class LogicGateRunConfig:
    gate: LogicGate | str = LogicGate.AND
    seed: int = 7
    steps: int = 5000
    dt: float = 1e-3
    sim_steps_per_trial: int = 10
    device: str = "cpu"
    learning_mode: LearningMode = "rstdp"
    engine_learning_rule: LearningRuleChoice = "three_factor_elig_stdp"
    learning_modulator_kind: LearningModulatorKind = "dopamine"
    neuron_model: LogicGateNeuronModel = "adex_3c"
    debug: bool = False
    debug_every: int = 25
    debug_top_k: int = 3
    export_every: int = 25
    dump_last_trials_csv: bool = False
    dump_last_trials_n: int = 50
    sampling_method: SamplingMethod = "sequential"
    out_dir: Path | None = None
    artifacts_root: Path | None = None
    advanced_synapse: AdvancedSynapseConfig = field(default_factory=AdvancedSynapseConfig)
    modulators: LogicModulatorConfig = field(default_factory=LogicModulatorConfig)
    wrapper: LearningWrapperConfig = field(default_factory=LearningWrapperConfig)
    excitability_modulation: ExcitabilityModulationRunConfig = field(default_factory=ExcitabilityModulationRunConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    reward_delivery_steps: int = 2
    reward_delivery_clamp_input: bool = True
    learning_lr_default: float = 1e-3
    dopamine_amount_default: float = 0.10
    dopamine_decay_tau_default: float = 0.05
    learn_every_default: int = 1

    def __post_init__(self) -> None:
        self.gate = coerce_gate(self.gate)
        self.device = str(self.device).lower().strip()
        self.learning_mode = cast_learning_mode(self.learning_mode)
        self.engine_learning_rule = cast_learning_rule_choice(self.engine_learning_rule)
        self.learning_modulator_kind = cast_learning_modulator_kind(self.learning_modulator_kind)
        self.neuron_model = cast_neuron_model(self.neuron_model)
        self.sampling_method = cast_sampling_method(self.sampling_method)
        if isinstance(self.advanced_synapse, Mapping):
            adv = self.advanced_synapse
            self.advanced_synapse = AdvancedSynapseConfig(
                enabled=bool(adv.get("enabled", False)),
                conductance_mode=bool(adv.get("conductance_mode", False)),
                reversal_potential_v=dict(cast(dict[str, float], adv.get("reversal_potential_v", _default_reversal_potential_v()))),
                bio_synapse=bool(adv.get("bio_synapse", False)),
                bio_nmda_block=bool(adv.get("bio_nmda_block", False)),
                bio_stp=bool(adv.get("bio_stp", False)),
                nmda_voltage_block=bool(adv.get("nmda_voltage_block", False)),
                nmda_mg_mM=float(adv.get("nmda_mg_mM", 1.0)),
                nmda_v_half_v=float(adv.get("nmda_v_half_v", -0.020)),
                nmda_slope_v=float(adv.get("nmda_slope_v", 0.016)),
                stp_enabled=bool(adv.get("stp_enabled", False)),
                stp_u=float(adv.get("stp_u", 0.2)),
                stp_tau_rec_s=float(adv.get("stp_tau_rec_s", 0.2)),
                stp_tau_facil_s=float(adv.get("stp_tau_facil_s", 0.0)),
                stp_state_dtype=cast(str | None, adv.get("stp_state_dtype", "float16")),
                post_voltage_source=cast(
                    AdvancedPostVoltageSource,
                    str(adv.get("post_voltage_source", "auto")).strip().lower(),
                ),
            )
        elif not isinstance(self.advanced_synapse, AdvancedSynapseConfig):
            self.advanced_synapse = AdvancedSynapseConfig()
        if isinstance(self.modulators, Mapping):
            mods = self.modulators
            self.modulators = LogicModulatorConfig(
                enabled=bool(mods.get("enabled", False)),
                field_type=cast(ModulatorFieldType, str(mods.get("field_type", "global_scalar")).strip().lower()),
                kinds=tuple(str(kind) for kind in _coerce_kind_sequence(mods.get("kinds", ("dopamine",)))),
                pulse_step=int(mods.get("pulse_step", 50)),
                amount=float(mods.get("amount", 1.0)),
                grid_size=_coerce_int_pair(mods.get("grid_size"), default=(16, 16)),
                world_extent=_coerce_float_pair(mods.get("world_extent"), default=(1.0, 1.0)),
                diffusion=float(mods.get("diffusion", 0.0)),
                decay_tau=float(mods.get("decay_tau", 1.0)),
                deposit_sigma=float(mods.get("deposit_sigma", 0.0)),
            )
        elif not isinstance(self.modulators, LogicModulatorConfig):
            self.modulators = LogicModulatorConfig()
        if isinstance(self.wrapper, Mapping):
            wrapper = self.wrapper
            self.wrapper = LearningWrapperConfig(
                enabled=bool(wrapper.get("enabled", False)),
                ach_lr_gain=float(wrapper.get("ach_lr_gain", 0.0)),
                ne_lr_gain=float(wrapper.get("ne_lr_gain", 0.0)),
                ht_lr_gain=float(wrapper.get("ht_lr_gain", 0.0)),
                ht_extra_weight_decay=float(wrapper.get("ht_extra_weight_decay", 0.0)),
                lr_clip_min=float(wrapper.get("lr_clip_min", 0.1)),
                lr_clip_max=float(wrapper.get("lr_clip_max", 10.0)),
                dopamine_baseline=float(wrapper.get("dopamine_baseline", 0.0)),
                ach_baseline=float(wrapper.get("ach_baseline", 0.0)),
                ne_baseline=float(wrapper.get("ne_baseline", 0.0)),
                ht_baseline=float(wrapper.get("ht_baseline", 0.0)),
                combine_mode=cast(WrapperCombineMode, str(wrapper.get("combine_mode", "exp")).strip().lower()),
                missing_modulators_policy=cast(Literal["zero"], str(wrapper.get("missing_modulators_policy", "zero")).strip().lower()),
            )
        elif not isinstance(self.wrapper, LearningWrapperConfig):
            self.wrapper = LearningWrapperConfig()
        if isinstance(self.excitability_modulation, Mapping):
            exc = self.excitability_modulation
            self.excitability_modulation = ExcitabilityModulationRunConfig(
                enabled=bool(exc.get("enabled", False)),
                targets=tuple(str(token) for token in _coerce_kind_sequence(exc.get("targets", ("hidden", "out")))),
                compartment=str(exc.get("compartment", "soma")),
                ach_gain=float(exc.get("ach_gain", 0.0)),
                ne_gain=float(exc.get("ne_gain", 0.0)),
                ht_gain=float(exc.get("ht_gain", 0.0)),
                clamp_abs=float(exc.get("clamp_abs", 1.0)),
            )
        elif not isinstance(self.excitability_modulation, ExcitabilityModulationRunConfig):
            self.excitability_modulation = ExcitabilityModulationRunConfig()
        if isinstance(self.exploration, Mapping):
            exploration = self.exploration
            self.exploration = ExplorationConfig(
                enabled=bool(exploration.get("enabled", True)),
                mode=cast(ExplorationMode, "epsilon_greedy"),
                epsilon_start=float(exploration.get("epsilon_start", 0.20)),
                epsilon_end=float(exploration.get("epsilon_end", 0.01)),
                epsilon_decay_trials=int(exploration.get("epsilon_decay_trials", 3000)),
                tie_break=cast(
                    ExplorationTieBreak,
                    str(exploration.get("tie_break", "random_among_max")).strip().lower(),
                ),
                seed=int(exploration.get("seed", 123)),
            )
        elif not isinstance(self.exploration, ExplorationConfig):
            self.exploration = ExplorationConfig()
        self.reward_delivery_steps = max(0, int(self.reward_delivery_steps))
        self.reward_delivery_clamp_input = bool(self.reward_delivery_clamp_input)
        self.learning_lr_default = max(1e-9, float(self.learning_lr_default))
        self.dopamine_amount_default = max(0.0, float(self.dopamine_amount_default))
        self.dopamine_decay_tau_default = max(1e-9, float(self.dopamine_decay_tau_default))
        self.learn_every_default = max(1, int(self.learn_every_default))
        if self.steps <= 0:
            raise ValueError("steps must be > 0")
        if self.dt <= 0.0:
            raise ValueError("dt must be > 0")
        if self.sim_steps_per_trial <= 0:
            raise ValueError("sim_steps_per_trial must be > 0")
        if self.debug_every <= 0:
            raise ValueError("debug_every must be > 0")
        if self.debug_top_k <= 0:
            raise ValueError("debug_top_k must be > 0")
        if self.export_every <= 0:
            raise ValueError("export_every must be > 0")
        if self.dump_last_trials_n <= 0:
            raise ValueError("dump_last_trials_n must be > 0")


def cast_learning_mode(value: str) -> LearningMode:
    mode = str(value).lower().strip()
    if mode not in {"rstdp", "surrogate", "none"}:
        raise ValueError("learning_mode must be one of: rstdp, surrogate, none")
    return cast(LearningMode, mode)


def cast_sampling_method(value: str) -> SamplingMethod:
    mode = str(value).lower().strip()
    if mode not in {"sequential", "random_balanced"}:
        raise ValueError("sampling_method must be one of: sequential, random_balanced")
    return cast(SamplingMethod, mode)


def cast_learning_rule_choice(value: str) -> LearningRuleChoice:
    rule = str(value).lower().strip()
    if rule not in {"three_factor_elig_stdp", "rstdp_elig", "none"}:
        raise ValueError("engine_learning_rule must be one of: three_factor_elig_stdp, rstdp_elig, none")
    return cast(LearningRuleChoice, rule)


def cast_neuron_model(value: str) -> LogicGateNeuronModel:
    model = str(value).lower().strip()
    if model not in {"lif_3c", "adex_3c"}:
        raise ValueError("neuron_model must be one of: lif_3c, adex_3c")
    return cast(LogicGateNeuronModel, model)


def cast_modulator_field_type(value: str) -> ModulatorFieldType:
    field_type = str(value).lower().strip()
    if field_type not in {"global_scalar", "grid_diffusion_2d"}:
        raise ValueError("field_type must be one of: global_scalar, grid_diffusion_2d")
    return cast(ModulatorFieldType, field_type)


def cast_wrapper_combine_mode(value: str) -> WrapperCombineMode:
    mode = str(value).lower().strip()
    if mode not in {"exp", "linear"}:
        raise ValueError("combine_mode must be one of: exp, linear")
    return cast(WrapperCombineMode, mode)


def cast_learning_modulator_kind(value: str) -> LearningModulatorKind:
    token = str(value).strip().lower()
    aliases = {
        "da": "dopamine",
        "ach": "acetylcholine",
        "na": "noradrenaline",
        "5ht": "serotonin",
    }
    token = aliases.get(token, token)
    if token not in {"dopamine", "acetylcholine", "noradrenaline", "serotonin"}:
        raise ValueError(
            "learning_modulator_kind must be one of: dopamine, acetylcholine, noradrenaline, serotonin"
        )
    return cast(LearningModulatorKind, token)


def cast_exploration_tie_break(value: str) -> ExplorationTieBreak:
    token = str(value).strip().lower()
    if token not in {"random_among_max", "alternate", "prefer_last"}:
        token = "random_among_max"
    return cast(ExplorationTieBreak, token)


def _coerce_kind_sequence(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",")]
        return tuple(item for item in items if item)
    if isinstance(value, Mapping):
        return tuple()
    if isinstance(value, tuple):
        return tuple(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return tuple()


def _coerce_int_pair(value: object, *, default: tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, str):
        token = value.strip().lower().replace(" ", "")
        if "x" in token:
            left, right = token.split("x", 1)
            try:
                return max(1, int(left)), max(1, int(right))
            except Exception:
                return default
    if isinstance(value, tuple) and len(value) == 2:
        try:
            return max(1, int(value[0])), max(1, int(value[1]))
        except Exception:
            return default
    if isinstance(value, list) and len(value) == 2:
        try:
            return max(1, int(value[0])), max(1, int(value[1]))
        except Exception:
            return default
    return default


def _coerce_float_pair(value: object, *, default: tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, tuple) and len(value) == 2:
        try:
            return float(value[0]), float(value[1])
        except Exception:
            return default
    if isinstance(value, list) and len(value) == 2:
        try:
            return float(value[0]), float(value[1])
        except Exception:
            return default
    if isinstance(value, str):
        token = value.strip().replace(" ", "")
        if "," in token:
            left, right = token.split(",", 1)
            try:
                return float(left), float(right)
            except Exception:
                return default
    return default


__all__ = [
    "AdvancedPostVoltageSource",
    "AdvancedSynapseConfig",
    "ExcitabilityModulationRunConfig",
    "ExplorationConfig",
    "ExplorationMode",
    "ExplorationTieBreak",
    "LearningModulatorKind",
    "LearningRuleChoice",
    "LearningWrapperConfig",
    "LearningMode",
    "LogicModulatorConfig",
    "LogicGateNeuronModel",
    "ModulatorFieldType",
    "LogicGateRunConfig",
    "WrapperCombineMode",
    "cast_learning_rule_choice",
    "cast_learning_modulator_kind",
    "cast_learning_mode",
    "cast_exploration_tie_break",
    "cast_modulator_field_type",
    "cast_neuron_model",
    "cast_sampling_method",
    "cast_wrapper_combine_mode",
]
