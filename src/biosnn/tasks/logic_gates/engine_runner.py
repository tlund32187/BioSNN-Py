"""Engine-backed logic-gate runners.

This module mirrors the CSV artifact contract of the deterministic harness while
running through ``TorchNetworkEngine`` with configurable synapse/learning/modulator
settings from a run spec mapping.
"""

from __future__ import annotations

import fnmatch
import random
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from biosnn.contracts.learning import ILearningRule
from biosnn.contracts.modulators import IModulatorField, ModulatorKind, ModulatorRelease
from biosnn.contracts.neurons import Compartment, StepContext
from biosnn.contracts.simulation import ExcitabilityModulationConfig, SimulationConfig
from biosnn.core.torch_utils import require_torch
from biosnn.io.dashboard_export import export_population_topology_json
from biosnn.io.sinks import CsvSink
from biosnn.learning import (
    EligibilityTraceHebbianParams,
    EligibilityTraceHebbianRule,
    HomeostasisScope,
    MetaplasticProjectionParams,
    MetaplasticProjectionRule,
    ModulatedRuleWrapper,
    ModulatedRuleWrapperParams,
    RateEmaThresholdHomeostasis,
    RateEmaThresholdHomeostasisConfig,
    RStdpEligibilityParams,
    RStdpEligibilityRule,
    ThreeFactorEligibilityStdpParams,
    ThreeFactorEligibilityStdpRule,
    ThreeFactorHebbianParams,
    ThreeFactorHebbianRule,
)
from biosnn.neuromodulators import (
    GlobalScalarField,
    GlobalScalarParams,
    GridDiffusion2DField,
    GridDiffusion2DParams,
)
from biosnn.simulation.engine import TorchNetworkEngine
from biosnn.simulation.network import ModulatorSpec, PopulationSpec, ProjectionSpec
from biosnn.synapses.dynamics.delayed_sparse_matmul import DelayedSparseMatmulSynapse

from .configs import CurriculumGateContextConfig, ExplorationConfig, LogicGateRunConfig
from .datasets import LogicGate, coerce_gate, make_truth_table, sample_case_indices
from .encoding import (
    INPUT_NEURON_INDICES,
    OUTPUT_NEURON_INDICES,
    encode_inputs,
    gate_context_level_for_gate,
)
from .evaluators import PassTracker, eval_accuracy
from .topologies import build_logic_gate_ff, build_logic_gate_xor

DEFAULT_CURRICULUM_GATES: tuple[LogicGate, ...] = (
    LogicGate.OR,
    LogicGate.AND,
    LogicGate.NOR,
    LogicGate.NAND,
    LogicGate.XOR,
    LogicGate.XNOR,
)


@dataclass(slots=True)
class _EngineBuild:
    engine: TorchNetworkEngine
    input_population: str
    output_population: str
    hidden_populations: tuple[str, ...]
    current_input_drive: dict[str, Any]
    context_drive_by_population: dict[str, Mapping[Compartment, Any]]
    action_drive_by_population: dict[str, Mapping[Compartment, Any]]
    pending_releases: dict[ModulatorKind, float]
    modulator_kinds: tuple[ModulatorKind, ...]
    modulator_amount: float
    modulator_state: dict[str, Any]
    learning_proj_names: tuple[str, ...]
    learning_proj_name: str | None


@dataclass(slots=True)
class _WeightExportProjection:
    name: str
    pre_idx: Any
    post_idx: Any


@dataclass(slots=True)
class _MonitorWriters:
    spikes_sink: CsvSink | None
    weights_sink: CsvSink | None
    spike_populations: tuple[str, ...]
    weight_projections: tuple[_WeightExportProjection, ...]


def select_action_wta(
    spike_counts: Any,
    *,
    last_action: int | None,
    exploration: ExplorationConfig,
    trial_idx: int,
    is_eval: bool,
    rng: random.Random,
) -> tuple[int, dict[str, Any]]:
    torch = require_torch()
    counts = spike_counts.reshape(-1).to(dtype=torch.float32)
    if int(counts.numel()) <= 0:
        raise ValueError("WTA action selection requires at least one output value.")

    max_value = float(counts.max().item())
    max_indices_tensor = torch.nonzero(counts == max_value, as_tuple=False).reshape(-1)
    max_indices = [int(value.item()) for value in max_indices_tensor]
    if not max_indices:
        max_indices = [0]
    tie = len(max_indices) > 1
    greedy = _select_tie_break_action(
        max_indices=max_indices,
        last_action=last_action,
        tie_break=exploration.tie_break,
        rng=rng,
    )

    epsilon = 0.0
    if exploration.enabled and not is_eval:
        decay_trials = max(1, int(exploration.epsilon_decay_trials))
        frac = min(1.0, max(0.0, float(trial_idx)) / float(decay_trials))
        epsilon = float(exploration.epsilon_start) + frac * (
            float(exploration.epsilon_end) - float(exploration.epsilon_start)
        )
        lo = min(float(exploration.epsilon_start), float(exploration.epsilon_end))
        hi = max(float(exploration.epsilon_start), float(exploration.epsilon_end))
        epsilon = _clamp(epsilon, lo, hi)

    explored = False
    action = int(greedy)
    if epsilon > 0.0 and int(counts.numel()) > 1 and rng.random() < epsilon:
        action = int(rng.randrange(int(counts.numel())))
        explored = True

    return action, {
        "epsilon": float(epsilon),
        "explored": bool(explored),
        "tie": bool(tie),
        "greedy": int(greedy),
    }


def _should_commit_action_force(
    *,
    action_info: Mapping[str, Any] | None,
    chosen_action: int,
    out_spike_counts: Any,
    mode: str,
) -> bool:
    mode_norm = str(mode).strip().lower()
    explored = bool(action_info.get("explored", False)) if action_info else False
    if mode_norm == "always":
        return True
    try:
        chosen_idx = int(chosen_action)
        if chosen_idx < 0:
            return False
        selected_spikes = float(out_spike_counts[chosen_idx].item())
        silent = selected_spikes <= 0.0
        if mode_norm == "explore_only":
            return explored
        if mode_norm == "silent_only":
            return silent
        # Default: explore_or_silent
        return bool(explored or silent)
    except Exception:
        return explored if mode_norm == "explore_only" else False


def _select_tie_break_action(
    *,
    max_indices: Sequence[int],
    last_action: int | None,
    tie_break: str,
    rng: random.Random,
) -> int:
    if not max_indices:
        return 0
    mode = str(tie_break).strip().lower()
    if mode == "random_among_max":
        return int(rng.choice(list(max_indices)))
    if mode == "alternate":
        if last_action is None:
            return int(max_indices[0])
        if int(last_action) in max_indices:
            cursor = list(max_indices).index(int(last_action))
            return int(max_indices[(cursor + 1) % len(max_indices)])
        return int(max_indices[0])
    if mode == "prefer_last":
        if last_action is not None and int(last_action) in max_indices:
            return int(last_action)
        return int(max_indices[0])
    return int(max_indices[0])


def _resolve_exploration_cfg(
    run_spec: Mapping[str, Any],
    *,
    default_cfg: ExplorationConfig,
) -> ExplorationConfig:
    logic = _as_mapping(run_spec.get("logic"))
    raw = _as_mapping(_first_non_none(logic.get("exploration"), run_spec.get("exploration")))
    return ExplorationConfig(
        enabled=bool(_first_non_none(raw.get("enabled"), default_cfg.enabled)),
        mode="epsilon_greedy",
        epsilon_start=_coerce_float(
            _first_non_none(raw.get("epsilon_start"), default_cfg.epsilon_start),
            default_cfg.epsilon_start,
        ),
        epsilon_end=_coerce_float(
            _first_non_none(raw.get("epsilon_end"), default_cfg.epsilon_end),
            default_cfg.epsilon_end,
        ),
        epsilon_decay_trials=_coerce_positive_int(
            _first_non_none(raw.get("epsilon_decay_trials"), default_cfg.epsilon_decay_trials),
            default_cfg.epsilon_decay_trials,
        ),
        tie_break=cast(
            Any,
            _coerce_choice(
                _first_non_none(raw.get("tie_break"), default_cfg.tie_break),
                allowed={"random_among_max", "alternate", "prefer_last"},
                default=default_cfg.tie_break,
            ),
        ),
        seed=_coerce_nonnegative_int(
            _first_non_none(raw.get("seed"), default_cfg.seed), default_cfg.seed
        ),
    )


def _resolve_reward_window_cfg(
    run_spec: Mapping[str, Any],
    *,
    default_steps: int,
    default_clamp_input: bool,
) -> tuple[int, bool]:
    logic = _as_mapping(run_spec.get("logic"))
    reward_steps = _coerce_nonnegative_int(
        _first_non_none(
            logic.get("reward_delivery_steps"),
            run_spec.get("reward_delivery_steps"),
            default_steps,
        ),
        default_steps,
    )
    clamp_raw = _coerce_optional_bool(
        _first_non_none(
            logic.get("reward_delivery_clamp_input"),
            run_spec.get("reward_delivery_clamp_input"),
            default_clamp_input,
        )
    )
    clamp_input = bool(default_clamp_input if clamp_raw is None else clamp_raw)
    return int(reward_steps), bool(clamp_input)


def _resolve_action_force_cfg(
    run_spec: Mapping[str, Any],
    *,
    default_enabled: bool,
    default_window: str = "reward_window",
    default_steps: int = 1,
    default_amplitude: float = 0.35,
    default_compartment: str = "soma",
    default_mode: str = "explore_or_silent",
) -> dict[str, Any]:
    logic = _as_mapping(run_spec.get("logic"))
    raw = _as_mapping(_first_non_none(logic.get("action_force"), run_spec.get("action_force")))
    return {
        "enabled": bool(_first_non_none(raw.get("enabled"), default_enabled)),
        "window": _coerce_choice(
            _first_non_none(raw.get("window"), default_window),
            allowed={"reward_window", "post_decision"},
            default=default_window,
        ),
        "steps": _coerce_nonnegative_int(
            _first_non_none(raw.get("steps"), default_steps),
            default_steps,
        ),
        "amplitude": max(
            0.0,
            _coerce_float(
                _first_non_none(raw.get("amplitude"), default_amplitude), default_amplitude
            ),
        ),
        "compartment": _coerce_choice(
            _first_non_none(raw.get("compartment"), default_compartment),
            allowed={"soma", "dendrite", "ais", "axon"},
            default=default_compartment,
        ),
        "mode": _coerce_choice(
            _first_non_none(raw.get("mode"), default_mode),
            allowed={"explore_only", "silent_only", "explore_or_silent", "always"},
            default=default_mode,
        ),
    }


def _reward_window_input_drive(
    *,
    input_drive: Any,
    clamp_input: bool,
) -> Any:
    # Clamp=True keeps the same stimulus input during reward steps.
    if bool(clamp_input):
        return input_drive
    return input_drive.new_zeros(input_drive.shape)


def _effective_curriculum_engine_run_spec(
    *,
    run_spec: Mapping[str, Any],
    config: LogicGateRunConfig,
) -> dict[str, Any]:
    effective = dict(run_spec)
    learning = dict(_as_mapping(effective.get("learning")))
    if learning.get("lr") is None:
        learning["lr"] = float(config.learning_lr_default)
    effective["learning"] = learning

    modulators = dict(_as_mapping(effective.get("modulators")))
    if modulators.get("amount") is None:
        modulators["amount"] = float(config.dopamine_amount_default)
    if modulators.get("decay_tau") is None:
        modulators["decay_tau"] = float(config.dopamine_decay_tau_default)
    effective["modulators"] = modulators

    logic = dict(_as_mapping(effective.get("logic")))
    if logic.get("learn_every") is None:
        logic["learn_every"] = int(config.learn_every_default)
    if logic.get("reward_delivery_steps") is None:
        logic["reward_delivery_steps"] = int(config.reward_delivery_steps)
    if logic.get("reward_delivery_clamp_input") is None:
        logic["reward_delivery_clamp_input"] = bool(config.reward_delivery_clamp_input)
    if logic.get("exploration") is None:
        logic["exploration"] = asdict(config.exploration)
    if logic.get("gate_context") is None:
        logic["gate_context"] = asdict(config.curriculum_gate_context)
    if logic.get("action_force") is None:
        logic["action_force"] = {
            "enabled": False,
            "window": "reward_window",
            "steps": 1,
            "amplitude": 0.35,
            "compartment": "soma",
            "mode": "explore_or_silent",
        }
    effective["logic"] = logic
    return effective


def _resolve_curriculum_gate_context_cfg(
    run_spec: Mapping[str, Any],
    *,
    default_cfg: CurriculumGateContextConfig,
) -> dict[str, Any]:
    logic = _as_mapping(run_spec.get("logic"))
    raw = _as_mapping(_first_non_none(logic.get("gate_context"), run_spec.get("gate_context")))
    targets = _coerce_string_list(_first_non_none(raw.get("targets"), default_cfg.targets))
    if not targets:
        targets = [str(token) for token in default_cfg.targets]
    compartment = _coerce_choice(
        _first_non_none(raw.get("compartment"), default_cfg.compartment),
        allowed={"soma", "dendrite", "ais", "axon"},
        default=str(default_cfg.compartment),
    )
    return {
        "enabled": bool(_first_non_none(raw.get("enabled"), default_cfg.enabled)),
        "amplitude": max(
            0.0,
            _coerce_float(
                _first_non_none(raw.get("amplitude"), default_cfg.amplitude), default_cfg.amplitude
            ),
        ),
        "compartment": compartment,
        "targets": targets,
    }


def run_logic_gate_engine(
    config: LogicGateRunConfig,
    run_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Run a single logic-gate task through ``TorchNetworkEngine``."""
    torch = require_torch()
    gate = coerce_gate(config.gate)
    device = _resolve_device(torch, str(config.device))
    run_dir = _resolve_run_dir(config, gate=gate)

    torch.manual_seed(int(config.seed))
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.seed))
    sampling_generator = torch.Generator(device="cpu")
    sampling_generator.manual_seed(int(config.seed))

    engine_build = _build_engine(
        config=config,
        gate=gate,
        run_spec=run_spec,
        device=device,
    )
    engine = engine_build.engine
    _write_engine_topology_json(engine=engine, run_dir=run_dir)

    inputs, targets = make_truth_table(gate, device=device, dtype="float32")
    targets_flat = targets.reshape(4)
    encoded_inputs = _build_encoded_input_table(inputs=inputs, dt=config.dt)

    case_indices = sample_case_indices(
        config.steps,
        method=config.sampling_method,
        generator=sampling_generator,
        device="cpu",
    )
    case_sequence = [int(value) for value in case_indices.tolist()]
    exploration_cfg = _resolve_exploration_cfg(
        run_spec,
        default_cfg=ExplorationConfig(
            enabled=False,
            mode=config.exploration.mode,
            epsilon_start=config.exploration.epsilon_start,
            epsilon_end=config.exploration.epsilon_end,
            epsilon_decay_trials=config.exploration.epsilon_decay_trials,
            tie_break=config.exploration.tie_break,
            seed=config.exploration.seed,
        ),
    )
    decision_rng = random.Random(int(exploration_cfg.seed))
    reward_delivery_steps, reward_delivery_clamp_input = _resolve_reward_window_cfg(
        run_spec,
        default_steps=0,
        default_clamp_input=config.reward_delivery_clamp_input,
    )
    action_force_cfg = _resolve_action_force_cfg(
        run_spec,
        default_enabled=False,
        default_window="reward_window",
        default_steps=1,
        default_amplitude=0.75,
        default_compartment="soma",
    )

    monitors_enabled = _resolve_monitors_enabled(run_spec)
    trial_flush_every = 1 if monitors_enabled else max(1, config.export_every)
    trial_sink = CsvSink(run_dir / "trials.csv", flush_every=trial_flush_every)
    eval_sink = CsvSink(run_dir / "eval.csv", flush_every=1)
    confusion_sink = CsvSink(run_dir / "confusion.csv", flush_every=1)
    last_trials: deque[dict[str, Any]] | None = (
        deque(maxlen=config.dump_last_trials_n) if config.dump_last_trials_csv else None
    )
    monitor_writers = _create_monitor_writers(
        engine=engine,
        engine_build=engine_build,
        run_spec=run_spec,
        run_dir=run_dir,
        flush_every=max(1, config.export_every),
    )

    tracker = PassTracker(gate)
    first_pass_trial: int | None = None
    sampled_correct = 0
    last_pred: int | None = None
    rolling_correct: deque[int] = deque(maxlen=100)
    last_trial_acc_rolling = 0.0
    last_mean_abs_dw = 0.0
    t0 = perf_counter()
    sim_step = 0

    predictions = torch.zeros((4,), device=inputs.device, dtype=inputs.dtype)
    case_scores = predictions.clone()

    init_elig, init_dw, init_w_min, init_w_max, init_w_mean = _learning_stats(
        engine=engine,
        learning_proj_names=engine_build.learning_proj_names,
    )
    init_eval_acc, init_confusion = eval_accuracy(
        predictions,
        targets_flat,
        report_confusion=True,
    )
    eval_sink.write_row(
        {
            "trial": 0,
            "eval_accuracy": float(init_eval_acc),
            "sample_accuracy": 0.0,
            "trial_acc_rolling": 0.0,
            "pred_00": int(predictions[0].item()),
            "pred_01": int(predictions[1].item()),
            "pred_10": int(predictions[2].item()),
            "pred_11": int(predictions[3].item()),
            "mean_eligibility_abs": init_elig,
            "mean_abs_dw": init_dw,
            "weights_min": init_w_min,
            "weights_max": init_w_max,
            "weights_mean": init_w_mean,
            "perfect_streak": 0,
            "high_streak": 0,
            "passed": 0,
        }
    )
    confusion_sink.write_row({"trial": 0, **init_confusion})

    last_trials_csv: Path | None = None
    try:
        for trial_idx, case_idx in enumerate(case_sequence, start=1):
            target_value = targets_flat[case_idx]
            input_drive = encoded_inputs[case_idx].clone()
            _apply_gate_context_drive(input_drive, gate=gate)
            out_spike_counts, hidden_spike_counts, sim_step = _run_trial_steps(
                engine_build=engine_build,
                input_drive=input_drive,
                sim_steps_per_trial=config.sim_steps_per_trial,
                step_offset=sim_step,
                monitor_writers=monitor_writers,
            )

            pred_bit, action_info = select_action_wta(
                out_spike_counts,
                last_action=last_pred,
                exploration=exploration_cfg,
                trial_idx=trial_idx - 1,
                is_eval=False,
                rng=decision_rng,
            )
            chosen_action = int(pred_bit)
            engine_build.modulator_state["dopamine_focus_action"] = chosen_action
            target_bit = int((target_value >= 0.5).item())
            correct = int(chosen_action == target_bit)
            sampled_correct += correct
            rolling_correct.append(correct)
            trial_acc_rolling = sum(rolling_correct) / float(len(rolling_correct))
            predictions[case_idx] = float(chosen_action)
            case_scores.copy_(predictions)

            action_forced = 0
            action_drive_amplitude = 0.0
            action_window = str(action_force_cfg["window"])
            action_steps = int(action_force_cfg["steps"])

            dopamine_pulse = _queue_trial_feedback_releases(
                pending_releases=engine_build.pending_releases,
                modulator_kinds=engine_build.modulator_kinds,
                modulator_amount=engine_build.modulator_amount,
                correct=bool(correct),
            )
            if reward_delivery_steps > 0:
                should_force_action = (
                    bool(action_force_cfg["enabled"])
                    and _should_commit_action_force(
                        action_info=action_info,
                        chosen_action=chosen_action,
                        out_spike_counts=out_spike_counts,
                        mode=str(action_force_cfg.get("mode", "explore_or_silent")),
                    )
                    and action_forced == 0
                    and action_window in {"reward_window", "post_decision"}
                    and int(action_steps) > 0
                )
                reward_input = _reward_window_input_drive(
                    input_drive=input_drive,
                    clamp_input=reward_delivery_clamp_input and not should_force_action,
                )
                if should_force_action:
                    forced = _set_action_force_drive(
                        engine_build=engine_build,
                        chosen_action=chosen_action,
                        amplitude=float(action_force_cfg["amplitude"]),
                        compartment=str(action_force_cfg["compartment"]),
                    )
                    if forced:
                        action_forced = 1
                        action_drive_amplitude = float(action_force_cfg["amplitude"])
                force_steps = (
                    min(max(0, int(action_steps)), max(0, int(reward_delivery_steps)))
                    if action_forced
                    else 0
                )
                reward_out_counts, _, sim_step = _run_trial_steps(
                    engine_build=engine_build,
                    input_drive=reward_input,
                    sim_steps_per_trial=reward_delivery_steps,
                    action_force_steps=force_steps,
                    step_offset=sim_step,
                    monitor_writers=monitor_writers,
                )
                if action_forced and action_window == "reward_window":
                    out_spike_counts.add_(reward_out_counts)
            else:
                # Prevent delayed modulators from spilling into the next trial stimulus window.
                _discard_pending_releases(pending_releases=engine_build.pending_releases)
            (
                mean_eligibility_abs,
                mean_abs_dw,
                weights_min,
                weights_max,
                weights_mean,
            ) = _learning_stats(
                engine=engine,
                learning_proj_names=engine_build.learning_proj_names,
            )
            hidden_mean_spikes = float(hidden_spike_counts.mean().item()) / float(
                config.sim_steps_per_trial
            )
            tie_behavior = int(bool(action_info.get("tie", False)))
            no_spikes = int(float(out_spike_counts.sum().item()) <= 0.0)

            trial_row = {
                "trial": trial_idx,
                "sim_step_end": sim_step,
                "case_idx": case_idx,
                "x0": float(inputs[case_idx, 0].item()),
                "x1": float(inputs[case_idx, 1].item()),
                "in_bit0_0_drive": float(input_drive[INPUT_NEURON_INDICES["bit0_0"]].item()),
                "in_bit0_1_drive": float(input_drive[INPUT_NEURON_INDICES["bit0_1"]].item()),
                "in_bit1_0_drive": float(input_drive[INPUT_NEURON_INDICES["bit1_0"]].item()),
                "in_bit1_1_drive": float(input_drive[INPUT_NEURON_INDICES["bit1_1"]].item()),
                "in_gate_context_drive": _gate_context_drive_value(input_drive),
                "out_spikes_0": float(out_spike_counts[OUTPUT_NEURON_INDICES["class_0"]].item()),
                "out_spikes_1": float(out_spike_counts[OUTPUT_NEURON_INDICES["class_1"]].item()),
                "hidden_mean_spikes": hidden_mean_spikes,
                "dopamine_pulse": dopamine_pulse,
                "trial_acc_rolling": trial_acc_rolling,
                "mean_eligibility_abs": mean_eligibility_abs,
                "mean_abs_dw": mean_abs_dw,
                "weights_min": weights_min,
                "weights_max": weights_max,
                "weights_mean": weights_mean,
                "tie_wta": tie_behavior,
                "epsilon": float(action_info.get("epsilon", 0.0)),
                "explored": int(bool(action_info.get("explored", False))),
                "greedy_action": int(action_info.get("greedy", chosen_action)),
                "chosen_action": chosen_action,
                "action_forced": int(action_forced),
                "action_drive_amplitude": float(action_drive_amplitude),
                "no_output_spikes": no_spikes,
                "target": target_bit,
                "pred": chosen_action,
                "correct": correct,
            }
            trial_sink.write_row(trial_row)
            if last_trials is not None:
                last_trials.append(dict(trial_row))

            eval_acc = float(eval_accuracy(predictions, targets_flat))
            tracker.update(eval_acc)
            if tracker.passed and first_pass_trial is None:
                first_pass_trial = trial_idx

            if (trial_idx % config.export_every) == 0 or trial_idx == config.steps:
                sample_acc = sampled_correct / float(trial_idx)
                eval_acc_full, confusion = eval_accuracy(
                    predictions, targets_flat, report_confusion=True
                )
                eval_sink.write_row(
                    {
                        "trial": trial_idx,
                        "eval_accuracy": eval_acc_full,
                        "sample_accuracy": sample_acc,
                        "trial_acc_rolling": trial_acc_rolling,
                        "pred_00": int(predictions[0].item()),
                        "pred_01": int(predictions[1].item()),
                        "pred_10": int(predictions[2].item()),
                        "pred_11": int(predictions[3].item()),
                        "mean_eligibility_abs": mean_eligibility_abs,
                        "mean_abs_dw": mean_abs_dw,
                        "weights_min": weights_min,
                        "weights_max": weights_max,
                        "weights_mean": weights_mean,
                        "perfect_streak": tracker.perfect_streak,
                        "high_streak": tracker.high_streak,
                        "passed": int(tracker.passed),
                    }
                )
                confusion_sink.write_row({"trial": trial_idx, **confusion})
                _write_weights_snapshot(
                    engine=engine,
                    monitor_writers=monitor_writers,
                    step=sim_step,
                )

            last_pred = chosen_action
            last_trial_acc_rolling = trial_acc_rolling
            last_mean_abs_dw = mean_abs_dw
    finally:
        trial_sink.close()
        eval_sink.close()
        confusion_sink.close()
        _close_monitor_writers(monitor_writers)

    if last_trials is not None and len(last_trials) > 0:
        last_trials_csv = run_dir / f"trials_last_{config.dump_last_trials_n}.csv"
        last_sink = CsvSink(last_trials_csv, flush_every=1)
        try:
            for row in last_trials:
                last_sink.write_row(row)
        finally:
            last_sink.close()

    elapsed_s = perf_counter() - t0
    final_eval_acc = float(eval_accuracy(predictions, targets_flat))
    sample_accuracy = sampled_correct / float(config.steps)

    return {
        "out_dir": run_dir,
        "gate": gate.value,
        "device": device,
        "learning_mode": config.learning_mode,
        "topology": "engine_ff",
        "steps": config.steps,
        "sim_steps_per_trial": config.sim_steps_per_trial,
        "inputs": inputs,
        "targets": targets,
        "preds": predictions,
        "case_scores": case_scores,
        "sample_indices": case_indices,
        "eval_accuracy": final_eval_acc,
        "sample_accuracy": sample_accuracy,
        "trial_acc_rolling": last_trial_acc_rolling,
        "mean_abs_dw": last_mean_abs_dw,
        "passed": tracker.passed,
        "first_pass_trial": first_pass_trial,
        "elapsed_s": elapsed_s,
        "input_neuron_indices": dict(INPUT_NEURON_INDICES),
        "output_neuron_indices": dict(OUTPUT_NEURON_INDICES),
        "debug_every": config.debug_every,
        "last_trials_csv": last_trials_csv,
        "trials_csv": run_dir / "trials.csv",
        "eval_csv": run_dir / "eval.csv",
        "confusion_csv": run_dir / "confusion.csv",
        "topology_json": run_dir / "topology.json",
    }


def _run_trial_steps(
    *,
    engine_build: _EngineBuild,
    input_drive: Any,
    sim_steps_per_trial: int,
    action_force_steps: int = 0,
    step_offset: int = 0,
    monitor_writers: _MonitorWriters | None = None,
) -> tuple[Any, Any, int]:
    torch = require_torch()
    engine_build.current_input_drive["tensor"] = input_drive
    mod_state = engine_build.modulator_state
    input_active = bool(float(input_drive.abs().sum().item()) > 0.0)
    if mod_state:
        mod_state["input_active"] = input_active
        mod_state["ach_pulse_emitted"] = False
        mod_state["trial_steps"] = max(1, int(sim_steps_per_trial))
    out_spikes = engine_build.engine._pop_states[engine_build.output_population].spikes
    hidden_spikes_by_pop = [
        engine_build.engine._pop_states[name].spikes for name in engine_build.hidden_populations
    ]
    if not hidden_spikes_by_pop:
        raise RuntimeError("Engine topology does not define any hidden population.")
    output_counts = torch.zeros_like(out_spikes, dtype=input_drive.dtype)
    hidden_counts_by_pop = [
        torch.zeros_like(hidden_spikes, dtype=input_drive.dtype)
        for hidden_spikes in hidden_spikes_by_pop
    ]
    global_step = int(step_offset)
    force_steps = max(0, int(action_force_steps))

    try:
        for trial_step_idx in range(max(1, int(sim_steps_per_trial))):
            if mod_state:
                mod_state["trial_step_idx"] = int(trial_step_idx)
            if force_steps > 0 and int(trial_step_idx) >= force_steps:
                _clear_action_force_drive(engine_build=engine_build)
            engine_build.engine.step()
            global_step += 1
            _write_spike_events_for_step(
                engine=engine_build.engine,
                monitor_writers=monitor_writers,
                step=global_step,
            )
            output_counts.add_(
                engine_build.engine._pop_states[engine_build.output_population].spikes.to(
                    dtype=input_drive.dtype
                )
            )
            for pop_idx, pop_name in enumerate(engine_build.hidden_populations):
                hidden_counts_by_pop[pop_idx].add_(
                    engine_build.engine._pop_states[pop_name].spikes.to(dtype=input_drive.dtype)
                )
    finally:
        if force_steps > 0:
            _clear_action_force_drive(engine_build=engine_build)
        if mod_state:
            mod_state["input_active"] = False
            mod_state["trial_step_idx"] = -1
    if len(hidden_counts_by_pop) == 1:
        hidden_counts = hidden_counts_by_pop[0]
    else:
        hidden_counts = torch.cat(hidden_counts_by_pop, dim=0)
    return output_counts, hidden_counts, global_step


def _build_curriculum_gate_context_cache(
    *,
    engine_build: _EngineBuild,
    gate_count: int,
    cfg: Mapping[str, Any],
) -> dict[int, dict[str, Mapping[Compartment, Any]]]:
    torch = require_torch()
    if gate_count <= 0 or not bool(cfg.get("enabled", False)):
        return {}
    amplitude = max(0.0, float(cfg.get("amplitude", 0.0)))
    if amplitude <= 0.0:
        return {}
    target_pops = _resolve_context_target_populations(
        engine_build=engine_build,
        targets=cast(Sequence[Any], cfg.get("targets", ("hidden",))),
    )
    if not target_pops:
        return {}
    compartment = _compartment_from_name(str(cfg.get("compartment", "dendrite")))
    cache: dict[int, dict[str, Mapping[Compartment, Any]]] = {}
    for gate_index in range(gate_count):
        pop_drive: dict[str, Mapping[Compartment, Any]] = {}
        for pop_name in target_pops:
            pop_state = engine_build.engine._pop_states.get(pop_name)
            if pop_state is None:
                continue
            n = int(pop_state.spikes.numel())
            if n <= 0:
                continue
            drive = _build_gate_context_vector(
                n=n,
                gate_index=gate_index,
                gate_count=gate_count,
                amplitude=amplitude,
                device=pop_state.spikes.device,
                dtype=torch.float32
                if pop_state.spikes.dtype == torch.bool
                else pop_state.spikes.dtype,
            )
            pop_drive[pop_name] = {compartment: drive}
        cache[int(gate_index)] = pop_drive
    return cache


def _activate_curriculum_gate_context(
    *,
    engine_build: _EngineBuild,
    gate_context_cache: Mapping[int, Mapping[str, Mapping[Compartment, Any]]],
    gate_index: int,
) -> None:
    engine_build.context_drive_by_population.clear()
    if not gate_context_cache:
        return
    pop_drive = gate_context_cache.get(int(gate_index))
    if not pop_drive:
        return
    engine_build.context_drive_by_population.update(pop_drive)


def _set_action_force_drive(
    *,
    engine_build: _EngineBuild,
    chosen_action: int,
    amplitude: float,
    compartment: str,
) -> bool:
    torch = require_torch()
    if float(amplitude) <= 0.0:
        return False
    pop_state = engine_build.engine._pop_states.get(engine_build.output_population)
    if pop_state is None:
        return False
    spikes = pop_state.spikes
    n_out = int(spikes.numel())
    if n_out <= 0:
        return False
    action_idx = int(chosen_action)
    if action_idx < 0 or action_idx >= n_out:
        return False
    drive = torch.zeros(
        (n_out,),
        device=spikes.device,
        dtype=torch.float32 if spikes.dtype == torch.bool else spikes.dtype,
    )
    # Amplify chosen output
    drive[action_idx] = float(amplitude)
    # Suppress other outputs to create output separation
    suppression_strength = -0.25 * float(amplitude)  # Slightly negative to suppress alternatives
    for i in range(n_out):
        if i != action_idx:
            drive[i] = suppression_strength

    engine_build.action_drive_by_population.clear()
    engine_build.action_drive_by_population[engine_build.output_population] = {
        _compartment_from_name(compartment): drive
    }
    return True


def _clear_action_force_drive(*, engine_build: _EngineBuild) -> None:
    engine_build.action_drive_by_population.clear()


def _discard_pending_releases(*, pending_releases: dict[ModulatorKind, float]) -> None:
    if not pending_releases:
        return
    for kind in tuple(pending_releases.keys()):
        pending_releases[kind] = 0.0


def _resolve_context_target_populations(
    *,
    engine_build: _EngineBuild,
    targets: Sequence[Any],
) -> tuple[str, ...]:
    available = (
        engine_build.input_population,
        *engine_build.hidden_populations,
        engine_build.output_population,
    )
    lookup = {name.strip().lower(): name for name in available}
    resolved: list[str] = []

    def _append(name: str) -> None:
        if name not in resolved:
            resolved.append(name)

    for raw in targets:
        token = str(raw).strip().lower()
        if not token:
            continue
        if token == "hidden":
            for pop_name in engine_build.hidden_populations:
                _append(pop_name)
            continue
        if token in {"out", "output"}:
            _append(engine_build.output_population)
            continue
        if token in {"in", "input"}:
            _append(engine_build.input_population)
            continue
        match = lookup.get(token)
        if match is not None:
            _append(match)
    if resolved:
        return tuple(resolved)
    return tuple(engine_build.hidden_populations)


def _build_gate_context_vector(
    *,
    n: int,
    gate_index: int,
    gate_count: int,
    amplitude: float,
    device: Any,
    dtype: Any,
) -> Any:
    torch = require_torch()
    out = torch.zeros((int(n),), device=device, dtype=dtype)
    if int(n) <= 0 or gate_count <= 0 or amplitude <= 0.0:
        return out
    idx = int(gate_index) % max(1, int(gate_count))
    if int(gate_count) <= int(n):
        out[idx] = float(amplitude)
        return out
    primary = idx % int(n)
    secondary = (idx // int(n)) % int(n)
    out[primary] = float(amplitude)
    if secondary != primary:
        out[secondary] = float(amplitude) * 0.5
    return out


def _compartment_from_name(value: str) -> Compartment:
    token = str(value).strip().lower()
    if token == "dendrite":
        return Compartment.DENDRITE
    if token == "ais":
        return Compartment.AIS
    if token == "axon":
        return Compartment.AXON
    return Compartment.SOMA


def _learning_stats(
    *,
    engine: TorchNetworkEngine,
    learning_proj_names: tuple[str, ...],
) -> tuple[float, float, float, float, float]:
    if not learning_proj_names:
        return 0.0, 0.0, float("nan"), float("nan"), float("nan")

    # Aggregate across all learning projections
    all_eligibilities = []
    all_d_weights = []
    all_weights = []

    for proj_name in learning_proj_names:
        proj_state = engine._proj_states.get(proj_name)
        proj_spec = next((spec for spec in engine._proj_specs if spec.name == proj_name), None)
        if proj_spec is not None and proj_spec.learning is not None and proj_state is not None:
            learning_state = proj_state.learning_state
            if learning_state is not None:
                tensors = proj_spec.learning.state_tensors(learning_state)
                eligibility = tensors.get("eligibility")
                if eligibility is not None and int(eligibility.numel()) > 0:
                    all_eligibilities.append(eligibility.abs())

        d_weights = engine.last_d_weights.get(proj_name)
        if d_weights is not None and int(d_weights.numel()) > 0:
            all_d_weights.append(d_weights.abs())

        weights = None
        if proj_state is not None:
            weights = getattr(proj_state.state, "weights", None)
        if weights is None and proj_spec is not None:
            weights = proj_spec.topology.weights
        if weights is not None and int(weights.numel()) > 0:
            all_weights.append(weights)

    mean_eligibility_abs = 0.0
    if all_eligibilities:
        torch_local = require_torch()
        stacked = torch_local.cat(all_eligibilities, dim=0)
        mean_eligibility_abs = float(stacked.mean().item())

    mean_abs_dw = 0.0
    if all_d_weights:
        torch_local = require_torch()
        stacked = torch_local.cat(all_d_weights, dim=0)
        mean_abs_dw = float(stacked.mean().item())

    if not all_weights:
        return mean_eligibility_abs, mean_abs_dw, float("nan"), float("nan"), float("nan")

    torch_local = require_torch()
    all_weights_flat = torch_local.cat([w.flatten() for w in all_weights], dim=0)
    return (
        mean_eligibility_abs,
        mean_abs_dw,
        float(all_weights_flat.min().item()),
        float(all_weights_flat.max().item()),
        float(all_weights_flat.mean().item()),
    )


def _resolve_monitors_enabled(run_spec: Mapping[str, Any]) -> bool:
    explicit = _coerce_optional_bool(run_spec.get("monitors_enabled"))
    return True if explicit is None else bool(explicit)


def _create_monitor_writers(
    *,
    engine: TorchNetworkEngine,
    engine_build: _EngineBuild,
    run_spec: Mapping[str, Any],
    run_dir: Path,
    flush_every: int,
) -> _MonitorWriters | None:
    if not _resolve_monitors_enabled(run_spec):
        return None
    spikes_sink = CsvSink(run_dir / "spikes.csv", flush_every=max(1, flush_every))
    weights_sink = CsvSink(run_dir / "weights.csv", flush_every=max(1, flush_every))
    spike_populations = (
        engine_build.input_population,
        *engine_build.hidden_populations,
        engine_build.output_population,
    )
    return _MonitorWriters(
        spikes_sink=spikes_sink,
        weights_sink=weights_sink,
        spike_populations=spike_populations,
        weight_projections=_prepare_weight_export_projections(engine=engine),
    )


def _prepare_weight_export_projections(
    *, engine: TorchNetworkEngine
) -> tuple[_WeightExportProjection, ...]:
    torch = require_torch()
    projections: list[_WeightExportProjection] = []
    for projection in engine._proj_specs:
        pre_idx = projection.topology.pre_idx
        post_idx = projection.topology.post_idx
        if pre_idx is None or post_idx is None:
            continue
        projections.append(
            _WeightExportProjection(
                name=projection.name,
                pre_idx=pre_idx.detach().to(device="cpu", dtype=torch.long).reshape(-1),
                post_idx=post_idx.detach().to(device="cpu", dtype=torch.long).reshape(-1),
            )
        )
    return tuple(projections)


def _write_spike_events_for_step(
    *,
    engine: TorchNetworkEngine,
    monitor_writers: _MonitorWriters | None,
    step: int,
) -> None:
    if monitor_writers is None or monitor_writers.spikes_sink is None:
        return
    torch = require_torch()
    for pop_name in monitor_writers.spike_populations:
        pop_state = engine._pop_states.get(pop_name)
        if pop_state is None:
            continue
        spikes = pop_state.spikes
        if spikes is None or int(spikes.numel()) == 0:
            continue
        active = torch.nonzero(spikes, as_tuple=False).reshape(-1)
        if int(active.numel()) == 0:
            continue
        active_cpu = active.detach().to(device="cpu", dtype=torch.long)
        for neuron_idx in active_cpu.tolist():
            monitor_writers.spikes_sink.write_row(
                {
                    "step": int(step),
                    "pop": pop_name,
                    "neuron": int(neuron_idx),
                }
            )


def _write_weights_snapshot(
    *,
    engine: TorchNetworkEngine,
    monitor_writers: _MonitorWriters | None,
    step: int,
) -> None:
    if monitor_writers is None or monitor_writers.weights_sink is None:
        return
    for export in monitor_writers.weight_projections:
        proj_state = engine._proj_states.get(export.name)
        weights = getattr(proj_state.state, "weights", None) if proj_state is not None else None
        if weights is None:
            projection = next(
                (proj for proj in engine._proj_specs if proj.name == export.name), None
            )
            weights = projection.topology.weights if projection is not None else None
        if weights is None or int(weights.numel()) == 0:
            continue
        weights_cpu = weights.detach().to(device="cpu").reshape(-1)
        n_edges = min(
            int(weights_cpu.numel()),
            int(export.pre_idx.numel()),
            int(export.post_idx.numel()),
        )
        if n_edges <= 0:
            continue
        for edge_idx in range(n_edges):
            monitor_writers.weights_sink.write_row(
                {
                    "step": int(step),
                    "proj": export.name,
                    "pre": int(export.pre_idx[edge_idx].item()),
                    "post": int(export.post_idx[edge_idx].item()),
                    "w": float(weights_cpu[edge_idx].item()),
                }
            )


def _close_monitor_writers(monitor_writers: _MonitorWriters | None) -> None:
    if monitor_writers is None:
        return
    if monitor_writers.spikes_sink is not None:
        monitor_writers.spikes_sink.close()
    if monitor_writers.weights_sink is not None:
        monitor_writers.weights_sink.close()


_ACH_INPUT_PULSE_SCALE = 0.35
_NA_SURPRISE_PULSE_SCALE = 0.70
_SEROTONIN_BASELINE_SCALE = 0.05
_SEROTONIN_CORRECT_PULSE_SCALE = 0.20


def _queue_trial_feedback_releases(
    *,
    pending_releases: Mapping[ModulatorKind, float] | dict[ModulatorKind, float],
    modulator_kinds: Sequence[ModulatorKind],
    modulator_amount: float,
    correct: bool,
) -> float:
    if not modulator_kinds or modulator_amount == 0.0:
        return 0.0
    pending = cast(dict[ModulatorKind, float], pending_releases)
    base = abs(float(modulator_amount))
    dopamine_pulse = 0.0

    if ModulatorKind.DOPAMINE in modulator_kinds:
        dopamine_pulse = base if bool(correct) else -base
        pending[ModulatorKind.DOPAMINE] = (
            float(pending.get(ModulatorKind.DOPAMINE, 0.0)) + dopamine_pulse
        )
    if ModulatorKind.NORADRENALINE in modulator_kinds and not bool(correct):
        na_pulse = base * _NA_SURPRISE_PULSE_SCALE
        pending[ModulatorKind.NORADRENALINE] = (
            float(pending.get(ModulatorKind.NORADRENALINE, 0.0)) + na_pulse
        )
    if ModulatorKind.SEROTONIN in modulator_kinds and bool(correct):
        serotonin_pulse = base * _SEROTONIN_CORRECT_PULSE_SCALE
        pending[ModulatorKind.SEROTONIN] = (
            float(pending.get(ModulatorKind.SEROTONIN, 0.0)) + serotonin_pulse
        )
    return float(dopamine_pulse)


def _build_modulator_releases_for_step(
    *,
    pending_releases: Mapping[ModulatorKind, float] | dict[ModulatorKind, float],
    modulator_kinds: Sequence[ModulatorKind],
    modulator_amount: float,
    input_active: bool,
    ach_pulse_emitted: bool,
    field_type: str,
    world_extent: tuple[float, float],
    output_positions: Any,
    dopamine_focus_action: int | None,
    device: Any,
    dtype: Any,
) -> tuple[list[ModulatorRelease], bool]:
    torch = require_torch()
    if not modulator_kinds:
        return [], ach_pulse_emitted
    pending = cast(dict[ModulatorKind, float], pending_releases)
    releases: list[ModulatorRelease] = []
    base = abs(float(modulator_amount))

    for kind in modulator_kinds:
        amount = float(pending.get(kind, 0.0))
        if input_active and kind == ModulatorKind.ACETYLCHOLINE and not ach_pulse_emitted:
            amount += base * _ACH_INPUT_PULSE_SCALE
            ach_pulse_emitted = True
        if input_active and kind == ModulatorKind.SEROTONIN:
            amount += base * _SEROTONIN_BASELINE_SCALE
        pending[kind] = 0.0
        if amount == 0.0:
            continue

        # Use action-specific dopamine positions if this is dopamine and focus_action is set
        if kind == ModulatorKind.DOPAMINE and dopamine_focus_action is not None:
            positions = _release_positions_for_field(
                field_type=field_type,
                world_extent=world_extent,
                output_positions=output_positions,
                dopamine_focus_action=int(dopamine_focus_action),
                device=device,
                dtype=dtype,
            )
        else:
            positions = _release_positions_for_field(
                field_type=field_type,
                world_extent=world_extent,
                output_positions=output_positions,
                dopamine_focus_action=None,
                device=device,
                dtype=dtype,
            )

        releases.append(
            ModulatorRelease(
                kind=kind,
                positions=positions,
                amount=torch.tensor([amount], device=device, dtype=dtype),
            )
        )
    return releases, ach_pulse_emitted


def _release_positions_for_field(
    *,
    field_type: str,
    world_extent: tuple[float, float],
    output_positions: Any,
    dopamine_focus_action: int | None,
    device: Any,
    dtype: Any,
) -> Any:
    torch = require_torch()
    if str(field_type).strip().lower() == "grid_diffusion_2d":
        # For grid_diffusion_2d with action-specific dopamine, return only the chosen output position
        if dopamine_focus_action is not None and output_positions is not None:
            n_outputs = int(output_positions.shape[0]) if hasattr(output_positions, "shape") else 0
            action_idx = int(dopamine_focus_action)
            if 0 <= action_idx < n_outputs:
                return cast(
                    Any,
                    output_positions[action_idx : action_idx + 1].to(device=device, dtype=dtype),
                )
        # Fallback: return all positions
        return cast(Any, output_positions.to(device=device, dtype=dtype))

    # For global_scalar, use center location by default
    # For action-specific targeting, use the chosen output's position if available
    if dopamine_focus_action is not None and output_positions is not None:
        n_outputs = int(output_positions.shape[0]) if hasattr(output_positions, "shape") else 0
        action_idx = int(dopamine_focus_action)
        if 0 <= action_idx < n_outputs:
            return cast(
                Any, output_positions[action_idx : action_idx + 1].to(device=device, dtype=dtype)
            )

    # Default center position
    center = torch.tensor(
        [[0.5 * float(world_extent[0]), 0.5 * float(world_extent[1]), 0.0]],
        device=device,
        dtype=dtype,
    )
    return cast(Any, center)


def _build_encoded_input_table(*, inputs: Any, dt: float) -> Any:
    torch = require_torch()
    encoded = torch.zeros(
        (int(inputs.shape[0]), len(INPUT_NEURON_INDICES)),
        device=inputs.device,
        dtype=inputs.dtype,
    )
    for idx in range(int(inputs.shape[0])):
        drive = encode_inputs(
            inputs[idx],
            mode="rate",
            dt=dt,
            high=1.0,
            low=0.0,
            compartment=Compartment.SOMA,
        )
        encoded[idx].copy_(drive[Compartment.SOMA])
    return encoded


def _apply_gate_context_drive(input_drive: Any, *, gate: LogicGate | str) -> None:
    gate_idx = INPUT_NEURON_INDICES.get("gate_context")
    if gate_idx is None:
        return
    idx = int(gate_idx)
    if idx < 0 or idx >= int(input_drive.numel()):
        return
    input_drive[idx] = float(
        gate_context_level_for_gate(gate.value if isinstance(gate, LogicGate) else gate)
    )


def _gate_context_drive_value(input_drive: Any) -> float:
    gate_idx = INPUT_NEURON_INDICES.get("gate_context")
    if gate_idx is None:
        return 0.0
    idx = int(gate_idx)
    if idx < 0 or idx >= int(input_drive.numel()):
        return 0.0
    return float(input_drive[idx].item())


def run_logic_gate_curriculum_engine(
    config: LogicGateRunConfig,
    run_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Run curriculum-style logic-gate training with one persistent engine."""
    engine_run_spec = _effective_curriculum_engine_run_spec(run_spec=run_spec, config=config)
    torch = require_torch()
    gate_sequence = _resolve_curriculum_gates(engine_run_spec)
    if not gate_sequence:
        raise ValueError("At least one gate is required for curriculum mode.")
    replay_ratio = _clamp(
        float(engine_run_spec.get("logic_curriculum_replay_ratio", 0.35)), 0.0, 1.0
    )

    device = _resolve_device(torch, str(config.device))
    run_dir = _resolve_curriculum_run_dir(config, gates=gate_sequence)
    phase_trials = int(config.steps)
    if phase_trials <= 0:
        raise ValueError("phase steps must be > 0")

    torch.manual_seed(int(config.seed))
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.seed))
    sampling_generator = torch.Generator(device="cpu")
    sampling_generator.manual_seed(int(config.seed))

    engine_build = _build_engine(
        config=config,
        gate=_curriculum_topology_gate(gate_sequence),
        run_spec=engine_run_spec,
        device=device,
    )
    engine = engine_build.engine
    _write_engine_topology_json(engine=engine, run_dir=run_dir)

    inputs, _ = make_truth_table(LogicGate.AND, device=device, dtype="float32")
    encoded_inputs = _build_encoded_input_table(inputs=inputs, dt=config.dt)
    targets_by_gate: dict[LogicGate, Any] = {}
    for gate in gate_sequence:
        _, targets = make_truth_table(gate, device=device, dtype=inputs.dtype)
        targets_by_gate[gate] = targets.reshape(4)

    predictions_by_gate: dict[LogicGate, Any] = {
        gate: torch.zeros((4,), device=inputs.device, dtype=inputs.dtype) for gate in gate_sequence
    }
    exploration_cfg = _resolve_exploration_cfg(engine_run_spec, default_cfg=config.exploration)
    decision_rng = random.Random(int(exploration_cfg.seed))
    reward_delivery_steps, reward_delivery_clamp_input = _resolve_reward_window_cfg(
        engine_run_spec,
        default_steps=config.reward_delivery_steps,
        default_clamp_input=config.reward_delivery_clamp_input,
    )
    action_force_cfg = _resolve_action_force_cfg(
        engine_run_spec,
        default_enabled=True,
        default_window="reward_window",
        default_steps=1,
        default_amplitude=0.75,
        default_compartment="soma",
    )
    gate_context_cfg = _resolve_curriculum_gate_context_cfg(
        engine_run_spec,
        default_cfg=config.curriculum_gate_context,
    )
    gate_to_index = {gate_value: idx for idx, gate_value in enumerate(gate_sequence)}
    gate_context_cache = _build_curriculum_gate_context_cache(
        engine_build=engine_build,
        gate_count=len(gate_sequence),
        cfg=gate_context_cfg,
    )

    monitors_enabled = _resolve_monitors_enabled(engine_run_spec)
    trial_flush_every = 1 if monitors_enabled else max(1, config.export_every)
    trial_sink = CsvSink(run_dir / "trials.csv", flush_every=trial_flush_every)
    eval_sink = CsvSink(run_dir / "eval.csv", flush_every=1)
    confusion_sink = CsvSink(run_dir / "confusion.csv", flush_every=1)
    phase_sink = CsvSink(run_dir / "phase_summary.csv", flush_every=1)
    last_trials: deque[dict[str, Any]] | None = (
        deque(maxlen=config.dump_last_trials_n) if config.dump_last_trials_csv else None
    )
    monitor_writers = _create_monitor_writers(
        engine=engine,
        engine_build=engine_build,
        run_spec=engine_run_spec,
        run_dir=run_dir,
        flush_every=max(1, config.export_every),
    )

    debug_every = max(1, int(config.debug_every))
    last_trials_csv: Path | None = None
    last_pred: int | None = None
    rolling_correct: deque[int] = deque(maxlen=100)
    sampled_correct_total = 0
    global_trial = 0
    sim_step = 0
    t0 = perf_counter()
    phase_results: list[dict[str, Any]] = []

    try:
        for phase_idx, gate in enumerate(gate_sequence, start=1):
            targets_flat = targets_by_gate[gate]
            train_gate_idx = _build_curriculum_gate_indices(
                phase_index=phase_idx - 1,
                phase_trials=phase_trials,
                replay_ratio=replay_ratio,
                device="cpu",
            )
            phase_correct = 0
            phase_gate_correct = 0
            phase_gate_sample_count = 0
            tracker = PassTracker(gate)
            first_pass_local: int | None = None
            phase_eval_last = 0.0
            case_indices = sample_case_indices(
                phase_trials,
                method=config.sampling_method,
                generator=sampling_generator,
                device="cpu",
            )
            case_sequence = [int(value) for value in case_indices.tolist()]

            for local_trial, case_idx in enumerate(case_sequence, start=1):
                global_trial += 1
                active_gate = gate_sequence[int(train_gate_idx[local_trial - 1].item())]
                train_targets = targets_by_gate[active_gate]
                target_value = train_targets[case_idx]
                _activate_curriculum_gate_context(
                    engine_build=engine_build,
                    gate_context_cache=gate_context_cache,
                    gate_index=int(gate_to_index.get(active_gate, 0)),
                )

                input_drive = encoded_inputs[case_idx].clone()
                _apply_gate_context_drive(input_drive, gate=active_gate)
                out_spike_counts, hidden_spike_counts, sim_step = _run_trial_steps(
                    engine_build=engine_build,
                    input_drive=input_drive,
                    sim_steps_per_trial=config.sim_steps_per_trial,
                    step_offset=sim_step,
                    monitor_writers=monitor_writers,
                )
                pred_bit, action_info = select_action_wta(
                    out_spike_counts,
                    last_action=last_pred,
                    exploration=exploration_cfg,
                    trial_idx=global_trial - 1,
                    is_eval=False,
                    rng=decision_rng,
                )
                chosen_action = int(pred_bit)
                engine_build.modulator_state["dopamine_focus_action"] = chosen_action

                target_bit = int((target_value >= 0.5).item())
                correct = int(chosen_action == target_bit)
                phase_correct += correct
                if active_gate == gate:
                    phase_gate_correct += correct
                    phase_gate_sample_count += 1
                sampled_correct_total += correct
                rolling_correct.append(correct)
                trial_acc_rolling = sum(rolling_correct) / float(len(rolling_correct))

                predictions_by_gate[active_gate][case_idx] = float(chosen_action)
                action_forced = 0
                action_drive_amplitude = 0.0
                action_window = str(action_force_cfg["window"])
                action_steps = int(action_force_cfg["steps"])
                dopamine_pulse = _queue_trial_feedback_releases(
                    pending_releases=engine_build.pending_releases,
                    modulator_kinds=engine_build.modulator_kinds,
                    modulator_amount=engine_build.modulator_amount,
                    correct=bool(correct),
                )
                if reward_delivery_steps > 0:
                    should_force_action = (
                        bool(action_force_cfg["enabled"])
                        and _should_commit_action_force(
                            action_info=action_info,
                            chosen_action=chosen_action,
                            out_spike_counts=out_spike_counts,
                            mode=str(action_force_cfg.get("mode", "explore_or_silent")),
                        )
                        and action_forced == 0
                        and action_window in {"reward_window", "post_decision"}
                        and int(action_steps) > 0
                    )
                    reward_input = _reward_window_input_drive(
                        input_drive=input_drive,
                        clamp_input=reward_delivery_clamp_input and not should_force_action,
                    )
                    if should_force_action:
                        forced = _set_action_force_drive(
                            engine_build=engine_build,
                            chosen_action=chosen_action,
                            amplitude=float(action_force_cfg["amplitude"]),
                            compartment=str(action_force_cfg["compartment"]),
                        )
                        if forced:
                            action_forced = 1
                            action_drive_amplitude = float(action_force_cfg["amplitude"])
                    force_steps = (
                        min(max(0, int(action_steps)), max(0, int(reward_delivery_steps)))
                        if action_forced
                        else 0
                    )
                    reward_out_counts, _, sim_step = _run_trial_steps(
                        engine_build=engine_build,
                        input_drive=reward_input,
                        sim_steps_per_trial=reward_delivery_steps,
                        action_force_steps=force_steps,
                        step_offset=sim_step,
                        monitor_writers=monitor_writers,
                    )
                    if action_forced and action_window == "reward_window":
                        out_spike_counts.add_(reward_out_counts)
                else:
                    _discard_pending_releases(pending_releases=engine_build.pending_releases)
                (
                    mean_eligibility_abs,
                    mean_abs_dw,
                    weights_min,
                    weights_max,
                    weights_mean,
                ) = _learning_stats(
                    engine=engine,
                    learning_proj_names=engine_build.learning_proj_names,
                )
                hidden_mean_spikes = float(hidden_spike_counts.mean().item()) / float(
                    config.sim_steps_per_trial
                )
                out0 = float(out_spike_counts[OUTPUT_NEURON_INDICES["class_0"]].item())
                out1 = float(out_spike_counts[OUTPUT_NEURON_INDICES["class_1"]].item())
                tie_behavior = int(bool(action_info.get("tie", False)))
                no_spikes = int((out0 + out1) <= 0.0)

                trial_row = {
                    "phase": phase_idx,
                    "gate": gate.value,
                    "train_gate": active_gate.value,
                    "phase_trial": local_trial,
                    "trial": global_trial,
                    "sim_step_end": sim_step,
                    "case_idx": case_idx,
                    "x0": float(inputs[case_idx, 0].item()),
                    "x1": float(inputs[case_idx, 1].item()),
                    "in_bit0_0_drive": float(input_drive[INPUT_NEURON_INDICES["bit0_0"]].item()),
                    "in_bit0_1_drive": float(input_drive[INPUT_NEURON_INDICES["bit0_1"]].item()),
                    "in_bit1_0_drive": float(input_drive[INPUT_NEURON_INDICES["bit1_0"]].item()),
                    "in_bit1_1_drive": float(input_drive[INPUT_NEURON_INDICES["bit1_1"]].item()),
                    "in_gate_context_drive": _gate_context_drive_value(input_drive),
                    "out_spikes_0": out0,
                    "out_spikes_1": out1,
                    "hidden_mean_spikes": hidden_mean_spikes,
                    "dopamine_pulse": dopamine_pulse,
                    "trial_acc_rolling": trial_acc_rolling,
                    "mean_eligibility_abs": mean_eligibility_abs,
                    "mean_abs_dw": mean_abs_dw,
                    "weights_min": weights_min,
                    "weights_max": weights_max,
                    "weights_mean": weights_mean,
                    "tie_wta": tie_behavior,
                    "epsilon": float(action_info.get("epsilon", 0.0)),
                    "explored": int(bool(action_info.get("explored", False))),
                    "greedy_action": int(action_info.get("greedy", chosen_action)),
                    "chosen_action": chosen_action,
                    "action_forced": int(action_forced),
                    "action_drive_amplitude": float(action_drive_amplitude),
                    "gate_context_enabled": int(bool(gate_context_cfg["enabled"])),
                    "gate_context_index": int(gate_to_index.get(active_gate, -1))
                    if bool(gate_context_cfg["enabled"])
                    else -1,
                    "gate_context_amplitude": float(gate_context_cfg["amplitude"]),
                    "no_output_spikes": no_spikes,
                    "target": target_bit,
                    "pred": chosen_action,
                    "correct": correct,
                }
                trial_sink.write_row(trial_row)
                if last_trials is not None:
                    last_trials.append(dict(trial_row))

                current_preds = predictions_by_gate[gate]
                phase_eval_last = float(eval_accuracy(current_preds, targets_flat))
                tracker.update(phase_eval_last)
                if tracker.passed and first_pass_local is None:
                    first_pass_local = local_trial

                if (local_trial % config.export_every) == 0 or local_trial == phase_trials:
                    sample_acc_phase = phase_correct / float(local_trial)
                    sample_acc_phase_gate = (
                        phase_gate_correct / float(phase_gate_sample_count)
                        if phase_gate_sample_count > 0
                        else 0.0
                    )
                    sample_acc_global = sampled_correct_total / float(global_trial)
                    global_eval_by_gate: dict[str, float] = {}
                    for eval_gate in gate_sequence:
                        global_eval_by_gate[eval_gate.value] = float(
                            eval_accuracy(
                                predictions_by_gate[eval_gate], targets_by_gate[eval_gate]
                            )
                        )
                    global_eval_accuracy = sum(global_eval_by_gate.values()) / float(
                        len(global_eval_by_gate)
                    )
                    eval_acc_full, confusion = eval_accuracy(
                        current_preds,
                        targets_flat,
                        report_confusion=True,
                    )
                    eval_sink.write_row(
                        {
                            "phase": phase_idx,
                            "gate": gate.value,
                            "phase_trial": local_trial,
                            "trial": global_trial,
                            "eval_accuracy": eval_acc_full,
                            "sample_accuracy": sample_acc_phase,
                            "sample_accuracy_phase_gate": sample_acc_phase_gate,
                            "sample_accuracy_global": sample_acc_global,
                            "global_eval_accuracy": global_eval_accuracy,
                            "trial_acc_rolling": trial_acc_rolling,
                            "pred_00": int(current_preds[0].item()),
                            "pred_01": int(current_preds[1].item()),
                            "pred_10": int(current_preds[2].item()),
                            "pred_11": int(current_preds[3].item()),
                            "mean_eligibility_abs": mean_eligibility_abs,
                            "mean_abs_dw": mean_abs_dw,
                            "weights_min": weights_min,
                            "weights_max": weights_max,
                            "weights_mean": weights_mean,
                            "perfect_streak": tracker.perfect_streak,
                            "high_streak": tracker.high_streak,
                            "passed": int(tracker.passed),
                            **{
                                f"eval_{name}": value for name, value in global_eval_by_gate.items()
                            },
                        }
                    )
                    confusion_sink.write_row(
                        {
                            "phase": phase_idx,
                            "gate": gate.value,
                            "phase_trial": local_trial,
                            "trial": global_trial,
                            **confusion,
                        }
                    )
                    _write_weights_snapshot(
                        engine=engine,
                        monitor_writers=monitor_writers,
                        step=sim_step,
                    )

                if config.debug and (
                    local_trial == 1
                    or local_trial == phase_trials
                    or local_trial % debug_every == 0
                ):
                    print(
                        f"[logic-curriculum-engine] phase={phase_idx} gate={gate.value} "
                        f"trial={local_trial}/{phase_trials} train_gate={active_gate.value} "
                        f"target={target_bit} pred={chosen_action} correct={correct}"
                    )

                last_pred = chosen_action

            phase_duration = perf_counter() - t0
            phase_sample_acc = phase_correct / float(phase_trials)
            phase_current_gate_sample_acc = (
                phase_gate_correct / float(phase_gate_sample_count)
                if phase_gate_sample_count > 0
                else 0.0
            )
            phase_replay_trials = max(0, phase_trials - phase_gate_sample_count)
            phase_row = {
                "phase": phase_idx,
                "gate": gate.value,
                "phase_trials": phase_trials,
                "phase_gate_trials": phase_gate_sample_count,
                "replay_trials": phase_replay_trials,
                "replay_ratio": replay_ratio,
                "global_trial_end": global_trial,
                "eval_accuracy": phase_eval_last,
                "sample_accuracy": phase_sample_acc,
                "sample_accuracy_phase_gate": phase_current_gate_sample_acc,
                "passed": int(tracker.passed),
                "first_pass_phase_trial": first_pass_local if first_pass_local is not None else "",
                "elapsed_s_since_start": phase_duration,
            }
            phase_sink.write_row(phase_row)
            phase_results.append(dict(phase_row))
    finally:
        trial_sink.close()
        eval_sink.close()
        confusion_sink.close()
        phase_sink.close()
        _close_monitor_writers(monitor_writers)

    if last_trials is not None and len(last_trials) > 0:
        last_trials_csv = run_dir / f"trials_last_{config.dump_last_trials_n}.csv"
        last_sink = CsvSink(last_trials_csv, flush_every=1)
        try:
            for row in last_trials:
                last_sink.write_row(row)
        finally:
            last_sink.close()

    elapsed_s = perf_counter() - t0
    final_eval_by_gate: dict[str, float] = {}
    for gate in gate_sequence:
        final_eval_by_gate[gate.value] = float(
            eval_accuracy(predictions_by_gate[gate], targets_by_gate[gate])
        )
    final_gate = gate_sequence[-1]
    preds = predictions_by_gate[final_gate]

    return {
        "out_dir": run_dir,
        "device": device,
        "learning_mode": config.learning_mode,
        "topology": "engine_ff",
        "gates": [gate.value for gate in gate_sequence],
        "phase_steps": phase_trials,
        "replay_ratio": replay_ratio,
        "total_steps": global_trial,
        "sim_steps_per_trial": config.sim_steps_per_trial,
        "inputs": inputs,
        "preds": preds,
        "case_scores": preds.clone(),
        "final_eval_by_gate": final_eval_by_gate,
        "final_gate": final_gate.value,
        "elapsed_s": elapsed_s,
        "phase_results": phase_results,
        "input_neuron_indices": dict(INPUT_NEURON_INDICES),
        "output_neuron_indices": dict(OUTPUT_NEURON_INDICES),
        "debug_every": config.debug_every,
        "last_trials_csv": last_trials_csv,
        "trials_csv": run_dir / "trials.csv",
        "eval_csv": run_dir / "eval.csv",
        "confusion_csv": run_dir / "confusion.csv",
        "phase_summary_csv": run_dir / "phase_summary.csv",
        "topology_json": run_dir / "topology.json",
    }


def _build_engine(
    *,
    config: LogicGateRunConfig,
    gate: LogicGate,
    run_spec: Mapping[str, Any],
    device: str,
) -> _EngineBuild:
    torch = require_torch()
    logic_cfg = _as_mapping(run_spec.get("logic"))
    learning_learn_every = _coerce_positive_int(
        _first_non_none(
            logic_cfg.get("learn_every"),
            run_spec.get("logic_learn_every"),
            config.learn_every_default,
        ),
        config.learn_every_default,
    )
    learning_cfg = _resolve_learning_cfg(
        run_spec,
        default_enabled=config.learning_mode == "rstdp",
        default_rule=str(config.engine_learning_rule),
        default_lr=0.1,
        default_modulator_kind=str(config.learning_modulator_kind),
    )
    wrapper_cfg = _resolve_wrapper_cfg(
        run_spec,
        default_cfg=cast(Mapping[str, Any], asdict(config.wrapper)),
    )
    mods_cfg = _resolve_modulators_cfg(
        run_spec,
        default_cfg=cast(Mapping[str, Any], asdict(config.modulators)),
    )
    homeostasis_cfg = _resolve_homeostasis_cfg(run_spec)
    pruning_cfg = _resolve_pruning_cfg(run_spec)
    neuro_cfg = _resolve_neurogenesis_cfg(run_spec)
    excitability_cfg = _resolve_excitability_cfg(
        run_spec,
        default_cfg=cast(Mapping[str, Any], asdict(config.excitability_modulation)),
    )

    seed = int(config.seed)
    dtype = torch.float32

    if gate in {LogicGate.XOR, LogicGate.XNOR}:
        _, topology, handles = build_logic_gate_xor(
            device=device,
            seed=seed,
            neuron_model=config.neuron_model,
            advanced_synapse=config.advanced_synapse,
            run_spec=run_spec,
        )
    else:
        _, topology, handles = build_logic_gate_ff(
            gate,
            device=device,
            seed=seed,
            neuron_model=config.neuron_model,
            advanced_synapse=config.advanced_synapse,
            run_spec=run_spec,
        )

    populations: tuple[PopulationSpec, ...] = tuple(topology.populations)
    base_projections = tuple(topology.projections)

    learning_rule = _make_learning_rule(learning_cfg)
    learning_rule = _maybe_wrap_learning_rule(learning_rule, wrapper_cfg)
    learning_enabled = learning_rule is not None

    # Resolve learning targets (exact names or globs like "*Excit->Out")
    learning_targets = _resolve_learning_targets(
        projections=base_projections,
        targets=learning_cfg.get("targets"),
        output_population=handles.output_population,
    )

    projections, learning_proj_names, learning_proj_name = _apply_learning_projection(
        base_projections=base_projections,
        learning_rule=learning_rule,
        learning_targets=learning_targets,
        output_population=handles.output_population,
        learn_every=learning_learn_every,
    )
    _validate_advanced_synapse_prerequisites(projections=projections)

    current_input_drive: dict[str, Any] = {"tensor": None}
    context_drive_by_population: dict[str, Mapping[Compartment, Any]] = {}
    action_drive_by_population: dict[str, Mapping[Compartment, Any]] = {}
    pending_releases: dict[ModulatorKind, float] = {}
    mod_specs: list[ModulatorSpec] = []
    modulator_kinds: tuple[ModulatorKind, ...] = ()
    modulator_amount = float(mods_cfg["amount"])
    modulator_state: dict[str, Any] = {}
    releases_fn = None
    if mods_cfg["enabled"]:
        modulator_kinds = _coerce_modulator_kinds(cast(Sequence[Any], mods_cfg["kinds"]))
        if not modulator_kinds:
            modulator_kinds = (ModulatorKind.DOPAMINE,)
        field_type = str(mods_cfg["field_type"])
        world_extent = cast(tuple[float, float], mods_cfg["world_extent"])
        output_release_positions = _output_positions_for_population(
            populations=populations,
            output_population=handles.output_population,
            device=torch.device(device),
            dtype=dtype,
        )
        field: IModulatorField
        if field_type == "grid_diffusion_2d":
            field = GridDiffusion2DField(
                params=GridDiffusion2DParams(
                    kinds=modulator_kinds,
                    grid_size=cast(tuple[int, int], mods_cfg["grid_size"]),
                    world_extent=world_extent,
                    diffusion=float(mods_cfg["diffusion"]),
                    decay_tau=float(mods_cfg["decay_tau"]),
                    deposit_sigma=float(mods_cfg["deposit_sigma"]),
                )
            )
        else:
            field = GlobalScalarField(
                kinds=modulator_kinds,
                params=GlobalScalarParams(decay_tau=float(mods_cfg["decay_tau"])),
            )
        mod_specs.append(ModulatorSpec(name="logic_modulators", field=field, kinds=modulator_kinds))
        pending_releases = {kind: 0.0 for kind in modulator_kinds}
        modulator_state = {
            "field_type": field_type,
            "world_extent": world_extent,
            "output_positions": output_release_positions,
            "input_active": False,
            "trial_step_idx": -1,
            "trial_steps": 0,
            "ach_pulse_emitted": False,
            "dopamine_focus_action": None,
        }

        def releases_fn(t: float, step: int, ctx: StepContext):  # type: ignore[no-redef]
            _ = (t, step)
            torch_local = require_torch()
            dtype_local = _resolve_torch_dtype(torch_local, ctx.dtype, fallback=torch_local.float32)
            device_local = (
                torch_local.device(ctx.device) if ctx.device else torch_local.device("cpu")
            )
            releases, ach_emitted = _build_modulator_releases_for_step(
                pending_releases=pending_releases,
                modulator_kinds=modulator_kinds,
                modulator_amount=modulator_amount,
                input_active=bool(modulator_state.get("input_active", False)),
                ach_pulse_emitted=bool(modulator_state.get("ach_pulse_emitted", False)),
                field_type=str(modulator_state.get("field_type", "global_scalar")),
                world_extent=cast(
                    tuple[float, float], modulator_state.get("world_extent", (1.0, 1.0))
                ),
                output_positions=modulator_state.get("output_positions"),
                dopamine_focus_action=modulator_state.get("dopamine_focus_action"),
                device=device_local,
                dtype=dtype_local,
            )
            modulator_state["ach_pulse_emitted"] = ach_emitted
            return releases

    def external_drive_fn(t: float, step: int, pop_name: str, ctx: StepContext):
        _ = (t, step, ctx)
        merged_drive: dict[Compartment, Any] = {}
        if pop_name == handles.input_population:
            drive = current_input_drive.get("tensor")
            if drive is not None:
                merged_drive[Compartment.SOMA] = drive
        context_drive = context_drive_by_population.get(pop_name)
        if context_drive:
            for compartment, values in context_drive.items():
                existing = merged_drive.get(compartment)
                if existing is None:
                    merged_drive[compartment] = values
                else:
                    merged_drive[compartment] = existing + values
        action_drive = action_drive_by_population.get(pop_name)
        if action_drive:
            for compartment, values in action_drive.items():
                existing = merged_drive.get(compartment)
                if existing is None:
                    merged_drive[compartment] = values
                else:
                    merged_drive[compartment] = existing + values
        return merged_drive

    homeostasis_rule = None
    if homeostasis_cfg["enabled"]:
        homeostasis_rule = RateEmaThresholdHomeostasis(
            RateEmaThresholdHomeostasisConfig(
                alpha=float(homeostasis_cfg["alpha"]),
                eta=float(homeostasis_cfg["eta"]),
                r_target=float(homeostasis_cfg["r_target"]),
                clamp_min=float(homeostasis_cfg["clamp_min"]),
                clamp_max=float(homeostasis_cfg["clamp_max"]),
                scope=cast(HomeostasisScope, homeostasis_cfg["scope"]),
            )
        )

    engine = TorchNetworkEngine(
        populations=populations,
        projections=projections,
        modulators=tuple(mod_specs),
        homeostasis=homeostasis_rule,
        external_drive_fn=external_drive_fn,
        releases_fn=releases_fn,
        fast_mode=True,
        compiled_mode=True,
        learning_use_scratch=True,
    )
    engine.reset(
        config=SimulationConfig(
            dt=float(config.dt),
            seed=seed,
            device=device,
            dtype=str(run_spec.get("dtype", "float32")),
            max_ring_mib=float(run_spec.get("max_ring_mib", 2048.0)),
            enable_pruning=bool(pruning_cfg["enabled"]),
            prune_interval_steps=int(pruning_cfg["prune_interval_steps"]),
            usage_alpha=float(pruning_cfg["usage_alpha"]),
            w_min=float(pruning_cfg["w_min"]),
            usage_min=float(pruning_cfg["usage_min"]),
            k_min_out=int(pruning_cfg["k_min_out"]),
            k_min_in=int(pruning_cfg["k_min_in"]),
            max_prune_fraction_per_interval=float(pruning_cfg["max_prune_fraction_per_interval"]),
            pruning_verbose=bool(pruning_cfg["verbose"]),
            enable_neurogenesis=bool(neuro_cfg["enabled"]),
            growth_interval_steps=int(neuro_cfg["growth_interval_steps"]),
            add_neurons_per_event=int(neuro_cfg["add_neurons_per_event"]),
            newborn_plasticity_multiplier=float(neuro_cfg["newborn_plasticity_multiplier"]),
            newborn_duration_steps=int(neuro_cfg["newborn_duration_steps"]),
            max_total_neurons=int(neuro_cfg["max_total_neurons"]),
            neurogenesis_verbose=bool(neuro_cfg["verbose"]),
            excitability_modulation=ExcitabilityModulationConfig(
                enabled=bool(excitability_cfg["enabled"]),
                targets=tuple(excitability_cfg["targets"]),
                compartment=str(excitability_cfg["compartment"]),
                ach_gain=float(excitability_cfg["ach_gain"]),
                ne_gain=float(excitability_cfg["ne_gain"]),
                ht_gain=float(excitability_cfg["ht_gain"]),
                clamp_abs=float(excitability_cfg["clamp_abs"]),
            ),
        )
    )
    _validate_compiled_advanced_synapse_prerequisites(engine=engine)
    engine.set_training(learning_enabled)

    return _EngineBuild(
        engine=engine,
        input_population=handles.input_population,
        output_population=handles.output_population,
        hidden_populations=handles.hidden_populations,
        current_input_drive=current_input_drive,
        context_drive_by_population=context_drive_by_population,
        action_drive_by_population=action_drive_by_population,
        pending_releases=pending_releases,
        modulator_kinds=modulator_kinds,
        modulator_amount=modulator_amount,
        modulator_state=modulator_state,
        learning_proj_names=learning_proj_names,
        learning_proj_name=learning_proj_name,
    )


def _validate_advanced_synapse_prerequisites(*, projections: Sequence[ProjectionSpec]) -> None:
    for projection in projections:
        synapse = projection.synapse
        if not isinstance(synapse, DelayedSparseMatmulSynapse):
            continue
        params = synapse.params
        conductance_like = bool(params.conductance_mode or params.nmda_voltage_block)
        if conductance_like:
            meta_key = str(getattr(params, "post_voltage_meta_key", "post_membrane"))
            if meta_key != "post_membrane":
                raise ValueError(
                    "conductance_mode/nmda_voltage_block requires post-membrane voltage meta; "
                    "enable engine voltage meta plumbing "
                    "(expected DelayedSparseMatmulParams.post_voltage_meta_key='post_membrane')."
                )
        if bool(params.stp_enabled):
            requirements = synapse.compilation_requirements()
            if not bool(requirements.get("needs_pre_adjacency", False)):
                raise ValueError(
                    "STP-enabled DelayedSparseMatmulSynapse must set "
                    "compilation_requirements()['needs_pre_adjacency']=True."
                )


def _validate_compiled_advanced_synapse_prerequisites(*, engine: TorchNetworkEngine) -> None:
    for projection in engine._proj_specs:
        synapse = projection.synapse
        if not isinstance(synapse, DelayedSparseMatmulSynapse):
            continue
        if not bool(synapse.params.stp_enabled):
            continue
        meta = projection.topology.meta or {}
        if meta.get("pre_ptr") is None or meta.get("edge_idx") is None:
            raise ValueError(
                "STP-enabled DelayedSparseMatmulSynapse requires pre-adjacency topology meta "
                "(pre_ptr/edge_idx); ensure compilation_requirements requests "
                "needs_pre_adjacency=True and compile_topology honors it."
            )


def _make_learning_rule(learning_cfg: Mapping[str, Any]) -> ILearningRule | None:
    if not bool(learning_cfg.get("enabled", False)):
        return None
    rule_name = _normalize_learning_rule(str(learning_cfg.get("rule", "three_factor_elig_stdp")))
    lr = float(learning_cfg.get("lr", 0.1))
    modulator_kind = _mod_kind_from_token(str(learning_cfg.get("modulator_kind", "dopamine")))
    if modulator_kind is None:
        modulator_kind = ModulatorKind.DOPAMINE

    if rule_name in {"none", "surrogate"}:
        return None
    if rule_name in {"rstdp", "rstdp_eligibility", "rstdp_elig"}:
        w_min_raw = learning_cfg.get("w_min")
        w_max_raw = learning_cfg.get("w_max")
        w_min = _coerce_float(w_min_raw, 0.0) if w_min_raw is not None else 0.0
        w_max = _coerce_float(w_max_raw, 0.5) if w_max_raw is not None else 0.5
        if w_max < w_min:
            w_min, w_max = w_max, w_min
        return RStdpEligibilityRule(
            RStdpEligibilityParams(
                lr=lr,
                w_min=w_min,
                w_max=w_max,
            )
        )
    if rule_name in {"three_factor_eligibility_stdp", "three_factor_elig_stdp"}:
        return ThreeFactorEligibilityStdpRule(
            ThreeFactorEligibilityStdpParams(
                lr=lr,
                modulator_kind=modulator_kind,
            )
        )
    if rule_name == "three_factor_hebbian":
        return ThreeFactorHebbianRule(ThreeFactorHebbianParams(lr=lr))
    if rule_name == "eligibility_trace_hebbian":
        return EligibilityTraceHebbianRule(
            EligibilityTraceHebbianParams(lr=lr, enable_eligibility=True)
        )
    if rule_name == "metaplastic_projection":
        base = ThreeFactorHebbianRule(ThreeFactorHebbianParams(lr=lr))
        return MetaplasticProjectionRule(base, MetaplasticProjectionParams(enabled=True))
    raise ValueError(f"Unsupported learning rule for engine runner: {rule_name}")


def _maybe_wrap_learning_rule(
    learning_rule: ILearningRule | None,
    wrapper_cfg: Mapping[str, Any],
) -> ILearningRule | None:
    if learning_rule is None:
        return None
    if not bool(wrapper_cfg.get("enabled", False)):
        return learning_rule
    combine_mode = _coerce_choice(
        wrapper_cfg.get("combine_mode"),
        allowed={"exp", "linear"},
        default="exp",
    )
    missing_policy = _coerce_choice(
        wrapper_cfg.get("missing_modulators_policy"),
        allowed={"zero"},
        default="zero",
    )
    params = ModulatedRuleWrapperParams(
        ach_lr_gain=_coerce_float(wrapper_cfg.get("ach_lr_gain"), 0.0),
        ne_lr_gain=_coerce_float(wrapper_cfg.get("ne_lr_gain"), 0.0),
        ht_lr_gain=_coerce_float(wrapper_cfg.get("ht_lr_gain"), 0.0),
        ht_extra_weight_decay=_coerce_float(wrapper_cfg.get("ht_extra_weight_decay"), 0.0),
        lr_clip_min=_coerce_float(wrapper_cfg.get("lr_clip_min"), 0.1),
        lr_clip_max=_coerce_float(wrapper_cfg.get("lr_clip_max"), 10.0),
        dopamine_baseline=_coerce_float(wrapper_cfg.get("dopamine_baseline"), 0.0),
        ach_baseline=_coerce_float(wrapper_cfg.get("ach_baseline"), 0.0),
        ne_baseline=_coerce_float(wrapper_cfg.get("ne_baseline"), 0.0),
        ht_baseline=_coerce_float(wrapper_cfg.get("ht_baseline"), 0.0),
        combine_mode=cast(Any, combine_mode),
        missing_modulators_policy=cast(Any, missing_policy),
    )
    return ModulatedRuleWrapper(inner=learning_rule, params=params)


def _resolve_learning_targets(
    *,
    projections: Sequence[ProjectionSpec],
    targets: Sequence[str] | None,
    output_population: str,
) -> tuple[str, ...]:
    """Resolve learning targets using exact names or glob patterns.

    If targets is None, falls back to default behavior: select single primary projection.
    If targets is a list, each entry can be:
      - Exact projection name: "Hidden1Excit->Out"
      - Glob pattern: "*Excit->Out", "Hidden*->*"
    """
    by_name = {proj.name: proj for proj in projections}

    if not targets:
        # Fallback to single default projection (backward compatible)
        primary = _select_learning_projection_name(
            projections=projections,
            output_population=output_population,
        )
        return (primary,)

    matched = set()
    for target_pattern in targets:
        target_pattern = str(target_pattern).strip()
        # Try exact match first
        if target_pattern in by_name:
            matched.add(target_pattern)
        else:
            # Try glob pattern
            for proj_name in by_name:
                if fnmatch.fnmatch(proj_name, target_pattern):
                    matched.add(proj_name)

    if not matched:
        raise ValueError(
            f"No projections matched learning targets {targets}. Available: {list(by_name.keys())}"
        )

    return tuple(sorted(matched))


def _apply_learning_projection(
    *,
    base_projections: Sequence[ProjectionSpec],
    learning_rule: ILearningRule | None,
    learning_targets: tuple[str, ...],
    output_population: str,
    learn_every: int,
) -> tuple[tuple[ProjectionSpec, ...], tuple[str, ...], str | None]:
    """Apply learning rule to one or more projections.

    Returns:
        (updated_projections, learning_proj_names, primary_learning_proj_name)

    The primary_learning_proj_name is chosen for reporting:
      - Prefer projection whose post == output_population
      - Otherwise first in learning_proj_names
      - None if no learning rule or no targets
    """
    if learning_rule is None:
        return tuple(base_projections), (), None

    supports_sparse = bool(getattr(learning_rule, "supports_sparse", False))
    updated: list[ProjectionSpec] = []
    matched_names: list[str] = []

    for projection in base_projections:
        if projection.name in learning_targets:
            updated.append(
                ProjectionSpec(
                    name=projection.name,
                    synapse=projection.synapse,
                    topology=projection.topology,
                    pre=projection.pre,
                    post=projection.post,
                    learning=learning_rule,
                    learn_every=max(1, int(learn_every)),
                    sparse_learning=supports_sparse,
                    meta=projection.meta,
                )
            )
            matched_names.append(projection.name)
            continue
        updated.append(projection)

    # Select primary projection for reporting
    primary_name: str | None = None
    if matched_names:
        # Prefer output-targeting projection
        for name in matched_names:
            proj = next((p for p in base_projections if p.name == name), None)
            if proj is not None and proj.post == output_population:
                primary_name = name
                break
        # Otherwise use first
        if primary_name is None:
            primary_name = matched_names[0]

    return tuple(updated), tuple(matched_names), primary_name


def _select_learning_projection_name(
    *,
    projections: Sequence[ProjectionSpec],
    output_population: str,
) -> str:
    by_name = {projection.name: projection for projection in projections}
    preferred_names = (
        "HiddenExcit->Out",
        "Hidden1Excit->Out",
        "Hidden->Out",
    )
    for name in preferred_names:
        projection = by_name.get(name)
        if projection is not None and projection.post == output_population:
            return name
    for projection in projections:
        if projection.post == output_population and "inhib" not in projection.name.lower():
            return projection.name
    for projection in projections:
        if projection.post == output_population:
            return projection.name
    raise ValueError("No projection found that targets the output population for learning.")


def _output_positions_for_population(
    *,
    populations: Sequence[PopulationSpec],
    output_population: str,
    device: Any,
    dtype: Any,
) -> Any:
    torch = require_torch()
    for population in populations:
        if population.name != output_population:
            continue
        if population.positions is not None:
            return population.positions.detach().to(device=device, dtype=dtype)
        return torch.zeros((population.n, 3), device=device, dtype=dtype)
    return torch.zeros((1, 3), device=device, dtype=dtype)


def _curriculum_topology_gate(gates: Sequence[LogicGate]) -> LogicGate:
    if any(gate in {LogicGate.XOR, LogicGate.XNOR} for gate in gates):
        return LogicGate.XOR
    return gates[0]


def _resolve_learning_cfg(
    run_spec: Mapping[str, Any],
    *,
    default_enabled: bool = False,
    default_rule: str = "three_factor_elig_stdp",
    default_lr: float = 0.1,
    default_modulator_kind: str = "dopamine",
) -> dict[str, Any]:
    learning = _as_mapping(run_spec.get("learning"))
    modulator_kind = _coerce_choice(
        _first_non_none(
            learning.get("modulator_kind"),
            run_spec.get("learning_modulator_kind"),
            default_modulator_kind,
        ),
        allowed={
            "dopamine",
            "da",
            "acetylcholine",
            "ach",
            "noradrenaline",
            "na",
            "serotonin",
            "5ht",
        },
        default=default_modulator_kind,
    )
    if modulator_kind == "da":
        modulator_kind = "dopamine"
    elif modulator_kind == "ach":
        modulator_kind = "acetylcholine"
    elif modulator_kind == "na":
        modulator_kind = "noradrenaline"
    elif modulator_kind == "5ht":
        modulator_kind = "serotonin"
    return {
        "enabled": bool(_first_non_none(learning.get("enabled"), default_enabled)),
        "rule": _normalize_learning_rule(str(learning.get("rule", default_rule))),
        "lr": _coerce_float(_first_non_none(learning.get("lr"), default_lr), default_lr),
        "modulator_kind": modulator_kind,
        "w_min": _coerce_optional_float(learning.get("w_min")),
        "w_max": _coerce_optional_float(learning.get("w_max")),
        "targets": _coerce_string_list(learning.get("targets", ())),
    }


def _normalize_learning_rule(value: str) -> str:
    rule = str(value).strip().lower()
    if rule in {"three_factor_elig_stdp", "three_factor_eligibility_stdp"}:
        return "three_factor_elig_stdp"
    if rule in {"rstdp_elig", "rstdp_eligibility", "rstdp"}:
        return "rstdp_elig"
    if rule in {"none", "surrogate"}:
        return "none"
    return rule


def _resolve_modulators_cfg(
    run_spec: Mapping[str, Any],
    *,
    default_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    defaults = default_cfg or {}
    mods = _as_mapping(run_spec.get("modulators"))
    grid_size = _coerce_grid_size(
        _first_non_none(mods.get("grid_size"), defaults.get("grid_size")),
        default=_coerce_grid_size(defaults.get("grid_size"), default=(16, 16)),
    )
    world_extent = _coerce_float_pair(
        _first_non_none(mods.get("world_extent"), defaults.get("world_extent")),
        default=_coerce_float_pair(defaults.get("world_extent"), default=(1.0, 1.0)),
    )
    field_type = _coerce_choice(
        _first_non_none(mods.get("field_type"), defaults.get("field_type")),
        allowed={"global_scalar", "grid_diffusion_2d"},
        default=str(defaults.get("field_type", "global_scalar")),
    )
    return {
        "enabled": bool(_first_non_none(mods.get("enabled"), defaults.get("enabled", False))),
        "kinds": _coerce_string_list(_first_non_none(mods.get("kinds"), defaults.get("kinds", ()))),
        "pulse_step": _coerce_positive_int(
            _first_non_none(mods.get("pulse_step"), defaults.get("pulse_step"), 50),
            50,
        ),
        "amount": _coerce_float(
            _first_non_none(mods.get("amount"), defaults.get("amount"), 1.0), 1.0
        ),
        "field_type": field_type,
        "grid_size": grid_size,
        "world_extent": world_extent,
        "diffusion": max(
            0.0,
            _coerce_float(
                _first_non_none(mods.get("diffusion"), defaults.get("diffusion"), 0.0), 0.0
            ),
        ),
        "decay_tau": max(
            1e-9,
            _coerce_float(
                _first_non_none(mods.get("decay_tau"), defaults.get("decay_tau"), 1.0), 1.0
            ),
        ),
        "deposit_sigma": max(
            0.0,
            _coerce_float(
                _first_non_none(mods.get("deposit_sigma"), defaults.get("deposit_sigma"), 0.0),
                0.0,
            ),
        ),
    }


def _resolve_wrapper_cfg(
    run_spec: Mapping[str, Any],
    *,
    default_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    defaults = default_cfg or {}
    learning = _as_mapping(run_spec.get("learning"))
    wrapper = _as_mapping(_first_non_none(run_spec.get("wrapper"), learning.get("wrapper")))
    combine_mode = _coerce_choice(
        _first_non_none(wrapper.get("combine_mode"), defaults.get("combine_mode")),
        allowed={"exp", "linear"},
        default=str(defaults.get("combine_mode", "exp")),
    )
    lr_clip_min = _coerce_float(
        _first_non_none(wrapper.get("lr_clip_min"), defaults.get("lr_clip_min"), 0.1),
        0.1,
    )
    lr_clip_max = _coerce_float(
        _first_non_none(wrapper.get("lr_clip_max"), defaults.get("lr_clip_max"), 10.0),
        10.0,
    )
    if lr_clip_max < lr_clip_min:
        lr_clip_max = lr_clip_min
    return {
        "enabled": bool(_first_non_none(wrapper.get("enabled"), defaults.get("enabled", False))),
        "ach_lr_gain": _coerce_float(
            _first_non_none(wrapper.get("ach_lr_gain"), defaults.get("ach_lr_gain"), 0.0), 0.0
        ),
        "ne_lr_gain": _coerce_float(
            _first_non_none(wrapper.get("ne_lr_gain"), defaults.get("ne_lr_gain"), 0.0), 0.0
        ),
        "ht_lr_gain": _coerce_float(
            _first_non_none(wrapper.get("ht_lr_gain"), defaults.get("ht_lr_gain"), 0.0), 0.0
        ),
        "ht_extra_weight_decay": _coerce_float(
            _first_non_none(
                wrapper.get("ht_extra_weight_decay"),
                defaults.get("ht_extra_weight_decay"),
                0.0,
            ),
            0.0,
        ),
        "lr_clip_min": lr_clip_min,
        "lr_clip_max": lr_clip_max,
        "dopamine_baseline": _coerce_float(
            _first_non_none(
                wrapper.get("dopamine_baseline"), defaults.get("dopamine_baseline"), 0.0
            ),
            0.0,
        ),
        "ach_baseline": _coerce_float(
            _first_non_none(wrapper.get("ach_baseline"), defaults.get("ach_baseline"), 0.0), 0.0
        ),
        "ne_baseline": _coerce_float(
            _first_non_none(wrapper.get("ne_baseline"), defaults.get("ne_baseline"), 0.0), 0.0
        ),
        "ht_baseline": _coerce_float(
            _first_non_none(wrapper.get("ht_baseline"), defaults.get("ht_baseline"), 0.0), 0.0
        ),
        "combine_mode": combine_mode,
        "missing_modulators_policy": _coerce_choice(
            _first_non_none(
                wrapper.get("missing_modulators_policy"),
                defaults.get("missing_modulators_policy"),
                "zero",
            ),
            allowed={"zero"},
            default="zero",
        ),
    }


def _resolve_excitability_cfg(
    run_spec: Mapping[str, Any],
    *,
    default_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    defaults = default_cfg or {}
    exc = _as_mapping(run_spec.get("excitability_modulation"))
    targets_default = _coerce_string_list(defaults.get("targets", ("hidden", "out")))
    targets = _coerce_string_list(
        _first_non_none(exc.get("targets"), defaults.get("targets", targets_default))
    )
    if not targets:
        targets = targets_default or ["hidden", "out"]
    compartment = _coerce_choice(
        _first_non_none(exc.get("compartment"), defaults.get("compartment")),
        allowed={"soma", "dendrite", "ais", "axon"},
        default=str(defaults.get("compartment", "soma")),
    )
    return {
        "enabled": bool(_first_non_none(exc.get("enabled"), defaults.get("enabled", False))),
        "targets": targets,
        "compartment": compartment,
        "ach_gain": _coerce_float(
            _first_non_none(exc.get("ach_gain"), defaults.get("ach_gain"), 0.0), 0.0
        ),
        "ne_gain": _coerce_float(
            _first_non_none(exc.get("ne_gain"), defaults.get("ne_gain"), 0.0), 0.0
        ),
        "ht_gain": _coerce_float(
            _first_non_none(exc.get("ht_gain"), defaults.get("ht_gain"), 0.0), 0.0
        ),
        "clamp_abs": abs(
            _coerce_float(
                _first_non_none(exc.get("clamp_abs"), defaults.get("clamp_abs"), 1.0), 1.0
            )
        ),
    }


def _resolve_homeostasis_cfg(run_spec: Mapping[str, Any]) -> dict[str, Any]:
    homeo = _as_mapping(run_spec.get("homeostasis"))
    scope_raw = str(homeo.get("scope", "per_neuron")).strip().lower()
    scope = "per_population" if scope_raw == "per_population" else "per_neuron"
    return {
        "enabled": bool(homeo.get("enabled", False)),
        "alpha": _clamp(_coerce_float(homeo.get("alpha"), 0.01), 1e-6, 1.0),
        "eta": max(0.0, _coerce_float(homeo.get("eta"), 1e-3)),
        "r_target": _coerce_float(homeo.get("r_target"), 0.05),
        "clamp_min": _coerce_float(homeo.get("clamp_min"), 0.0),
        "clamp_max": _coerce_float(homeo.get("clamp_max"), 0.05),
        "scope": scope,
    }


def _resolve_pruning_cfg(run_spec: Mapping[str, Any]) -> dict[str, Any]:
    pruning = _as_mapping(run_spec.get("pruning"))
    return {
        "enabled": bool(pruning.get("enabled", False)),
        "prune_interval_steps": _coerce_positive_int(pruning.get("prune_interval_steps"), 250),
        "usage_alpha": max(0.0, _coerce_float(pruning.get("usage_alpha"), 0.01)),
        "w_min": max(0.0, _coerce_float(pruning.get("w_min"), 0.05)),
        "usage_min": max(0.0, _coerce_float(pruning.get("usage_min"), 0.01)),
        "k_min_out": _coerce_nonnegative_int(pruning.get("k_min_out"), 1),
        "k_min_in": _coerce_nonnegative_int(pruning.get("k_min_in"), 1),
        "max_prune_fraction_per_interval": _clamp(
            _coerce_float(pruning.get("max_prune_fraction_per_interval"), 0.10),
            0.0,
            1.0,
        ),
        "verbose": bool(pruning.get("verbose", False)),
    }


def _resolve_neurogenesis_cfg(run_spec: Mapping[str, Any]) -> dict[str, Any]:
    neuro = _as_mapping(run_spec.get("neurogenesis"))
    return {
        "enabled": bool(neuro.get("enabled", False)),
        "growth_interval_steps": _coerce_positive_int(neuro.get("growth_interval_steps"), 500),
        "add_neurons_per_event": _coerce_positive_int(neuro.get("add_neurons_per_event"), 4),
        "newborn_plasticity_multiplier": max(
            1e-6, _coerce_float(neuro.get("newborn_plasticity_multiplier"), 1.5)
        ),
        "newborn_duration_steps": _coerce_positive_int(neuro.get("newborn_duration_steps"), 250),
        "max_total_neurons": _coerce_positive_int(neuro.get("max_total_neurons"), 20000),
        "verbose": bool(neuro.get("verbose", False)),
    }


def _resolve_curriculum_gates(run_spec: Mapping[str, Any]) -> tuple[LogicGate, ...]:
    raw = str(run_spec.get("logic_curriculum_gates", "or,and,nor,nand,xor,xnor"))
    items = [part.strip().lower() for part in raw.split(",") if part.strip()]
    gates: list[LogicGate] = []
    for item in items:
        try:
            gates.append(coerce_gate(item))
        except Exception:
            continue
    if gates:
        return tuple(gates)
    return DEFAULT_CURRICULUM_GATES


def _write_engine_topology_json(*, engine: TorchNetworkEngine, run_dir: Path) -> None:
    try:
        weights_by_projection: dict[str, Any] = {}
        for proj in engine._proj_specs:
            state = engine._proj_states.get(proj.name)
            weights = getattr(state.state, "weights", None) if state is not None else None
            if weights is None:
                weights = proj.topology.weights
            if weights is not None:
                weights_by_projection[proj.name] = weights
        export_population_topology_json(
            engine._pop_specs,
            engine._proj_specs,
            path=run_dir / "topology.json",
            weights_by_projection=weights_by_projection,
            include_neuron_topology=True,
        )
    except Exception as exc:  # pragma: no cover
        print(f"[logic-gate-engine] warning: failed to export topology.json: {exc}")


def _coerce_modulator_kinds(values: Sequence[Any]) -> tuple[ModulatorKind, ...]:
    out: list[ModulatorKind] = []
    for raw in values:
        token = str(raw).strip().lower()
        kind = _mod_kind_from_token(token)
        if kind is None or kind in out:
            continue
        out.append(kind)
    return tuple(out)


def _mod_kind_from_token(token: str) -> ModulatorKind | None:
    if token in {"dopamine", "da"}:
        return ModulatorKind.DOPAMINE
    if token in {"acetylcholine", "ach"}:
        return ModulatorKind.ACETYLCHOLINE
    if token in {"noradrenaline", "norepinephrine", "na"}:
        return ModulatorKind.NORADRENALINE
    if token in {"serotonin", "5ht"}:
        return ModulatorKind.SEROTONIN
    return None


def _resolve_torch_dtype(torch: Any, raw: str | None, *, fallback: Any) -> Any:
    if raw is None:
        return fallback
    name = str(raw).split(".", 1)[-1]
    return getattr(torch, name, fallback)


def _build_curriculum_gate_indices(
    *,
    phase_index: int,
    phase_trials: int,
    replay_ratio: float,
    device: str = "cpu",
) -> Any:
    torch = require_torch()
    gate_idx = torch.full((phase_trials,), phase_index, device=device, dtype=torch.long)
    if phase_index <= 0 or replay_ratio <= 0.0:
        return gate_idx
    max_replay = phase_trials - 1 if phase_trials > 1 else 0
    replay_count = min(max_replay, int(round(phase_trials * replay_ratio)))
    if replay_count <= 0:
        return gate_idx
    stride = max(1, phase_trials // replay_count)
    replay_positions = torch.arange(0, phase_trials, stride, device=device, dtype=torch.long)[
        :replay_count
    ]
    replay_gates = torch.arange(replay_count, device=device, dtype=torch.long).remainder(
        phase_index
    )
    gate_idx.scatter_(0, replay_positions, replay_gates)
    return gate_idx


def _resolve_device(torch: Any, requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def _resolve_run_dir(config: LogicGateRunConfig, *, gate: LogicGate) -> Path:
    if config.out_dir is not None:
        out_dir = Path(config.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
    base = (
        Path(config.artifacts_root)
        if config.artifacts_root is not None
        else _default_artifacts_root()
    )
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"run_{stamp}_{gate.value}_engine"
    if run_dir.exists():
        run_dir = base / f"run_{stamp}_{int(time.time())}_{gate.value}_engine"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _resolve_curriculum_run_dir(config: LogicGateRunConfig, *, gates: Sequence[LogicGate]) -> Path:
    if config.out_dir is not None:
        out_dir = Path(config.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
    base = (
        Path(config.artifacts_root)
        if config.artifacts_root is not None
        else _default_artifacts_root()
    )
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gate_suffix = "_".join(gate.value for gate in gates[:3])
    run_dir = base / f"run_{stamp}_curriculum_{gate_suffix}_engine"
    if run_dir.exists():
        run_dir = base / f"run_{stamp}_{int(time.time())}_curriculum_engine"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _default_artifacts_root() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "artifacts" / "logic_gates"


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value)
    return {}


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _coerce_choice(value: Any, *, allowed: set[str], default: str) -> str:
    token = str(value).strip().lower() if value is not None else default
    return token if token in allowed else default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        out = int(value)
    except Exception:
        out = int(default)
    return max(1, out)


def _coerce_nonnegative_int(value: Any, default: int) -> int:
    try:
        out = int(value)
    except Exception:
        out = int(default)
    return max(0, out)


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [token.strip() for token in value.split(",") if token.strip()]
    if isinstance(value, Sequence) and not isinstance(value, Mapping):
        out: list[str] = []
        for item in value:
            token = str(item).strip()
            if token:
                out.append(token)
        return out
    return []


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return None


def _coerce_grid_size(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, str):
        token = value.strip().lower().replace(" ", "")
        if "x" in token:
            left, right = token.split("x", 1)
            try:
                h = max(1, int(left))
                w = max(1, int(right))
                return h, w
            except Exception:
                return default
    if isinstance(value, Sequence) and len(value) == 2:
        try:
            return max(1, int(value[0])), max(1, int(value[1]))
        except Exception:
            return default
    return default


def _coerce_float_pair(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, str):
        token = value.strip().replace(" ", "")
        if "," in token:
            left, right = token.split(",", 1)
            try:
                return float(left), float(right)
            except Exception:
                return default
    if isinstance(value, Sequence) and len(value) == 2:
        try:
            return float(value[0]), float(value[1])
        except Exception:
            return default
    return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


__all__ = ["run_logic_gate_engine", "run_logic_gate_curriculum_engine"]
