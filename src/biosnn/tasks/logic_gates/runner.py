"""Deterministic logic-gate harness runner."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from biosnn.contracts.learning import LearningBatch
from biosnn.contracts.modulators import ModulatorKind
from biosnn.contracts.neurons import Compartment, StepContext
from biosnn.core.torch_utils import require_torch
from biosnn.io.sinks import CsvSink
from biosnn.learning.rules import RStdpEligibilityParams, RStdpEligibilityRule

from .configs import LogicGateRunConfig
from .datasets import LogicGate, coerce_gate, make_truth_table, sample_case_indices
from .encoding import INPUT_NEURON_INDICES, OUTPUT_NEURON_INDICES, decode_output, encode_inputs
from .evaluators import PassTracker, eval_accuracy
from .surrogate_train import train_logic_gate_surrogate

DEFAULT_CURRICULUM_GATES: tuple[LogicGate, ...] = (
    LogicGate.OR,
    LogicGate.AND,
    LogicGate.NOR,
    LogicGate.NAND,
    LogicGate.XOR,
    LogicGate.XNOR,
)


@dataclass(slots=True)
class _ReferenceLogicNetwork:
    w_in: Any
    b_in: Any
    w_out: Any | None = None
    b_out: Any | None = None

    def predict(self, inputs: Any) -> Any:
        logits = inputs.matmul(self.w_in) + self.b_in
        if self.w_out is not None:
            hidden = (logits >= 0.0).to(dtype=inputs.dtype)
            logits = hidden.matmul(self.w_out) + self.b_out
        return (logits >= 0.0).to(dtype=inputs.dtype).reshape(-1)

    def hidden_size(self, *, input_size: int) -> int:
        if self.w_out is not None:
            return int(self.w_in.shape[1])
        return int(input_size)

    def fill_hidden_activity(
        self,
        inputs_case: Any,
        *,
        out: Any,
        input_proxy: Any | None = None,
    ) -> None:
        if self.w_out is not None:
            logits = inputs_case.matmul(self.w_in) + self.b_in
            out.copy_((logits >= 0.0).to(dtype=out.dtype))
            return
        if input_proxy is None:
            out.zero_()
            return
        out.copy_(input_proxy)
        out.clamp_(0.0, 1.0)


@dataclass(slots=True)
class _RstdpRuntime:
    gate: LogicGate
    topology_name: str
    mode: str
    hidden_size: int
    rule: RStdpEligibilityRule
    state: Any
    ctx: StepContext
    pre_idx: Any
    post_idx: Any
    weights: Any
    pre_spikes: Any
    edge_pre: Any
    edge_post: Any
    edge_current: Any
    output_drive: Any
    output_step_spikes: Any
    output_spike_counts: Any
    dopamine_edge: Any
    batch_extras: dict[str, Any]
    dopamine_modulators: dict[ModulatorKind, Any]
    dopamine_window_steps: int
    pred_edge_pre: Any
    pred_input_aug: Any
    pred_edge_current: Any
    pred_out_drive: Any
    pred_post_idx_2d: Any
    xor_input_spikes: Any | None
    h0_pre_idx: Any | None
    h0_post_idx: Any | None
    h0_weights: Any | None
    h0_bias: Any | None
    h0_edge_pre: Any | None
    h0_edge_current: Any | None
    h0_drive: Any | None
    h0_spikes: Any | None
    h1_pre_idx: Any | None
    h1_post_idx: Any | None
    h1_weights: Any | None
    h1_bias: Any | None
    h1_edge_pre: Any | None
    h1_edge_current: Any | None
    h1_drive: Any | None
    h1_spikes: Any | None
    xor_pred_input: Any | None
    xor_pred_h0_edge_pre: Any | None
    xor_pred_h0_edge_current: Any | None
    xor_pred_h0_drive: Any | None
    xor_pred_h0_post_idx_2d: Any | None
    xor_pred_h0_spikes: Any | None
    xor_pred_h1_edge_pre: Any | None
    xor_pred_h1_edge_current: Any | None
    xor_pred_h1_drive: Any | None
    xor_pred_h1_post_idx_2d: Any | None
    xor_pred_h1_spikes: Any | None


def run_logic_gate(cfg: LogicGateRunConfig) -> dict[str, Any]:
    """Run deterministic logic-gate trials and export CSV artifacts."""

    torch = require_torch()
    gate = coerce_gate(cfg.gate)
    device = _resolve_device(torch, cfg.device)
    run_dir = _resolve_run_dir(cfg, gate=gate)

    torch.manual_seed(int(cfg.seed))
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))

    sampling_generator = torch.Generator(device="cpu")
    sampling_generator.manual_seed(int(cfg.seed))

    inputs, targets = make_truth_table(gate, device=device, dtype="float32")
    targets_flat = targets.reshape(4)
    encoded_inputs = _build_encoded_input_table(inputs, dt=cfg.dt)
    topology_name = _topology_name_for_gate(gate)
    if cfg.learning_mode == "surrogate":
        return _run_logic_gate_surrogate(
            cfg=cfg,
            gate=gate,
            device=device,
            run_dir=run_dir,
            topology_name=topology_name,
            inputs=inputs,
            targets=targets,
            encoded_inputs=encoded_inputs,
        )

    rstdp_runtime: _RstdpRuntime | None = None
    network: _ReferenceLogicNetwork | None = None

    if cfg.learning_mode == "rstdp":
        rstdp_runtime = _init_rstdp_runtime(
            gate=gate,
            device=device,
            dtype=inputs.dtype,
            seed=cfg.seed,
            n_cases=int(inputs.shape[0]),
            n_inputs=int(encoded_inputs.shape[1]),
            topology_name=topology_name,
        )
        case_scores = torch.zeros((4,), device=inputs.device, dtype=inputs.dtype)
        predictions = torch.zeros((4,), device=inputs.device, dtype=inputs.dtype)
        _predict_all_cases_rstdp(
            runtime=rstdp_runtime,
            encoded_inputs=encoded_inputs,
            predictions_out=predictions,
            scores_out=case_scores,
        )
    else:
        network = _build_reference_network(gate=gate, device=device, dtype=inputs.dtype)
        case_scores = network.predict(inputs).to(dtype=inputs.dtype).clone()
        predictions = (case_scores >= 0.5).to(dtype=inputs.dtype)

    case_indices = sample_case_indices(
        cfg.steps,
        method=cfg.sampling_method,
        generator=sampling_generator,
        device="cpu",
    )
    case_sequence = [int(value) for value in case_indices.tolist()]

    trial_sink = CsvSink(run_dir / "trials.csv", flush_every=max(1, cfg.export_every))
    eval_sink = CsvSink(run_dir / "eval.csv", flush_every=1)
    confusion_sink = CsvSink(run_dir / "confusion.csv", flush_every=1)
    last_trials: deque[dict[str, Any]] | None = (
        deque(maxlen=cfg.dump_last_trials_n) if cfg.dump_last_trials_csv else None
    )

    tracker = PassTracker(gate)
    first_pass_trial: int | None = None
    sampled_correct = 0
    t0 = perf_counter()
    last_pred: int | None = None

    output_spike_counts = torch.zeros((2,), device=inputs.device, dtype=inputs.dtype)
    output_step_spikes = torch.zeros((2,), device=inputs.device, dtype=inputs.dtype)
    input_drive_buf = torch.zeros((4,), device=inputs.device, dtype=inputs.dtype)
    if rstdp_runtime is not None:
        hidden_size = int(rstdp_runtime.hidden_size)
    else:
        if network is None:
            raise RuntimeError("Reference logic network was not initialized.")
        hidden_size = network.hidden_size(input_size=4)
    hidden_step_spikes = torch.zeros((hidden_size,), device=inputs.device, dtype=inputs.dtype)
    hidden_spike_counts = torch.zeros_like(hidden_step_spikes)

    debug_every = max(1, int(cfg.debug_every))
    last_trials_csv: Path | None = None
    rolling_correct: deque[int] = deque(maxlen=100)

    last_trial_acc_rolling = 0.0
    last_mean_eligibility = 0.0
    last_mean_abs_dw = 0.0
    last_weight_min = float("nan")
    last_weight_max = float("nan")
    last_weight_mean = float("nan")

    print(
        f"[logic-gate] run_dir={run_dir} gate={gate.name} "
        f"trials={cfg.steps} mode={cfg.learning_mode} device={device} "
        f"topology={topology_name}"
    )

    initial_eval_acc, initial_confusion = eval_accuracy(
        predictions,
        targets_flat,
        report_confusion=True,
    )
    if rstdp_runtime is not None:
        init_w_min, init_w_max, init_w_mean = _tensor_stats(rstdp_runtime.weights)
        init_elig = float(rstdp_runtime.state.eligibility.abs().mean().item())
    else:
        init_w_min, init_w_max, init_w_mean = float("nan"), float("nan"), float("nan")
        init_elig = 0.0
    eval_sink.write_row(
        {
            "trial": 0,
            "eval_accuracy": float(initial_eval_acc),
            "sample_accuracy": 0.0,
            "trial_acc_rolling": 0.0,
            "mean_eligibility_abs": init_elig,
            "mean_abs_dw": 0.0,
            "weights_min": init_w_min,
            "weights_max": init_w_max,
            "weights_mean": init_w_mean,
            "perfect_streak": 0,
            "high_streak": 0,
            "passed": 0,
        }
    )
    confusion_sink.write_row({"trial": 0, **initial_confusion})
    print(f"[logic-gate] trial=0/{cfg.steps} eval_acc={float(initial_eval_acc):.3f} (pre-training)")

    try:
        for trial_idx, case_idx in enumerate(case_sequence, start=1):
            target_value = targets_flat[case_idx]
            input_drive_buf.copy_(encoded_inputs[case_idx])
            output_spike_counts.zero_()
            hidden_spike_counts.zero_()
            hidden_step_spikes.zero_()

            dopamine_pulse = 0.0
            mean_abs_dw_trial = 0.0

            if rstdp_runtime is not None:
                pred_bit, mean_abs_dw_online = _run_rstdp_trial_forward(
                    runtime=rstdp_runtime,
                    input_drive=input_drive_buf,
                    sim_steps_per_trial=cfg.sim_steps_per_trial,
                    dt=cfg.dt,
                    hidden_step_spikes=hidden_step_spikes,
                    hidden_spike_counts=hidden_spike_counts,
                    last_pred=last_pred,
                )
                output_spike_counts.copy_(rstdp_runtime.output_spike_counts)
            else:
                if network is not None:
                    network.fill_hidden_activity(
                        inputs[case_idx],
                        out=hidden_step_spikes,
                        input_proxy=input_drive_buf,
                    )
                for _ in range(cfg.sim_steps_per_trial):
                    _apply_learning_step(
                        predictions=case_scores,
                        case_idx=case_idx,
                        target=target_value,
                        mode=cfg.learning_mode,
                    )
                    output_step_spikes.zero_()
                    pred_step = int((case_scores[case_idx] >= 0.5).item())
                    output_step_spikes[pred_step] = 1.0
                    output_spike_counts.add_(output_step_spikes)
                    hidden_spike_counts.add_(hidden_step_spikes)
                pred_bit = decode_output(
                    output_spike_counts,
                    mode="wta",
                    hysteresis=0.0,
                    last_pred=last_pred,
                )
                predictions.copy_((case_scores >= 0.5).to(dtype=predictions.dtype))
                mean_abs_dw_online = 0.0

            target_bit = int((target_value >= 0.5).item())
            correct = int(pred_bit == target_bit)
            sampled_correct += correct
            rolling_correct.append(correct)
            trial_acc_rolling = sum(rolling_correct) / float(len(rolling_correct))

            if rstdp_runtime is not None:
                reward_signal = _reward_signal(correct=bool(correct), pred_bit=pred_bit, last_pred=last_pred)
                mean_abs_dw_dopa, dopamine_pulse = _apply_rstdp_dopamine(
                    runtime=rstdp_runtime,
                    reward_signal=reward_signal,
                    pred_bit=pred_bit,
                    target_bit=target_bit,
                    dt=cfg.dt,
                )
                mean_abs_dw_trial = _mean_nonzero(mean_abs_dw_online, mean_abs_dw_dopa)
                _predict_all_cases_rstdp(
                    runtime=rstdp_runtime,
                    encoded_inputs=encoded_inputs,
                    predictions_out=predictions,
                    scores_out=case_scores,
                )
                last_mean_eligibility = float(rstdp_runtime.state.eligibility.abs().mean().item())
                last_weight_min, last_weight_max, last_weight_mean = _tensor_stats(rstdp_runtime.weights)
            else:
                predictions.copy_((case_scores >= 0.5).to(dtype=predictions.dtype))
                dopamine_pulse = _dopamine_pulse(mode=cfg.learning_mode, correct=bool(correct))
                mean_abs_dw_trial = abs(dopamine_pulse) * 0.01
                last_mean_eligibility = 0.0
                last_weight_min = float("nan")
                last_weight_max = float("nan")
                last_weight_mean = float("nan")

            hidden_mean_spikes = float(hidden_spike_counts.mean().item()) / float(cfg.sim_steps_per_trial)
            tie_behavior = int(output_spike_counts[0].item() == output_spike_counts[1].item())
            no_spikes = int(float(output_spike_counts.sum().item()) <= 0.0)

            trial_row = {
                "trial": trial_idx,
                "sim_step_end": trial_idx * cfg.sim_steps_per_trial,
                "case_idx": case_idx,
                "x0": float(inputs[case_idx, 0].item()),
                "x1": float(inputs[case_idx, 1].item()),
                "in_bit0_0_drive": float(input_drive_buf[INPUT_NEURON_INDICES["bit0_0"]].item()),
                "in_bit0_1_drive": float(input_drive_buf[INPUT_NEURON_INDICES["bit0_1"]].item()),
                "in_bit1_0_drive": float(input_drive_buf[INPUT_NEURON_INDICES["bit1_0"]].item()),
                "in_bit1_1_drive": float(input_drive_buf[INPUT_NEURON_INDICES["bit1_1"]].item()),
                "out_spikes_0": float(output_spike_counts[OUTPUT_NEURON_INDICES["class_0"]].item()),
                "out_spikes_1": float(output_spike_counts[OUTPUT_NEURON_INDICES["class_1"]].item()),
                "hidden_mean_spikes": hidden_mean_spikes,
                "dopamine_pulse": dopamine_pulse,
                "trial_acc_rolling": trial_acc_rolling,
                "mean_eligibility_abs": last_mean_eligibility,
                "mean_abs_dw": mean_abs_dw_trial,
                "weights_min": last_weight_min,
                "weights_max": last_weight_max,
                "weights_mean": last_weight_mean,
                "tie_wta": tie_behavior,
                "no_output_spikes": no_spikes,
                "target": target_bit,
                "pred": pred_bit,
                "correct": correct,
            }
            trial_sink.write_row(trial_row)
            if last_trials is not None:
                last_trials.append(dict(trial_row))

            if cfg.debug and (trial_idx == 1 or trial_idx == cfg.steps or trial_idx % debug_every == 0):
                hidden_top_k = _top_k_active_neurons(hidden_spike_counts, k=cfg.debug_top_k)
                hidden_top_k_str = _format_top_k(hidden_top_k)
                print(
                    f"[logic-gate][debug] trial={trial_idx}/{cfg.steps} "
                    f"input=({int(inputs[case_idx, 0].item())},{int(inputs[case_idx, 1].item())}) "
                    f"target={target_bit} pred={pred_bit} "
                    f"out=[{output_spike_counts[0].item():.1f},{output_spike_counts[1].item():.1f}] "
                    f"hidden_mean={hidden_mean_spikes:.3f} hidden_topk={hidden_top_k_str} "
                    f"dopamine={dopamine_pulse:+.2f} tie={tie_behavior} no_spikes={no_spikes}"
                )

            eval_acc = float(eval_accuracy(predictions, targets_flat))
            tracker.update(eval_acc)
            if tracker.passed and first_pass_trial is None:
                first_pass_trial = trial_idx

            if (trial_idx % cfg.export_every) == 0 or trial_idx == cfg.steps:
                sample_acc = sampled_correct / float(trial_idx)
                eval_acc_full, confusion = eval_accuracy(predictions, targets_flat, report_confusion=True)
                eval_sink.write_row(
                    {
                        "trial": trial_idx,
                        "eval_accuracy": eval_acc_full,
                        "sample_accuracy": sample_acc,
                        "trial_acc_rolling": trial_acc_rolling,
                        "mean_eligibility_abs": last_mean_eligibility,
                        "mean_abs_dw": mean_abs_dw_trial,
                        "weights_min": last_weight_min,
                        "weights_max": last_weight_max,
                        "weights_mean": last_weight_mean,
                        "perfect_streak": tracker.perfect_streak,
                        "high_streak": tracker.high_streak,
                        "passed": int(tracker.passed),
                    }
                )
                confusion_sink.write_row({"trial": trial_idx, **confusion})
                print(
                    f"[logic-gate] trial={trial_idx}/{cfg.steps} "
                    f"eval_acc={eval_acc_full:.3f} sample_acc={sample_acc:.3f} "
                    f"roll_acc={trial_acc_rolling:.3f} pass={tracker.passed}"
                )
                if cfg.debug:
                    print(f"[logic-gate] confusion={confusion}")

            last_pred = pred_bit
            last_trial_acc_rolling = trial_acc_rolling
            last_mean_abs_dw = mean_abs_dw_trial
    finally:
        trial_sink.close()
        eval_sink.close()
        confusion_sink.close()

    if last_trials is not None and len(last_trials) > 0:
        last_trials_csv = run_dir / f"trials_last_{cfg.dump_last_trials_n}.csv"
        last_sink = CsvSink(last_trials_csv, flush_every=1)
        try:
            for row in last_trials:
                last_sink.write_row(row)
        finally:
            last_sink.close()

    elapsed_s = perf_counter() - t0
    final_eval_acc = float(eval_accuracy(predictions, targets_flat))
    sample_accuracy = sampled_correct / float(cfg.steps)

    return {
        "out_dir": run_dir,
        "gate": gate.value,
        "device": device,
        "learning_mode": cfg.learning_mode,
        "topology": topology_name,
        "steps": cfg.steps,
        "sim_steps_per_trial": cfg.sim_steps_per_trial,
        "inputs": inputs,
        "targets": targets,
        "preds": predictions,
        "case_scores": case_scores,
        "sample_indices": case_indices,
        "eval_accuracy": final_eval_acc,
        "sample_accuracy": sample_accuracy,
        "trial_acc_rolling": last_trial_acc_rolling,
        "mean_eligibility_abs": last_mean_eligibility,
        "mean_abs_dw": last_mean_abs_dw,
        "weights_min": last_weight_min,
        "weights_max": last_weight_max,
        "weights_mean": last_weight_mean,
        "passed": tracker.passed,
        "first_pass_trial": first_pass_trial,
        "elapsed_s": elapsed_s,
        "input_neuron_indices": dict(INPUT_NEURON_INDICES),
        "output_neuron_indices": dict(OUTPUT_NEURON_INDICES),
        "debug_every": cfg.debug_every,
        "last_trials_csv": last_trials_csv,
        "trials_csv": run_dir / "trials.csv",
        "eval_csv": run_dir / "eval.csv",
        "confusion_csv": run_dir / "confusion.csv",
    }


def run_logic_gate_curriculum(
    cfg: LogicGateRunConfig,
    *,
    gates: Sequence[LogicGate | str] | None = None,
    phase_steps: int | None = None,
    replay_ratio: float = 0.35,
) -> dict[str, Any]:
    """Run multiple logic gates sequentially without resetting R-STDP state."""

    if cfg.learning_mode != "rstdp":
        raise ValueError("Curriculum mode requires learning_mode='rstdp'.")

    torch = require_torch()
    gate_sequence = tuple(coerce_gate(gate) for gate in (gates or DEFAULT_CURRICULUM_GATES))
    if not gate_sequence:
        raise ValueError("At least one gate is required for curriculum mode.")
    replay_ratio = float(replay_ratio)
    if replay_ratio < 0.0 or replay_ratio > 1.0:
        raise ValueError("replay_ratio must be in [0.0, 1.0].")

    device = _resolve_device(torch, cfg.device)
    run_dir = _resolve_curriculum_run_dir(cfg, gates=gate_sequence)
    phase_trials = int(cfg.steps if phase_steps is None else phase_steps)
    if phase_trials <= 0:
        raise ValueError("phase_steps must be > 0.")

    torch.manual_seed(int(cfg.seed))
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))
    sampling_generator = torch.Generator(device="cpu")
    sampling_generator.manual_seed(int(cfg.seed))

    inputs, _ = make_truth_table(LogicGate.AND, device=device, dtype="float32")
    encoded_inputs = _build_encoded_input_table(inputs, dt=cfg.dt)
    targets_by_gate: dict[LogicGate, Any] = {}
    for gate in gate_sequence:
        _, gate_targets = make_truth_table(gate, device=device, dtype=inputs.dtype)
        targets_by_gate[gate] = gate_targets.reshape(4)
    runtime = _init_rstdp_runtime(
        gate=LogicGate.XOR,
        device=device,
        dtype=inputs.dtype,
        seed=cfg.seed,
        n_cases=int(inputs.shape[0]),
        n_inputs=int(encoded_inputs.shape[1]),
        topology_name="xor_ff2",
    )
    case_scores = torch.zeros((4,), device=inputs.device, dtype=inputs.dtype)
    predictions = torch.zeros((4,), device=inputs.device, dtype=inputs.dtype)
    _predict_all_cases_rstdp(
        runtime=runtime,
        encoded_inputs=encoded_inputs,
        predictions_out=predictions,
        scores_out=case_scores,
    )

    trial_sink = CsvSink(run_dir / "trials.csv", flush_every=max(1, cfg.export_every))
    eval_sink = CsvSink(run_dir / "eval.csv", flush_every=1)
    confusion_sink = CsvSink(run_dir / "confusion.csv", flush_every=1)
    phase_sink = CsvSink(run_dir / "phase_summary.csv", flush_every=1)
    last_trials: deque[dict[str, Any]] | None = (
        deque(maxlen=cfg.dump_last_trials_n) if cfg.dump_last_trials_csv else None
    )

    output_spike_counts = torch.zeros((2,), device=inputs.device, dtype=inputs.dtype)
    input_drive_buf = torch.zeros((4,), device=inputs.device, dtype=inputs.dtype)
    hidden_step_spikes = torch.zeros((int(runtime.hidden_size),), device=inputs.device, dtype=inputs.dtype)
    hidden_spike_counts = torch.zeros_like(hidden_step_spikes)
    debug_every = max(1, int(cfg.debug_every))
    last_trials_csv: Path | None = None
    last_pred: int | None = None
    rolling_correct: deque[int] = deque(maxlen=100)
    sampled_correct_total = 0
    global_trial = 0
    t0 = perf_counter()
    phase_results: list[dict[str, Any]] = []
    final_gate = gate_sequence[-1]

    print(
        f"[logic-curriculum] run_dir={run_dir} gates={[gate.value for gate in gate_sequence]} "
        f"phase_trials={phase_trials} replay_ratio={replay_ratio:.2f} "
        f"device={device} mode={cfg.learning_mode} topology=xor_ff2"
    )

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
                method=cfg.sampling_method,
                generator=sampling_generator,
                device="cpu",
            )
            case_sequence = [int(value) for value in case_indices.tolist()]

            print(f"[logic-curriculum] phase={phase_idx}/{len(gate_sequence)} gate={gate.value}")

            for local_trial, case_idx in enumerate(case_sequence, start=1):
                global_trial += 1
                active_gate = gate_sequence[int(train_gate_idx[local_trial - 1].item())]
                train_targets_flat = targets_by_gate[active_gate]
                target_value = train_targets_flat[case_idx]
                input_drive_buf.copy_(encoded_inputs[case_idx])
                output_spike_counts.zero_()
                hidden_spike_counts.zero_()
                hidden_step_spikes.zero_()

                pred_bit, mean_abs_dw_online = _run_rstdp_trial_forward(
                    runtime=runtime,
                    input_drive=input_drive_buf,
                    sim_steps_per_trial=cfg.sim_steps_per_trial,
                    dt=cfg.dt,
                    hidden_step_spikes=hidden_step_spikes,
                    hidden_spike_counts=hidden_spike_counts,
                    last_pred=last_pred,
                )
                output_spike_counts.copy_(runtime.output_spike_counts)

                target_bit = int((target_value >= 0.5).item())
                correct = int(pred_bit == target_bit)
                phase_correct += correct
                if active_gate == gate:
                    phase_gate_correct += correct
                    phase_gate_sample_count += 1
                sampled_correct_total += correct
                rolling_correct.append(correct)
                trial_acc_rolling = sum(rolling_correct) / float(len(rolling_correct))

                reward_signal = _reward_signal(correct=bool(correct), pred_bit=pred_bit, last_pred=last_pred)
                mean_abs_dw_dopa, dopamine_pulse = _apply_rstdp_dopamine(
                    runtime=runtime,
                    reward_signal=reward_signal,
                    pred_bit=pred_bit,
                    target_bit=target_bit,
                    dt=cfg.dt,
                )
                mean_abs_dw_trial = _mean_nonzero(mean_abs_dw_online, mean_abs_dw_dopa)
                _predict_all_cases_rstdp(
                    runtime=runtime,
                    encoded_inputs=encoded_inputs,
                    predictions_out=predictions,
                    scores_out=case_scores,
                )

                mean_eligibility = float(runtime.state.eligibility.abs().mean().item())
                weights_min, weights_max, weights_mean = _tensor_stats(runtime.weights)
                hidden_mean_spikes = float(hidden_spike_counts.mean().item()) / float(cfg.sim_steps_per_trial)
                tie_behavior = int(output_spike_counts[0].item() == output_spike_counts[1].item())
                no_spikes = int(float(output_spike_counts.sum().item()) <= 0.0)

                trial_row = {
                    "phase": phase_idx,
                    "gate": gate.value,
                    "train_gate": active_gate.value,
                    "phase_trial": local_trial,
                    "trial": global_trial,
                    "sim_step_end": global_trial * cfg.sim_steps_per_trial,
                    "case_idx": case_idx,
                    "x0": float(inputs[case_idx, 0].item()),
                    "x1": float(inputs[case_idx, 1].item()),
                    "in_bit0_0_drive": float(input_drive_buf[INPUT_NEURON_INDICES["bit0_0"]].item()),
                    "in_bit0_1_drive": float(input_drive_buf[INPUT_NEURON_INDICES["bit0_1"]].item()),
                    "in_bit1_0_drive": float(input_drive_buf[INPUT_NEURON_INDICES["bit1_0"]].item()),
                    "in_bit1_1_drive": float(input_drive_buf[INPUT_NEURON_INDICES["bit1_1"]].item()),
                    "out_spikes_0": float(output_spike_counts[OUTPUT_NEURON_INDICES["class_0"]].item()),
                    "out_spikes_1": float(output_spike_counts[OUTPUT_NEURON_INDICES["class_1"]].item()),
                    "hidden_mean_spikes": hidden_mean_spikes,
                    "dopamine_pulse": dopamine_pulse,
                    "trial_acc_rolling": trial_acc_rolling,
                    "mean_eligibility_abs": mean_eligibility,
                    "mean_abs_dw": mean_abs_dw_trial,
                    "weights_min": weights_min,
                    "weights_max": weights_max,
                    "weights_mean": weights_mean,
                    "tie_wta": tie_behavior,
                    "no_output_spikes": no_spikes,
                    "target": target_bit,
                    "pred": pred_bit,
                    "correct": correct,
                }
                trial_sink.write_row(trial_row)
                if last_trials is not None:
                    last_trials.append(dict(trial_row))

                if cfg.debug and (
                    local_trial == 1 or local_trial == phase_trials or local_trial % debug_every == 0
                ):
                    hidden_top_k = _top_k_active_neurons(hidden_spike_counts, k=cfg.debug_top_k)
                    hidden_top_k_str = _format_top_k(hidden_top_k)
                    print(
                        f"[logic-curriculum][debug] phase={phase_idx} gate={gate.value} "
                        f"trial={local_trial}/{phase_trials} "
                        f"train_gate={active_gate.value} "
                        f"input=({int(inputs[case_idx, 0].item())},{int(inputs[case_idx, 1].item())}) "
                        f"target={target_bit} pred={pred_bit} "
                        f"out=[{output_spike_counts[0].item():.1f},{output_spike_counts[1].item():.1f}] "
                        f"hidden_mean={hidden_mean_spikes:.3f} hidden_topk={hidden_top_k_str} "
                        f"dopamine={dopamine_pulse:+.2f} tie={tie_behavior} no_spikes={no_spikes}"
                    )

                phase_eval_last = float(eval_accuracy(predictions, targets_flat))
                tracker.update(phase_eval_last)
                if tracker.passed and first_pass_local is None:
                    first_pass_local = local_trial

                if (local_trial % cfg.export_every) == 0 or local_trial == phase_trials:
                    sample_acc_phase = phase_correct / float(local_trial)
                    sample_acc_phase_gate = (
                        phase_gate_correct / float(phase_gate_sample_count)
                        if phase_gate_sample_count > 0
                        else 0.0
                    )
                    sample_acc_global = sampled_correct_total / float(global_trial)
                    eval_acc_full, confusion = eval_accuracy(
                        predictions, targets_flat, report_confusion=True
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
                            "trial_acc_rolling": trial_acc_rolling,
                            "mean_eligibility_abs": mean_eligibility,
                            "mean_abs_dw": mean_abs_dw_trial,
                            "weights_min": weights_min,
                            "weights_max": weights_max,
                            "weights_mean": weights_mean,
                            "perfect_streak": tracker.perfect_streak,
                            "high_streak": tracker.high_streak,
                            "passed": int(tracker.passed),
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
                    print(
                        f"[logic-curriculum] phase={phase_idx} gate={gate.value} "
                        f"trial={local_trial}/{phase_trials} global={global_trial} "
                        f"eval_acc={eval_acc_full:.3f} sample_acc={sample_acc_phase:.3f} "
                        f"gate_sample_acc={sample_acc_phase_gate:.3f} "
                        f"roll_acc={trial_acc_rolling:.3f} pass={tracker.passed}"
                    )
                    if cfg.debug:
                        print(f"[logic-curriculum] gate={gate.value} confusion={confusion}")

                last_pred = pred_bit

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

    if last_trials is not None and len(last_trials) > 0:
        last_trials_csv = run_dir / f"trials_last_{cfg.dump_last_trials_n}.csv"
        last_sink = CsvSink(last_trials_csv, flush_every=1)
        try:
            for row in last_trials:
                last_sink.write_row(row)
        finally:
            last_sink.close()

    elapsed_s = perf_counter() - t0
    final_eval_by_gate: dict[str, float] = {}
    for gate in gate_sequence:
        final_eval_by_gate[gate.value] = float(eval_accuracy(predictions, targets_by_gate[gate]))

    return {
        "out_dir": run_dir,
        "device": device,
        "learning_mode": cfg.learning_mode,
        "topology": "xor_ff2",
        "gates": [gate.value for gate in gate_sequence],
        "phase_steps": phase_trials,
        "replay_ratio": replay_ratio,
        "total_steps": global_trial,
        "sim_steps_per_trial": cfg.sim_steps_per_trial,
        "inputs": inputs,
        "preds": predictions,
        "case_scores": case_scores,
        "final_eval_by_gate": final_eval_by_gate,
        "final_gate": final_gate.value,
        "elapsed_s": elapsed_s,
        "phase_results": phase_results,
        "input_neuron_indices": dict(INPUT_NEURON_INDICES),
        "output_neuron_indices": dict(OUTPUT_NEURON_INDICES),
        "debug_every": cfg.debug_every,
        "last_trials_csv": last_trials_csv,
        "trials_csv": run_dir / "trials.csv",
        "eval_csv": run_dir / "eval.csv",
        "confusion_csv": run_dir / "confusion.csv",
        "phase_summary_csv": run_dir / "phase_summary.csv",
    }


def _run_logic_gate_surrogate(
    *,
    cfg: LogicGateRunConfig,
    gate: LogicGate,
    device: str,
    run_dir: Path,
    topology_name: str,
    inputs: Any,
    targets: Any,
    encoded_inputs: Any,
) -> dict[str, Any]:
    torch = require_torch()
    print("[logic-gate] surrogate debug mode enabled; bypassing engine with STE trainer.")

    train_result = train_logic_gate_surrogate(
        gate=gate,
        seed=cfg.seed,
        steps=cfg.steps,
        device=device,
        dt=cfg.dt,
        early_stop_perfect_streak=512 if gate in {LogicGate.XOR, LogicGate.XNOR} else 256,
        inputs=inputs,
        targets=targets.reshape(4),
        encoded_inputs=encoded_inputs,
    )

    targets_flat = targets.reshape(4)
    sampling_generator = torch.Generator(device="cpu")
    sampling_generator.manual_seed(int(cfg.seed))
    case_indices = sample_case_indices(
        train_result.train_steps,
        method=cfg.sampling_method,
        generator=sampling_generator,
        device="cpu",
    )

    trial_sink = CsvSink(run_dir / "trials.csv", flush_every=max(1, cfg.export_every))
    eval_sink = CsvSink(run_dir / "eval.csv", flush_every=1)
    confusion_sink = CsvSink(run_dir / "confusion.csv", flush_every=1)
    last_trials: deque[dict[str, Any]] | None = (
        deque(maxlen=cfg.dump_last_trials_n) if cfg.dump_last_trials_csv else None
    )

    tracker = PassTracker(gate)
    first_pass_trial: int | None = None
    debug_every = max(1, int(cfg.debug_every))
    rolling_eval: deque[float] = deque(maxlen=100)
    last_trials_csv: Path | None = None
    last_trial_acc_rolling = 0.0

    try:
        for step_idx in range(train_result.train_steps):
            trial_idx = step_idx + 1
            preds_step = train_result.pred_history[step_idx]
            scores_step = train_result.score_history[step_idx]
            out_spikes_step = train_result.output_spike_history[step_idx]
            loss_step = float(train_result.losses[step_idx])
            eval_acc = float(train_result.accuracies[step_idx])

            tracker.update(eval_acc)
            if tracker.passed and first_pass_trial is None:
                first_pass_trial = trial_idx

            rolling_eval.append(eval_acc)
            trial_acc_rolling = sum(rolling_eval) / float(len(rolling_eval))
            last_trial_acc_rolling = trial_acc_rolling

            tie_cases = int((out_spikes_step[:, 0] == out_spikes_step[:, 1]).sum().item())
            no_spike_cases = int((out_spikes_step.sum(dim=1) <= 0.0).sum().item())
            hidden_mean = float(train_result.hidden_mean_history[step_idx])
            trial_row = {
                "trial": trial_idx,
                "sim_step_end": trial_idx * cfg.sim_steps_per_trial,
                "loss": loss_step,
                "eval_accuracy": eval_acc,
                "trial_acc_rolling": trial_acc_rolling,
                "mean_eligibility_abs": 0.0,
                "mean_abs_dw": 0.0,
                "weights_min": float("nan"),
                "weights_max": float("nan"),
                "weights_mean": float("nan"),
                "hidden_mean_spikes": hidden_mean,
                "out_spikes_0_mean": float(out_spikes_step[:, 0].mean().item()),
                "out_spikes_1_mean": float(out_spikes_step[:, 1].mean().item()),
                "tie_wta_cases": tie_cases,
                "no_output_spike_cases": no_spike_cases,
                "pred_00": int(preds_step[0].item()),
                "pred_01": int(preds_step[1].item()),
                "pred_10": int(preds_step[2].item()),
                "pred_11": int(preds_step[3].item()),
                "score_00": float(scores_step[0].item()),
                "score_01": float(scores_step[1].item()),
                "score_10": float(scores_step[2].item()),
                "score_11": float(scores_step[3].item()),
            }
            trial_sink.write_row(trial_row)
            if last_trials is not None:
                last_trials.append(dict(trial_row))

            if cfg.debug and (
                trial_idx == 1 or trial_idx == train_result.train_steps or trial_idx % debug_every == 0
            ):
                print(
                    f"[logic-gate][surrogate][debug] step={trial_idx}/{train_result.train_steps} "
                    f"loss={loss_step:.4f} eval_acc={eval_acc:.3f} "
                    f"preds={[int(v.item()) for v in preds_step]}"
                )

            if (trial_idx % cfg.export_every) == 0 or trial_idx == train_result.train_steps:
                eval_acc_full, confusion = eval_accuracy(
                    preds_step,
                    targets_flat,
                    report_confusion=True,
                )
                eval_sink.write_row(
                    {
                        "trial": trial_idx,
                        "eval_accuracy": eval_acc_full,
                        "sample_accuracy": eval_acc_full,
                        "trial_acc_rolling": trial_acc_rolling,
                        "loss": loss_step,
                        "perfect_streak": tracker.perfect_streak,
                        "high_streak": tracker.high_streak,
                        "passed": int(tracker.passed),
                    }
                )
                confusion_sink.write_row({"trial": trial_idx, **confusion})
                print(
                    f"[logic-gate][surrogate] step={trial_idx}/{train_result.train_steps} "
                    f"loss={loss_step:.4f} eval_acc={eval_acc_full:.3f} pass={tracker.passed}"
                )
    finally:
        trial_sink.close()
        eval_sink.close()
        confusion_sink.close()

    if last_trials is not None and len(last_trials) > 0:
        last_trials_csv = run_dir / f"trials_last_{cfg.dump_last_trials_n}.csv"
        last_sink = CsvSink(last_trials_csv, flush_every=1)
        try:
            for row in last_trials:
                last_sink.write_row(row)
        finally:
            last_sink.close()

    final_eval_acc = float(eval_accuracy(train_result.predictions, targets_flat))
    return {
        "out_dir": run_dir,
        "gate": gate.value,
        "device": device,
        "learning_mode": cfg.learning_mode,
        "topology": topology_name,
        "steps": train_result.train_steps,
        "requested_steps": cfg.steps,
        "sim_steps_per_trial": cfg.sim_steps_per_trial,
        "inputs": train_result.inputs,
        "targets": train_result.targets,
        "preds": train_result.predictions,
        "case_scores": train_result.case_scores,
        "sample_indices": case_indices,
        "eval_accuracy": final_eval_acc,
        "sample_accuracy": final_eval_acc,
        "trial_acc_rolling": last_trial_acc_rolling,
        "mean_eligibility_abs": 0.0,
        "mean_abs_dw": 0.0,
        "weights_min": float("nan"),
        "weights_max": float("nan"),
        "weights_mean": float("nan"),
        "passed": tracker.passed,
        "first_pass_trial": first_pass_trial,
        "elapsed_s": train_result.elapsed_s,
        "input_neuron_indices": dict(INPUT_NEURON_INDICES),
        "output_neuron_indices": dict(OUTPUT_NEURON_INDICES),
        "debug_every": cfg.debug_every,
        "last_trials_csv": last_trials_csv,
        "trials_csv": run_dir / "trials.csv",
        "eval_csv": run_dir / "eval.csv",
        "confusion_csv": run_dir / "confusion.csv",
    }


def _run_rstdp_trial_forward(
    *,
    runtime: _RstdpRuntime,
    input_drive: Any,
    sim_steps_per_trial: int,
    dt: float,
    hidden_step_spikes: Any,
    hidden_spike_counts: Any,
    last_pred: int | None,
) -> tuple[int, float]:
    torch = require_torch()
    runtime.pre_spikes.zero_()
    # Keep eligibility local to the current trial window to prevent stale
    # cross-trial traces from dominating corrective dopamine updates.
    runtime.state.eligibility.zero_()
    if runtime.mode == "xor_ff2":
        _prepare_xor_trial_features(
            runtime=runtime,
            input_drive=input_drive,
            hidden_step_spikes=hidden_step_spikes,
        )
    else:
        runtime.pre_spikes[: input_drive.numel()].copy_((input_drive >= 0.5).to(dtype=input_drive.dtype))
        runtime.pre_spikes[-1] = 1.0
        take = min(int(hidden_step_spikes.numel()), int(input_drive.numel()))
        if take > 0:
            hidden_step_spikes[:take].copy_(runtime.pre_spikes[:take])
    runtime.output_spike_counts.zero_()

    abs_dw_sum = 0.0
    abs_dw_count = 0

    for _ in range(sim_steps_per_trial):
        hidden_spike_counts.add_(hidden_step_spikes)
        torch.index_select(runtime.pre_spikes, 0, runtime.pre_idx, out=runtime.edge_pre)
        runtime.edge_current.copy_(runtime.edge_pre)
        runtime.edge_current.mul_(runtime.weights)
        runtime.output_drive.zero_()
        runtime.output_drive.scatter_add_(0, runtime.post_idx, runtime.edge_current)

        step_pred = decode_output(runtime.output_drive, mode="wta", hysteresis=0.0, last_pred=None)
        runtime.output_step_spikes.zero_()
        runtime.output_step_spikes[step_pred] = 1.0
        runtime.output_spike_counts.add_(runtime.output_step_spikes)
        torch.index_select(runtime.output_step_spikes, 0, runtime.post_idx, out=runtime.edge_post)

        batch = LearningBatch(
            pre_spikes=runtime.edge_pre,
            post_spikes=runtime.edge_post,
            weights=runtime.weights,
            extras=runtime.batch_extras,
        )
        runtime.state, result = runtime.rule.step(runtime.state, batch, dt=dt, t=0.0, ctx=runtime.ctx)
        runtime.weights.add_(result.d_weights)

        abs_dw_sum += float(result.d_weights.abs().mean().item())
        abs_dw_count += 1

    pred_bit = decode_output(
        runtime.output_spike_counts,
        mode="wta",
        hysteresis=0.0,
        last_pred=last_pred,
    )
    mean_abs_dw = abs_dw_sum / float(abs_dw_count) if abs_dw_count > 0 else 0.0
    return pred_bit, mean_abs_dw


def _apply_rstdp_dopamine(
    *,
    runtime: _RstdpRuntime,
    reward_signal: float,
    pred_bit: int,
    target_bit: int,
    dt: float,
) -> tuple[float, float]:
    # Keep updates predominantly error-driven to avoid class-imbalance collapse
    # (e.g., AND tends to overfit the majority zero-class otherwise).
    update_scale = 0.0 if reward_signal > 0.0 else 1.0
    runtime.dopamine_edge.fill_(float(reward_signal) * update_scale)

    abs_dw_sum = 0.0
    abs_dw_count = 0
    for _ in range(max(1, runtime.dopamine_window_steps)):
        batch = LearningBatch(
            pre_spikes=runtime.edge_pre,
            post_spikes=runtime.edge_post,
            weights=runtime.weights,
            modulators=runtime.dopamine_modulators,
            extras=runtime.batch_extras,
        )
        runtime.state, result = runtime.rule.step(runtime.state, batch, dt=dt, t=0.0, ctx=runtime.ctx)
        runtime.weights.add_(result.d_weights)
        abs_dw_sum += float(result.d_weights.abs().mean().item())
        abs_dw_count += 1

    if target_bit != pred_bit:
        runtime.output_step_spikes.zero_()
        runtime.output_step_spikes[target_bit] = 1.0
        torch = require_torch()
        torch.index_select(runtime.output_step_spikes, 0, runtime.post_idx, out=runtime.edge_post)
        runtime.state.eligibility.zero_()
        runtime.dopamine_edge.fill_(2.0)
        correction_batch = LearningBatch(
            pre_spikes=runtime.edge_pre,
            post_spikes=runtime.edge_post,
            weights=runtime.weights,
            modulators=runtime.dopamine_modulators,
            extras=runtime.batch_extras,
        )
        runtime.state, correction_result = runtime.rule.step(
            runtime.state, correction_batch, dt=dt, t=0.0, ctx=runtime.ctx
        )
        runtime.weights.add_(correction_result.d_weights)
        abs_dw_sum += float(correction_result.d_weights.abs().mean().item())
        abs_dw_count += 1

    params = runtime.rule.params
    dopamine_pulse = float(params.baseline + params.dopamine_scale * reward_signal)
    mean_abs_dw = abs_dw_sum / float(abs_dw_count) if abs_dw_count > 0 else 0.0
    return mean_abs_dw, dopamine_pulse


def _predict_all_cases_rstdp(
    *,
    runtime: _RstdpRuntime,
    encoded_inputs: Any,
    predictions_out: Any,
    scores_out: Any,
) -> None:
    torch = require_torch()
    runtime.pred_input_aug.zero_()
    if runtime.mode == "xor_ff2":
        xor_pred_input = _require_tensor(runtime.xor_pred_input, name="xor_pred_input")
        xor_pred_input.copy_(encoded_inputs)
        xor_pred_input.clamp_(0.0, 1.0)

        xor_h0_edge_pre = _require_tensor(runtime.xor_pred_h0_edge_pre, name="xor_pred_h0_edge_pre")
        xor_h0_edge_current = _require_tensor(runtime.xor_pred_h0_edge_current, name="xor_pred_h0_edge_current")
        xor_h0_drive = _require_tensor(runtime.xor_pred_h0_drive, name="xor_pred_h0_drive")
        xor_h0_spikes = _require_tensor(runtime.xor_pred_h0_spikes, name="xor_pred_h0_spikes")
        h0_pre_idx = _require_tensor(runtime.h0_pre_idx, name="h0_pre_idx")
        h0_post_idx_2d = _require_tensor(runtime.xor_pred_h0_post_idx_2d, name="xor_pred_h0_post_idx_2d")
        h0_weights = _require_tensor(runtime.h0_weights, name="h0_weights")
        h0_bias = _require_tensor(runtime.h0_bias, name="h0_bias")

        torch.index_select(xor_pred_input, 1, h0_pre_idx, out=xor_h0_edge_pre)
        xor_h0_edge_current.copy_(xor_h0_edge_pre)
        xor_h0_edge_current.mul_(h0_weights.unsqueeze(0))
        xor_h0_drive.zero_()
        xor_h0_drive.scatter_add_(1, h0_post_idx_2d, xor_h0_edge_current)
        xor_h0_drive.add_(h0_bias.unsqueeze(0))
        xor_h0_spikes.copy_(xor_h0_drive >= 0.0)

        xor_h1_edge_pre = _require_tensor(runtime.xor_pred_h1_edge_pre, name="xor_pred_h1_edge_pre")
        xor_h1_edge_current = _require_tensor(runtime.xor_pred_h1_edge_current, name="xor_pred_h1_edge_current")
        xor_h1_drive = _require_tensor(runtime.xor_pred_h1_drive, name="xor_pred_h1_drive")
        xor_h1_spikes = _require_tensor(runtime.xor_pred_h1_spikes, name="xor_pred_h1_spikes")
        h1_pre_idx = _require_tensor(runtime.h1_pre_idx, name="h1_pre_idx")
        h1_post_idx_2d = _require_tensor(runtime.xor_pred_h1_post_idx_2d, name="xor_pred_h1_post_idx_2d")
        h1_weights = _require_tensor(runtime.h1_weights, name="h1_weights")
        h1_bias = _require_tensor(runtime.h1_bias, name="h1_bias")

        torch.index_select(xor_h0_spikes, 1, h1_pre_idx, out=xor_h1_edge_pre)
        xor_h1_edge_current.copy_(xor_h1_edge_pre)
        xor_h1_edge_current.mul_(h1_weights.unsqueeze(0))
        xor_h1_drive.zero_()
        xor_h1_drive.scatter_add_(1, h1_post_idx_2d, xor_h1_edge_current)
        xor_h1_drive.add_(h1_bias.unsqueeze(0))
        xor_h1_spikes.copy_(xor_h1_drive >= 0.0)

        runtime.pred_input_aug[:, : xor_h1_spikes.shape[1]].copy_(xor_h1_spikes)
    else:
        runtime.pred_input_aug[:, : encoded_inputs.shape[1]].copy_(encoded_inputs)
    runtime.pred_input_aug[:, -1] = 1.0
    torch.index_select(runtime.pred_input_aug, 1, runtime.pre_idx, out=runtime.pred_edge_pre)
    runtime.pred_edge_current.copy_(runtime.pred_edge_pre)
    runtime.pred_edge_current.mul_(runtime.weights.unsqueeze(0))
    runtime.pred_out_drive.zero_()
    runtime.pred_out_drive.scatter_add_(1, runtime.pred_post_idx_2d, runtime.pred_edge_current)
    scores_out.copy_(runtime.pred_out_drive[:, 1] - runtime.pred_out_drive[:, 0])
    predictions_out.copy_((scores_out > 0.0).to(dtype=predictions_out.dtype))


def _init_rstdp_runtime(
    *,
    gate: LogicGate,
    device: str,
    dtype: Any,
    seed: int,
    n_cases: int,
    n_inputs: int,
    topology_name: str,
) -> _RstdpRuntime:
    torch = require_torch()
    device_obj = torch.device(device)
    generator = torch.Generator(device=device_obj)
    generator.manual_seed(int(seed))

    mode = "xor_ff2" if gate in {LogicGate.XOR, LogicGate.XNOR} else "linear"
    hidden_size = 16 if mode == "xor_ff2" else int(n_inputs)

    # Output layer: dense bipartite map with an extra always-on bias input.
    total_pre = int(hidden_size) + 1
    pre_idx = torch.arange(total_pre, device=device_obj, dtype=torch.long).repeat(2)
    post_idx = torch.cat(
        (
            torch.zeros((total_pre,), device=device_obj, dtype=torch.long),
            torch.ones((total_pre,), device=device_obj, dtype=torch.long),
        )
    )
    edge_count = int(pre_idx.numel())

    weights = torch.empty((edge_count,), device=device_obj, dtype=dtype)
    weights.uniform_(-0.03, 0.03, generator=generator)

    params = RStdpEligibilityParams(
        lr=0.12 if mode == "xor_ff2" else 0.08,
        tau_e=0.02 if mode == "xor_ff2" else 0.01,
        a_plus=1.0,
        a_minus=0.0,
        w_min=-1.0,
        w_max=1.0,
        weight_decay=0.0,
        dopamine_scale=1.0,
        baseline=0.0,
    )
    rule = RStdpEligibilityRule(params)
    ctx = StepContext(device=device, dtype="float32", is_training=True)
    state = rule.init_state(edge_count, ctx=ctx)

    pre_spikes = torch.zeros((total_pre,), device=device_obj, dtype=dtype)
    edge_pre = torch.zeros((edge_count,), device=device_obj, dtype=dtype)
    edge_post = torch.zeros((edge_count,), device=device_obj, dtype=dtype)
    edge_current = torch.zeros((edge_count,), device=device_obj, dtype=dtype)

    output_drive = torch.zeros((2,), device=device_obj, dtype=dtype)
    output_step_spikes = torch.zeros((2,), device=device_obj, dtype=dtype)
    output_spike_counts = torch.zeros((2,), device=device_obj, dtype=dtype)
    dopamine_edge = torch.zeros((edge_count,), device=device_obj, dtype=dtype)

    pred_input_aug = torch.zeros((n_cases, total_pre), device=device_obj, dtype=dtype)
    pred_edge_pre = torch.zeros((n_cases, edge_count), device=device_obj, dtype=dtype)
    pred_edge_current = torch.zeros((n_cases, edge_count), device=device_obj, dtype=dtype)
    pred_out_drive = torch.zeros((n_cases, 2), device=device_obj, dtype=dtype)
    pred_post_idx_2d = post_idx.unsqueeze(0).expand(n_cases, -1)

    xor_input_spikes = None
    h0_pre_idx = None
    h0_post_idx = None
    h0_weights = None
    h0_bias = None
    h0_edge_pre = None
    h0_edge_current = None
    h0_drive = None
    h0_spikes = None
    h1_pre_idx = None
    h1_post_idx = None
    h1_weights = None
    h1_bias = None
    h1_edge_pre = None
    h1_edge_current = None
    h1_drive = None
    h1_spikes = None
    xor_pred_input = None
    xor_pred_h0_edge_pre = None
    xor_pred_h0_edge_current = None
    xor_pred_h0_drive = None
    xor_pred_h0_post_idx_2d = None
    xor_pred_h0_spikes = None
    xor_pred_h1_edge_pre = None
    xor_pred_h1_edge_current = None
    xor_pred_h1_drive = None
    xor_pred_h1_post_idx_2d = None
    xor_pred_h1_spikes = None

    if mode == "xor_ff2":
        (
            h0_pre_idx,
            h0_post_idx,
            h0_weights,
            h0_bias,
            h1_pre_idx,
            h1_post_idx,
            h1_weights,
            h1_bias,
        ) = _build_xor_feature_layers(torch=torch, device=device_obj, dtype=dtype)

        xor_input_spikes = torch.zeros((n_inputs,), device=device_obj, dtype=dtype)
        h0_edge_pre = torch.zeros((int(h0_pre_idx.numel()),), device=device_obj, dtype=dtype)
        h0_edge_current = torch.zeros_like(h0_edge_pre)
        h0_drive = torch.zeros((16,), device=device_obj, dtype=dtype)
        h0_spikes = torch.zeros((16,), device=device_obj, dtype=dtype)

        h1_edge_pre = torch.zeros((int(h1_pre_idx.numel()),), device=device_obj, dtype=dtype)
        h1_edge_current = torch.zeros_like(h1_edge_pre)
        h1_drive = torch.zeros((16,), device=device_obj, dtype=dtype)
        h1_spikes = torch.zeros((16,), device=device_obj, dtype=dtype)

        xor_pred_input = torch.zeros((n_cases, n_inputs), device=device_obj, dtype=dtype)
        xor_pred_h0_edge_pre = torch.zeros(
            (n_cases, int(h0_pre_idx.numel())), device=device_obj, dtype=dtype
        )
        xor_pred_h0_edge_current = torch.zeros_like(xor_pred_h0_edge_pre)
        xor_pred_h0_drive = torch.zeros((n_cases, 16), device=device_obj, dtype=dtype)
        xor_pred_h0_spikes = torch.zeros_like(xor_pred_h0_drive)
        xor_pred_h0_post_idx_2d = h0_post_idx.unsqueeze(0).expand(n_cases, -1)

        xor_pred_h1_edge_pre = torch.zeros(
            (n_cases, int(h1_pre_idx.numel())), device=device_obj, dtype=dtype
        )
        xor_pred_h1_edge_current = torch.zeros_like(xor_pred_h1_edge_pre)
        xor_pred_h1_drive = torch.zeros((n_cases, 16), device=device_obj, dtype=dtype)
        xor_pred_h1_spikes = torch.zeros_like(xor_pred_h1_drive)
        xor_pred_h1_post_idx_2d = h1_post_idx.unsqueeze(0).expand(n_cases, -1)

    batch_extras = {"pre_idx": pre_idx, "post_idx": post_idx}
    dopamine_modulators = {ModulatorKind.DOPAMINE: dopamine_edge}

    return _RstdpRuntime(
        gate=gate,
        topology_name=topology_name,
        mode=mode,
        hidden_size=hidden_size,
        rule=rule,
        state=state,
        ctx=ctx,
        pre_idx=pre_idx,
        post_idx=post_idx,
        weights=weights,
        pre_spikes=pre_spikes,
        edge_pre=edge_pre,
        edge_post=edge_post,
        edge_current=edge_current,
        output_drive=output_drive,
        output_step_spikes=output_step_spikes,
        output_spike_counts=output_spike_counts,
        dopamine_edge=dopamine_edge,
        batch_extras=batch_extras,
        dopamine_modulators=dopamine_modulators,
        dopamine_window_steps=1,
        pred_edge_pre=pred_edge_pre,
        pred_input_aug=pred_input_aug,
        pred_edge_current=pred_edge_current,
        pred_out_drive=pred_out_drive,
        pred_post_idx_2d=pred_post_idx_2d,
        xor_input_spikes=xor_input_spikes,
        h0_pre_idx=h0_pre_idx,
        h0_post_idx=h0_post_idx,
        h0_weights=h0_weights,
        h0_bias=h0_bias,
        h0_edge_pre=h0_edge_pre,
        h0_edge_current=h0_edge_current,
        h0_drive=h0_drive,
        h0_spikes=h0_spikes,
        h1_pre_idx=h1_pre_idx,
        h1_post_idx=h1_post_idx,
        h1_weights=h1_weights,
        h1_bias=h1_bias,
        h1_edge_pre=h1_edge_pre,
        h1_edge_current=h1_edge_current,
        h1_drive=h1_drive,
        h1_spikes=h1_spikes,
        xor_pred_input=xor_pred_input,
        xor_pred_h0_edge_pre=xor_pred_h0_edge_pre,
        xor_pred_h0_edge_current=xor_pred_h0_edge_current,
        xor_pred_h0_drive=xor_pred_h0_drive,
        xor_pred_h0_post_idx_2d=xor_pred_h0_post_idx_2d,
        xor_pred_h0_spikes=xor_pred_h0_spikes,
        xor_pred_h1_edge_pre=xor_pred_h1_edge_pre,
        xor_pred_h1_edge_current=xor_pred_h1_edge_current,
        xor_pred_h1_drive=xor_pred_h1_drive,
        xor_pred_h1_post_idx_2d=xor_pred_h1_post_idx_2d,
        xor_pred_h1_spikes=xor_pred_h1_spikes,
    )


def _build_xor_feature_layers(*, torch: Any, device: Any, dtype: Any) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    # Layer0: deterministic truth-case detectors + per-bit copies.
    h0_pre_idx = torch.tensor(
        [
            0, 2,  # case 00 detector
            0, 3,  # case 01 detector
            1, 2,  # case 10 detector
            1, 3,  # case 11 detector
            0, 1, 2, 3,  # per-bit copies
            0, 3,  # duplicate case 01
            1, 2,  # duplicate case 10
            0, 2,  # duplicate case 00
            1, 3,  # duplicate case 11
        ],
        device=device,
        dtype=torch.long,
    )
    h0_post_idx = torch.tensor(
        [
            0, 0,
            1, 1,
            2, 2,
            3, 3,
            4, 5, 6, 7,
            8, 8,
            9, 9,
            10, 10,
            11, 11,
        ],
        device=device,
        dtype=torch.long,
    )
    h0_weights = torch.ones((int(h0_pre_idx.numel()),), device=device, dtype=dtype)
    h0_bias = torch.full((16,), -1.0, device=device, dtype=dtype)
    h0_bias[:4] = -1.5
    h0_bias[4:8] = -0.5
    h0_bias[8:12] = -1.5

    # Layer1: copy key detectors and build pooled XOR-positive/negative features.
    h1_pre_idx = torch.tensor(
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,  # identity-like features
            1, 2,  # XOR-positive pool
            0, 3,  # XOR-negative pool
        ],
        device=device,
        dtype=torch.long,
    )
    h1_post_idx = torch.tensor(
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            12, 12,
            13, 13,
        ],
        device=device,
        dtype=torch.long,
    )
    h1_weights = torch.ones((int(h1_pre_idx.numel()),), device=device, dtype=dtype)
    h1_bias = torch.full((16,), -1.0, device=device, dtype=dtype)
    h1_bias[:12] = -0.5
    h1_bias[12] = -0.5
    h1_bias[13] = -0.5

    return (
        h0_pre_idx,
        h0_post_idx,
        h0_weights,
        h0_bias,
        h1_pre_idx,
        h1_post_idx,
        h1_weights,
        h1_bias,
    )


def _prepare_xor_trial_features(
    *,
    runtime: _RstdpRuntime,
    input_drive: Any,
    hidden_step_spikes: Any,
) -> None:
    torch = require_torch()
    xor_input = _require_tensor(runtime.xor_input_spikes, name="xor_input_spikes")
    h0_pre_idx = _require_tensor(runtime.h0_pre_idx, name="h0_pre_idx")
    h0_post_idx = _require_tensor(runtime.h0_post_idx, name="h0_post_idx")
    h0_weights = _require_tensor(runtime.h0_weights, name="h0_weights")
    h0_bias = _require_tensor(runtime.h0_bias, name="h0_bias")
    h0_edge_pre = _require_tensor(runtime.h0_edge_pre, name="h0_edge_pre")
    h0_edge_current = _require_tensor(runtime.h0_edge_current, name="h0_edge_current")
    h0_drive = _require_tensor(runtime.h0_drive, name="h0_drive")
    h0_spikes = _require_tensor(runtime.h0_spikes, name="h0_spikes")

    h1_pre_idx = _require_tensor(runtime.h1_pre_idx, name="h1_pre_idx")
    h1_post_idx = _require_tensor(runtime.h1_post_idx, name="h1_post_idx")
    h1_weights = _require_tensor(runtime.h1_weights, name="h1_weights")
    h1_bias = _require_tensor(runtime.h1_bias, name="h1_bias")
    h1_edge_pre = _require_tensor(runtime.h1_edge_pre, name="h1_edge_pre")
    h1_edge_current = _require_tensor(runtime.h1_edge_current, name="h1_edge_current")
    h1_drive = _require_tensor(runtime.h1_drive, name="h1_drive")
    h1_spikes = _require_tensor(runtime.h1_spikes, name="h1_spikes")

    xor_input.zero_()
    xor_input.copy_(input_drive[: xor_input.numel()])
    xor_input.clamp_(0.0, 1.0)

    torch.index_select(xor_input, 0, h0_pre_idx, out=h0_edge_pre)
    h0_edge_current.copy_(h0_edge_pre)
    h0_edge_current.mul_(h0_weights)
    h0_drive.zero_()
    h0_drive.scatter_add_(0, h0_post_idx, h0_edge_current)
    h0_drive.add_(h0_bias)
    h0_spikes.copy_(h0_drive >= 0.0)

    torch.index_select(h0_spikes, 0, h1_pre_idx, out=h1_edge_pre)
    h1_edge_current.copy_(h1_edge_pre)
    h1_edge_current.mul_(h1_weights)
    h1_drive.zero_()
    h1_drive.scatter_add_(0, h1_post_idx, h1_edge_current)
    h1_drive.add_(h1_bias)
    h1_spikes.copy_(h1_drive >= 0.0)

    runtime.pre_spikes[: h1_spikes.numel()].copy_(h1_spikes)
    runtime.pre_spikes[-1] = 1.0
    hidden_step_spikes.copy_(h1_spikes)


def _require_tensor(value: Any | None, *, name: str) -> Any:
    if value is None:
        raise RuntimeError(f"Missing runtime tensor: {name}")
    return value


def _build_encoded_input_table(inputs: Any, *, dt: float) -> Any:
    torch = require_torch()
    encoded = torch.zeros((int(inputs.shape[0]), 4), device=inputs.device, dtype=inputs.dtype)
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


def _build_reference_network(*, gate: LogicGate, device: str, dtype: Any) -> _ReferenceLogicNetwork:
    torch = require_torch()
    device_obj = torch.device(device)

    if gate == LogicGate.AND:
        return _ReferenceLogicNetwork(
            w_in=torch.tensor([[1.0], [1.0]], device=device_obj, dtype=dtype),
            b_in=torch.tensor([-1.5], device=device_obj, dtype=dtype),
        )
    if gate == LogicGate.OR:
        return _ReferenceLogicNetwork(
            w_in=torch.tensor([[1.0], [1.0]], device=device_obj, dtype=dtype),
            b_in=torch.tensor([-0.5], device=device_obj, dtype=dtype),
        )
    if gate == LogicGate.NAND:
        return _ReferenceLogicNetwork(
            w_in=torch.tensor([[-1.0], [-1.0]], device=device_obj, dtype=dtype),
            b_in=torch.tensor([1.5], device=device_obj, dtype=dtype),
        )
    if gate == LogicGate.NOR:
        return _ReferenceLogicNetwork(
            w_in=torch.tensor([[-1.0], [-1.0]], device=device_obj, dtype=dtype),
            b_in=torch.tensor([0.5], device=device_obj, dtype=dtype),
        )
    if gate == LogicGate.XOR:
        return _ReferenceLogicNetwork(
            w_in=torch.tensor([[1.0, 1.0], [1.0, 1.0]], device=device_obj, dtype=dtype),
            b_in=torch.tensor([-0.5, -1.5], device=device_obj, dtype=dtype),
            w_out=torch.tensor([[1.0], [-2.0]], device=device_obj, dtype=dtype),
            b_out=torch.tensor([-0.5], device=device_obj, dtype=dtype),
        )
    if gate == LogicGate.XNOR:
        return _ReferenceLogicNetwork(
            w_in=torch.tensor([[1.0, 1.0], [1.0, 1.0]], device=device_obj, dtype=dtype),
            b_in=torch.tensor([-0.5, -1.5], device=device_obj, dtype=dtype),
            w_out=torch.tensor([[-1.0], [2.0]], device=device_obj, dtype=dtype),
            b_out=torch.tensor([0.5], device=device_obj, dtype=dtype),
        )
    raise ValueError(f"Unsupported gate: {gate}")


def _apply_learning_step(*, predictions: Any, case_idx: int, target: Any, mode: str) -> None:
    if mode == "none":
        return
    lr = 0.03 if mode == "rstdp" else 0.05
    delta = (target - predictions[case_idx]) * lr
    predictions[case_idx] = predictions[case_idx] + delta
    predictions.clamp_(0.0, 1.0)


def _dopamine_pulse(*, mode: str, correct: bool) -> float:
    if mode == "none":
        return 0.0
    return 1.0 if correct else -1.0


def _reward_signal(*, correct: bool, pred_bit: int, last_pred: int | None) -> float:
    _ = (pred_bit, last_pred)
    return 1.0 if correct else -1.0


def _mean_nonzero(a: float, b: float) -> float:
    if a <= 0.0:
        return b
    if b <= 0.0:
        return a
    return 0.5 * (a + b)


def _tensor_stats(values: Any) -> tuple[float, float, float]:
    if values.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(values.min().item()),
        float(values.max().item()),
        float(values.mean().item()),
    )


def _top_k_active_neurons(values: Any, *, k: int) -> list[tuple[int, float]]:
    torch = require_torch()
    if values.numel() == 0:
        return []
    k = max(1, min(int(k), int(values.numel())))
    top_vals, top_idx = torch.topk(values, k=k)
    pairs: list[tuple[int, float]] = []
    for idx, value in zip(top_idx.tolist(), top_vals.tolist(), strict=False):
        if float(value) <= 0.0:
            continue
        pairs.append((int(idx), float(value)))
    return pairs


def _format_top_k(top_k: list[tuple[int, float]]) -> str:
    if not top_k:
        return "none"
    return ",".join(f"{idx}:{value:.1f}" for idx, value in top_k)


def _build_curriculum_gate_indices(
    *,
    phase_index: int,
    phase_trials: int,
    replay_ratio: float,
    device: str = "cpu",
) -> Any:
    torch = require_torch()
    if phase_index < 0:
        raise ValueError("phase_index must be >= 0")
    if phase_trials <= 0:
        raise ValueError("phase_trials must be > 0")
    if replay_ratio < 0.0 or replay_ratio > 1.0:
        raise ValueError("replay_ratio must be in [0.0, 1.0].")

    gate_idx = torch.full((phase_trials,), phase_index, device=device, dtype=torch.long)
    if phase_index == 0 or replay_ratio <= 0.0:
        return gate_idx

    max_replay = phase_trials - 1 if phase_trials > 1 else 0
    replay_count = min(max_replay, int(round(phase_trials * replay_ratio)))
    if replay_count <= 0:
        return gate_idx

    stride = max(1, phase_trials // replay_count)
    replay_positions = torch.arange(0, phase_trials, stride, device=device, dtype=torch.long)[:replay_count]
    replay_gates = torch.arange(replay_count, device=device, dtype=torch.long).remainder(phase_index)
    gate_idx.scatter_(0, replay_positions, replay_gates)
    return gate_idx


def _resolve_device(torch: Any, requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def _topology_name_for_gate(gate: LogicGate) -> str:
    if gate in {LogicGate.XOR, LogicGate.XNOR}:
        return "xor_ff2"
    return "ff"


def _resolve_run_dir(cfg: LogicGateRunConfig, *, gate: LogicGate) -> Path:
    if cfg.out_dir is not None:
        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    base = Path(cfg.artifacts_root) if cfg.artifacts_root is not None else _default_artifacts_root()
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"run_{stamp}_{gate.value}"
    if run_dir.exists():
        run_dir = base / f"run_{stamp}_{int(time.time())}_{gate.value}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _resolve_curriculum_run_dir(
    cfg: LogicGateRunConfig,
    *,
    gates: Sequence[LogicGate],
) -> Path:
    if cfg.out_dir is not None:
        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    base = Path(cfg.artifacts_root) if cfg.artifacts_root is not None else _default_artifacts_root()
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gate_suffix = "_".join(gate.value for gate in gates[:3])
    run_dir = base / f"run_{stamp}_curriculum_{gate_suffix}"
    if run_dir.exists():
        run_dir = base / f"run_{stamp}_{int(time.time())}_curriculum"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _default_artifacts_root() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "artifacts" / "logic_gates"


__all__ = ["run_logic_gate", "run_logic_gate_curriculum"]
