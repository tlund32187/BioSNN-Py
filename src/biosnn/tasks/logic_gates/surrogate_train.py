"""Surrogate-gradient debug trainer for logic-gate tasks."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

from biosnn.contracts.neurons import Compartment
from biosnn.core.torch_utils import require_torch

from .datasets import LogicGate, coerce_gate, make_truth_table
from .encoding import decode_output, encode_inputs
from .evaluators import eval_accuracy

OptimizerName = Literal["adam", "sgd"]


@dataclass(slots=True)
class SurrogateTrainResult:
    gate: LogicGate
    train_steps: int
    elapsed_s: float
    inputs: Any
    targets: Any
    encoded_inputs: Any
    predictions: Any
    case_scores: Any
    output_spikes: Any
    hidden_spikes: Any
    losses: list[float]
    accuracies: list[float]
    pred_history: list[Any]
    score_history: list[Any]
    output_spike_history: list[Any]
    hidden_mean_history: list[float]
    first_perfect_step: int | None


def train_logic_gate_surrogate(
    *,
    gate: LogicGate | str,
    seed: int,
    steps: int,
    device: str = "cpu",
    dt: float = 1e-3,
    lr: float = 0.08,
    hidden_size: int = 16,
    optimizer: OptimizerName = "adam",
    early_stop_perfect_streak: int = 64,
    inputs: Any | None = None,
    targets: Any | None = None,
    encoded_inputs: Any | None = None,
) -> SurrogateTrainResult:
    """Train a tiny STE network on logic-gate truth-table inputs."""

    torch = require_torch()
    gate_enum = coerce_gate(gate)
    if steps <= 0:
        raise ValueError("steps must be > 0")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be > 0")

    resolved_device = _resolve_device(torch, device)
    device_obj = torch.device(resolved_device)
    torch.manual_seed(int(seed))
    if resolved_device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    truth_inputs, truth_targets = make_truth_table(gate_enum, device=resolved_device, dtype="float32")
    if inputs is None:
        inputs = truth_inputs
    if targets is None:
        targets = truth_targets.reshape(4)
    else:
        targets = targets.reshape(-1).to(device=device_obj, dtype=truth_inputs.dtype)
    if encoded_inputs is None:
        encoded_inputs = _build_encoded_input_table(inputs, dt=dt)
    else:
        encoded_inputs = encoded_inputs.to(device=device_obj, dtype=truth_inputs.dtype)

    labels = (targets >= 0.5).to(dtype=torch.long)
    in_features = int(encoded_inputs.shape[1])

    w_in = torch.empty((in_features, hidden_size), device=device_obj, dtype=encoded_inputs.dtype)
    torch.nn.init.xavier_uniform_(w_in)
    w_in.requires_grad_(True)
    b_in = torch.zeros((hidden_size,), device=device_obj, dtype=encoded_inputs.dtype, requires_grad=True)
    w_out = torch.empty((hidden_size, 2), device=device_obj, dtype=encoded_inputs.dtype)
    torch.nn.init.xavier_uniform_(w_out)
    w_out.requires_grad_(True)
    b_out = torch.zeros((2,), device=device_obj, dtype=encoded_inputs.dtype, requires_grad=True)

    params = [w_in, b_in, w_out, b_out]
    if optimizer == "sgd":
        optim = torch.optim.SGD(params, lr=float(lr))
    elif optimizer == "adam":
        optim = torch.optim.Adam(params, lr=float(lr))
    else:
        raise ValueError("optimizer must be one of: adam, sgd")
    criterion = torch.nn.CrossEntropyLoss()

    losses: list[float] = []
    accuracies: list[float] = []
    pred_history: list[Any] = []
    score_history: list[Any] = []
    output_spike_history: list[Any] = []
    hidden_mean_history: list[float] = []
    perfect_streak = 0
    first_perfect_step: int | None = None

    t0 = perf_counter()
    for train_step in range(1, int(steps) + 1):
        logits, hidden_spikes, output_spikes = _forward_surrogate(
            encoded_inputs=encoded_inputs,
            w_in=w_in,
            b_in=b_in,
            w_out=w_out,
            b_out=b_out,
        )
        loss = criterion(logits, labels)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        with torch.no_grad():
            case_scores = logits[:, 1] - logits[:, 0]
            preds = _decode_predictions(logits=logits, output_spikes=output_spikes)
            eval_acc = float(eval_accuracy(preds, targets))

            losses.append(float(loss.item()))
            accuracies.append(eval_acc)
            pred_history.append(preds.detach().clone())
            score_history.append(case_scores.detach().clone())
            output_spike_history.append(output_spikes.detach().clone())
            hidden_mean_history.append(float(hidden_spikes.mean().item()))

            if eval_acc >= 1.0:
                perfect_streak += 1
                if first_perfect_step is None:
                    first_perfect_step = train_step
            else:
                perfect_streak = 0
            if perfect_streak >= max(1, int(early_stop_perfect_streak)):
                break

    elapsed_s = perf_counter() - t0
    with torch.no_grad():
        logits, hidden_spikes, output_spikes = _forward_surrogate(
            encoded_inputs=encoded_inputs,
            w_in=w_in,
            b_in=b_in,
            w_out=w_out,
            b_out=b_out,
        )
        case_scores = logits[:, 1] - logits[:, 0]
        predictions = _decode_predictions(logits=logits, output_spikes=output_spikes)

    return SurrogateTrainResult(
        gate=gate_enum,
        train_steps=len(losses),
        elapsed_s=elapsed_s,
        inputs=inputs,
        targets=targets.reshape(4, 1),
        encoded_inputs=encoded_inputs,
        predictions=predictions,
        case_scores=case_scores,
        output_spikes=output_spikes,
        hidden_spikes=hidden_spikes,
        losses=losses,
        accuracies=accuracies,
        pred_history=pred_history,
        score_history=score_history,
        output_spike_history=output_spike_history,
        hidden_mean_history=hidden_mean_history,
        first_perfect_step=first_perfect_step,
    )


def _forward_surrogate(
    *,
    encoded_inputs: Any,
    w_in: Any,
    b_in: Any,
    w_out: Any,
    b_out: Any,
) -> tuple[Any, Any, Any]:
    hidden_voltage = encoded_inputs.matmul(w_in) + b_in
    hidden_spikes = _surrogate_spike(hidden_voltage)
    output_voltage = hidden_spikes.matmul(w_out) + b_out
    output_spikes = _surrogate_spike(output_voltage)
    return output_voltage, hidden_spikes, output_spikes


def _surrogate_spike(values: Any, *, slope: float = 5.0) -> Any:
    torch = require_torch()
    hard = (values >= 0.0).to(dtype=values.dtype)
    soft = torch.sigmoid(values * float(slope))
    # Straight-through estimator: hard spikes in forward pass, smooth gradient in backward.
    return hard + soft - soft.detach()


def _decode_predictions(*, logits: Any, output_spikes: Any) -> Any:
    torch = require_torch()
    n_cases = int(logits.shape[0])
    preds = torch.zeros((n_cases,), device=logits.device, dtype=logits.dtype)
    for idx in range(n_cases):
        tie_break = int(torch.argmax(logits[idx]).item())
        pred = decode_output(
            output_spikes[idx],
            mode="wta",
            hysteresis=0.0,
            last_pred=tie_break,
        )
        preds[idx] = float(pred)
    return preds


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
        )
        encoded[idx].copy_(drive[Compartment.SOMA])
    return encoded


def _resolve_device(torch: Any, requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


__all__ = ["SurrogateTrainResult", "train_logic_gate_surrogate"]
