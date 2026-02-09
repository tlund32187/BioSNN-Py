"""Evaluation utilities for logic-gate task runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, overload

from biosnn.core.torch_utils import require_torch

from .datasets import LogicGate, coerce_gate


@dataclass(frozen=True, slots=True)
class PassCriterion:
    perfect_consecutive: int
    high_consecutive: int
    high_threshold: float = 0.99


@dataclass(slots=True)
class PassTracker:
    gate: LogicGate
    perfect_streak: int = 0
    high_streak: int = 0
    passed: bool = False

    def update(self, accuracy: float) -> bool:
        criterion = gate_pass_criterion(self.gate)
        self.perfect_streak = self.perfect_streak + 1 if accuracy >= 1.0 else 0
        self.high_streak = self.high_streak + 1 if accuracy >= criterion.high_threshold else 0
        self.passed = (
            self.passed
            or self.perfect_streak >= criterion.perfect_consecutive
            or self.high_streak >= criterion.high_consecutive
        )
        return self.passed


@overload
def eval_accuracy(
    preds: Any,
    targets: Any,
    *,
    threshold: float = 0.5,
    report_confusion: Literal[False] = False,
) -> float: ...


@overload
def eval_accuracy(
    preds: Any,
    targets: Any,
    *,
    threshold: float = 0.5,
    report_confusion: Literal[True],
) -> tuple[float, dict[str, int]]: ...


def eval_accuracy(
    preds: Any,
    targets: Any,
    *,
    threshold: float = 0.5,
    report_confusion: bool = False,
) -> float | tuple[float, dict[str, int]]:
    """Evaluate binary accuracy for 4-case truth-table predictions."""

    torch = require_torch()
    pred_bits = _to_binary(preds, threshold=threshold)
    target_bits = _to_binary(targets, threshold=threshold)

    if int(pred_bits.numel()) != 4 or int(target_bits.numel()) != 4:
        raise ValueError("eval_accuracy expects exactly 4 predictions and 4 targets.")

    correct = pred_bits == target_bits
    accuracy = float(correct.to(dtype=torch.float32).mean().item())
    if not report_confusion:
        return accuracy

    confusion = confusion_matrix(pred_bits, target_bits)
    return accuracy, confusion


def confusion_matrix(
    preds: Any,
    targets: Any,
    *,
    threshold: float = 0.5,
) -> dict[str, int]:
    torch = require_torch()
    pred_bits = _to_binary(preds, threshold=threshold)
    target_bits = _to_binary(targets, threshold=threshold)

    tp = int(torch.logical_and(pred_bits == 1, target_bits == 1).sum().item())
    tn = int(torch.logical_and(pred_bits == 0, target_bits == 0).sum().item())
    fp = int(torch.logical_and(pred_bits == 1, target_bits == 0).sum().item())
    fn = int(torch.logical_and(pred_bits == 0, target_bits == 1).sum().item())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def gate_pass_criterion(gate: LogicGate | str) -> PassCriterion:
    gate_enum = coerce_gate(gate)
    if gate_enum in {LogicGate.AND, LogicGate.OR, LogicGate.NAND, LogicGate.NOR}:
        return PassCriterion(perfect_consecutive=200, high_consecutive=500)
    if gate_enum in {LogicGate.XOR, LogicGate.XNOR}:
        return PassCriterion(perfect_consecutive=500, high_consecutive=2000)
    raise ValueError(f"Unsupported gate: {gate_enum}")


def _to_binary(values: Any, *, threshold: float) -> Any:
    torch = require_torch()
    flat = values.reshape(-1)
    if flat.dtype == torch.bool:
        return flat.to(dtype=torch.int64)
    return (flat >= float(threshold)).to(dtype=torch.int64)


__all__ = [
    "PassCriterion",
    "PassTracker",
    "confusion_matrix",
    "eval_accuracy",
    "gate_pass_criterion",
]
