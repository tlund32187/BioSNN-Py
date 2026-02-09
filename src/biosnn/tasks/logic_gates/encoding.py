"""Input encoding and output decoding for logic-gate harnesses."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Literal

from biosnn.contracts.neurons import Compartment
from biosnn.core.torch_utils import require_torch

InputEncodeMode = Literal["rate", "spike"]
OutputDecodeMode = Literal["wta", "threshold"]

# Per-bit one-hot encoding (4 input neurons total):
# - bit0_0: bit0 is 0
# - bit0_1: bit0 is 1
# - bit1_0: bit1 is 0
# - bit1_1: bit1 is 1
INPUT_NEURON_INDICES = MappingProxyType(
    {
        "bit0_0": 0,
        "bit0_1": 1,
        "bit1_0": 2,
        "bit1_1": 3,
    }
)

# Two output neurons for class 0 / class 1 with WTA decode.
OUTPUT_NEURON_INDICES = MappingProxyType(
    {
        "class_0": 0,
        "class_1": 1,
    }
)


def encode_inputs(
    inputs: Any,
    mode: InputEncodeMode = "rate",
    *,
    dt: float,
    high: float = 1.0,
    low: float = 0.0,
    compartment: Compartment = Compartment.SOMA,
) -> dict[Compartment, Any]:
    """Encode 2-bit inputs into a per-bit one-hot, 4-neuron drive vector."""

    torch = require_torch()
    flat = inputs.reshape(-1)
    if int(flat.numel()) != 2:
        raise ValueError("encode_inputs expects a tensor with exactly 2 values.")

    mode_norm = mode.lower().strip()
    if mode_norm not in {"rate", "spike"}:
        raise ValueError("mode must be 'rate' or 'spike'")
    if dt <= 0.0:
        raise ValueError("dt must be > 0")

    bits = (flat >= 0.5).to(dtype=torch.long)
    drive = torch.full((4,), float(low), dtype=flat.dtype, device=flat.device)

    # bit0 selects neuron 0 or 1; bit1 selects neuron 2 or 3.
    offsets = bits.new_tensor([0, 2])
    high_indices = offsets + bits
    drive.index_fill_(0, high_indices, float(high))

    if mode_norm == "spike":
        # Convert to impulse-like magnitude per step (unit-compatible with dt).
        drive = drive / float(dt)

    return {compartment: drive}


def decode_output(
    spike_counts: Any,
    mode: OutputDecodeMode = "wta",
    *,
    threshold: float = 0.5,
    hysteresis: float = 0.0,
    last_pred: int | None = None,
) -> int:
    """Decode output spikes into a binary prediction."""

    counts = spike_counts.reshape(-1)
    mode_norm = mode.lower().strip()
    if mode_norm == "wta":
        return _decode_wta(
            counts=counts,
            hysteresis=float(hysteresis),
            last_pred=last_pred,
        )
    if mode_norm == "threshold":
        return _decode_threshold(
            counts=counts,
            threshold=float(threshold),
            hysteresis=float(hysteresis),
            last_pred=last_pred,
        )
    raise ValueError("mode must be 'wta' or 'threshold'")


def _decode_wta(*, counts: Any, hysteresis: float, last_pred: int | None) -> int:
    if int(counts.numel()) < 2:
        raise ValueError("WTA decode requires at least 2 output neurons.")
    diff = float((counts[1] - counts[0]).item())
    if last_pred is not None and abs(diff) <= hysteresis:
        return int(last_pred)
    if diff > 0.0:
        return 1
    if diff < 0.0:
        return 0
    return int(last_pred) if last_pred is not None else 0


def _decode_threshold(
    *,
    counts: Any,
    threshold: float,
    hysteresis: float,
    last_pred: int | None,
) -> int:
    if int(counts.numel()) == 0:
        raise ValueError("threshold decode requires at least 1 output neuron.")
    if int(counts.numel()) == 1:
        score = float(counts[0].item())
    else:
        denom = float((counts[0] + counts[1]).item())
        score = float(counts[1].item()) / max(denom, 1e-12)

    pred = 1 if score >= threshold else 0
    if last_pred is not None and abs(score - threshold) <= hysteresis:
        return int(last_pred)
    return pred


__all__ = [
    "INPUT_NEURON_INDICES",
    "InputEncodeMode",
    "OUTPUT_NEURON_INDICES",
    "OutputDecodeMode",
    "decode_output",
    "encode_inputs",
]

