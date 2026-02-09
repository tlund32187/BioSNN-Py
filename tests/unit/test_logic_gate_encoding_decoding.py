from __future__ import annotations

import pytest

from biosnn.contracts.neurons import Compartment
from biosnn.tasks.logic_gates.encoding import decode_output, encode_inputs

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


def test_encode_inputs_per_bit_one_hot_drives_expected_neurons() -> None:
    inputs = torch.tensor([0.0, 1.0], dtype=torch.float32)
    drive = encode_inputs(inputs, mode="rate", dt=1e-3, high=2.0, low=0.1)

    soma = drive[Compartment.SOMA]
    assert soma.shape == (4,)
    expected = torch.tensor([2.0, 0.1, 0.1, 2.0], dtype=torch.float32)
    assert torch.allclose(soma, expected)


def test_decode_output_wta_tie_and_hysteresis_are_deterministic() -> None:
    tied = torch.tensor([5.0, 5.0], dtype=torch.float32)
    assert decode_output(tied, mode="wta") == 0
    assert decode_output(tied, mode="wta", last_pred=1) == 1

    near_tie = torch.tensor([10.0, 10.4], dtype=torch.float32)
    assert decode_output(near_tie, mode="wta", hysteresis=0.5, last_pred=0) == 0
    assert decode_output(near_tie, mode="wta", hysteresis=0.1, last_pred=0) == 1


def test_decode_output_threshold_mode_supports_hysteresis() -> None:
    counts = torch.tensor([9.0, 11.0], dtype=torch.float32)
    assert decode_output(counts, mode="threshold", threshold=0.5) == 1
    assert decode_output(counts, mode="threshold", threshold=0.55, hysteresis=0.1, last_pred=0) == 0

