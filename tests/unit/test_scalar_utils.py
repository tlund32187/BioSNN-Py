from __future__ import annotations

import pytest
pytestmark = pytest.mark.unit


from biosnn.contracts.monitors import StepEvent
from biosnn.monitors.metrics.scalar_utils import scalar_to_float

torch = pytest.importorskip("torch")


def test_step_event_accepts_tensor_scalars():
    event = StepEvent(t=0.0, dt=1e-3, scalars={"loss": torch.tensor(1.25)})
    assert event.scalars is not None
    value = event.scalars["loss"]
    assert isinstance(value, torch.Tensor)
    assert value.shape == torch.Size([])


def test_scalar_to_float_cpu_tensor():
    value = scalar_to_float(torch.tensor(2.5))
    assert value == pytest.approx(2.5)


def test_scalar_to_float_cuda_tensor():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    value = scalar_to_float(torch.tensor(3.5, device="cuda"))
    assert value == pytest.approx(3.5)
