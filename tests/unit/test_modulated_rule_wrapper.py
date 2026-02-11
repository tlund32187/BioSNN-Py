from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from biosnn.contracts.learning import LearningBatch, LearningStepResult
from biosnn.contracts.modulators import ModulatorKind
from biosnn.contracts.neurons import StepContext
from biosnn.learning.rules import ModulatedRuleWrapper, ModulatedRuleWrapperParams

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


@dataclass(slots=True)
class _DummyState:
    marker: Any


class _ConstantDwRule:
    name = "constant_dw_rule"
    supports_sparse = True

    def __init__(self, value: float) -> None:
        self._value = float(value)

    def init_state(self, e: int, *, ctx: Any) -> _DummyState:
        _ = (e, ctx)
        return _DummyState(marker=None)

    def step(
        self,
        state: _DummyState,
        batch: LearningBatch,
        *,
        dt: float,
        t: float,
        ctx: Any,
    ) -> tuple[_DummyState, LearningStepResult]:
        _ = (dt, t, ctx)
        d_weights = torch.full_like(batch.weights, self._value)
        extras = dict(batch.extras) if batch.extras is not None else {}
        return state, LearningStepResult(d_weights=d_weights, extras=extras)

    def state_tensors(self, state: _DummyState) -> dict[str, Any]:
        _ = state
        return {}


def test_wrapper_scales_dw_with_ach() -> None:
    wrapper = ModulatedRuleWrapper(
        inner=_ConstantDwRule(1.0),
        params=ModulatedRuleWrapperParams(
            ach_lr_gain=0.5,
            combine_mode="exp",
            lr_clip_min=0.1,
            lr_clip_max=10.0,
        ),
    )
    state = wrapper.init_state(3, ctx=StepContext(device="cpu", dtype="float32"))
    weights = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float32)
    ach = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)
    batch = LearningBatch(
        pre_spikes=torch.ones((3,), dtype=torch.float32),
        post_spikes=torch.ones((3,), dtype=torch.float32),
        weights=weights,
        modulators={ModulatorKind.ACETYLCHOLINE: ach},
    )

    _, result = wrapper.step(state, batch, dt=1e-3, t=0.0, ctx=StepContext(device="cpu", dtype="float32"))
    assert torch.all(result.d_weights > 1.0)


def test_wrapper_adds_serotonin_decay() -> None:
    wrapper = ModulatedRuleWrapper(
        inner=_ConstantDwRule(0.0),
        params=ModulatedRuleWrapperParams(
            ht_extra_weight_decay=0.5,
            ht_baseline=0.2,
        ),
    )
    state = wrapper.init_state(2, ctx=StepContext(device="cpu", dtype="float32"))
    weights = torch.tensor([2.0, -3.0], dtype=torch.float32)
    ht = torch.tensor([1.2, 1.2], dtype=torch.float32)
    batch = LearningBatch(
        pre_spikes=torch.ones((2,), dtype=torch.float32),
        post_spikes=torch.ones((2,), dtype=torch.float32),
        weights=weights,
        modulators={ModulatorKind.SEROTONIN: ht},
    )

    _, result = wrapper.step(state, batch, dt=1e-3, t=0.0, ctx=StepContext(device="cpu", dtype="float32"))
    expected = torch.tensor([-1.0, 1.5], dtype=torch.float32)
    torch.testing.assert_close(result.d_weights, expected)


def test_wrapper_preserves_active_edges_extras() -> None:
    wrapper = ModulatedRuleWrapper(
        inner=_ConstantDwRule(1.0),
        params=ModulatedRuleWrapperParams(),
    )
    state = wrapper.init_state(2, ctx=StepContext(device="cpu", dtype="float32"))
    active_edges = torch.tensor([3, 7], dtype=torch.long)
    batch = LearningBatch(
        pre_spikes=torch.ones((2,), dtype=torch.float32),
        post_spikes=torch.ones((2,), dtype=torch.float32),
        weights=torch.tensor([0.4, 0.6], dtype=torch.float32),
        modulators={ModulatorKind.ACETYLCHOLINE: torch.zeros((2,), dtype=torch.float32)},
        extras={"active_edges": active_edges},
    )

    _, result = wrapper.step(state, batch, dt=1e-3, t=0.0, ctx=StepContext(device="cpu", dtype="float32"))
    assert result.extras is not None
    assert "active_edges" in result.extras
    torch.testing.assert_close(result.extras["active_edges"], active_edges)

