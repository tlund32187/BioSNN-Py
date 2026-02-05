from __future__ import annotations

import pytest

from biosnn.contracts.modulators import ModulatorKind, ModulatorRelease
from biosnn.neuromodulators import GlobalScalarField, GlobalScalarParams

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


def test_global_scalar_field_decay_and_release():
    field = GlobalScalarField(kinds=(ModulatorKind.DOPAMINE,), params=GlobalScalarParams(decay_tau=1.0))
    state = field.init_state(ctx=None)

    release = ModulatorRelease(
        kind=ModulatorKind.DOPAMINE,
        positions=torch.zeros((1, 3), dtype=torch.float32),
        amount=torch.tensor([1.0], dtype=torch.float32),
    )
    state = field.step(state, releases=[release], dt=1.0, t=0.0, ctx=None)
    assert float(state.levels[0].item()) > 0.0

    prev = float(state.levels[0].item())
    state = field.step(state, releases=[], dt=1.0, t=1.0, ctx=None)
    assert float(state.levels[0].item()) < prev


def test_global_scalar_field_sample_shape():
    field = GlobalScalarField(kinds=(ModulatorKind.DOPAMINE,))
    state = field.init_state(ctx=None)
    positions = torch.zeros((4, 3), dtype=torch.float32)
    sample = field.sample_at(state, positions=positions, kind=ModulatorKind.DOPAMINE, ctx=None)
    assert sample.shape == (4,)
