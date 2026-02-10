from __future__ import annotations

import pytest

from biosnn.contracts.modulators import ModulatorKind, ModulatorRelease
from biosnn.contracts.neurons import StepContext
from biosnn.neuromodulators import GridDiffusion2DField, GridDiffusion2DParams

pytestmark = pytest.mark.acceptance

torch = pytest.importorskip("torch")


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
    ],
)
def test_grid_modulator_center_decays_without_diffusion(device: str) -> None:
    field = GridDiffusion2DField(
        params=GridDiffusion2DParams(
            kinds=(ModulatorKind.DOPAMINE,),
            grid_size=(16, 16),
            world_origin=(-1.0, -1.0),
            world_extent=(2.0, 2.0),
            diffusion=0.0,
            decay_tau=1.0,
            deposit_sigma=0.0,
            clamp_min=0.0,
        )
    )
    ctx = StepContext(device=device, dtype="float32")
    state = field.init_state(ctx=ctx)

    center_pos = torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    release = ModulatorRelease(
        kind=ModulatorKind.DOPAMINE,
        positions=center_pos,
        amount=torch.tensor([1.0], device=device, dtype=torch.float32),
    )
    state = field.step(state, releases=[release], dt=0.1, t=0.0, ctx=ctx)
    center_before = float(
        field.sample_at(state, positions=center_pos, kind=ModulatorKind.DOPAMINE, ctx=ctx)[0].item()
    )

    for step in range(1, 7):
        state = field.step(state, releases=[], dt=0.1, t=float(step) * 0.1, ctx=ctx)
    center_after = float(
        field.sample_at(state, positions=center_pos, kind=ModulatorKind.DOPAMINE, ctx=ctx)[0].item()
    )

    assert center_before > 0.0
    assert center_after < center_before


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
    ],
)
def test_grid_modulator_diffuses_to_nearby_positions(device: str) -> None:
    field = GridDiffusion2DField(
        params=GridDiffusion2DParams(
            kinds=(ModulatorKind.DOPAMINE,),
            grid_size=(16, 16),
            world_origin=(-1.0, -1.0),
            world_extent=(2.0, 2.0),
            diffusion=1.0,
            decay_tau=1_000.0,
            deposit_sigma=0.0,
            clamp_min=0.0,
        )
    )
    ctx = StepContext(device=device, dtype="float32")
    state = field.init_state(ctx=ctx)

    center_pos = torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    near_pos = torch.tensor([[0.25, 0.0, 0.0]], device=device, dtype=torch.float32)
    release = ModulatorRelease(
        kind=ModulatorKind.DOPAMINE,
        positions=center_pos,
        amount=torch.tensor([1.0], device=device, dtype=torch.float32),
    )

    # dt=0 deposits release into the grid without diffusion/decay.
    state = field.step(state, releases=[release], dt=0.0, t=0.0, ctx=ctx)
    near_before = float(
        field.sample_at(state, positions=near_pos, kind=ModulatorKind.DOPAMINE, ctx=ctx)[0].item()
    )
    center_before = float(
        field.sample_at(state, positions=center_pos, kind=ModulatorKind.DOPAMINE, ctx=ctx)[0].item()
    )

    state = field.step(state, releases=[], dt=0.1, t=0.1, ctx=ctx)
    near_after = float(
        field.sample_at(state, positions=near_pos, kind=ModulatorKind.DOPAMINE, ctx=ctx)[0].item()
    )
    center_after = float(
        field.sample_at(state, positions=center_pos, kind=ModulatorKind.DOPAMINE, ctx=ctx)[0].item()
    )

    assert near_after > near_before
    assert center_after < center_before

