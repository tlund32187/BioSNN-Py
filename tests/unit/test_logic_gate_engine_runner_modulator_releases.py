from __future__ import annotations

import pytest

from biosnn.contracts.modulators import ModulatorKind
from biosnn.tasks.logic_gates.engine_runner import (
    _ACH_INPUT_PULSE_SCALE,
    _NA_SURPRISE_PULSE_SCALE,
    _SEROTONIN_CORRECT_PULSE_SCALE,
    _build_modulator_releases_for_step,
    _queue_trial_feedback_releases,
)

pytestmark = pytest.mark.unit


def test_queue_trial_feedback_releases_routes_kinds_by_outcome() -> None:
    pending = {
        ModulatorKind.DOPAMINE: 0.0,
        ModulatorKind.NORADRENALINE: 0.0,
        ModulatorKind.SEROTONIN: 0.0,
    }
    dopamine = _queue_trial_feedback_releases(
        pending_releases=pending,
        modulator_kinds=tuple(pending.keys()),
        modulator_amount=2.0,
        correct=False,
    )
    assert dopamine == pytest.approx(-2.0)
    assert pending[ModulatorKind.DOPAMINE] == pytest.approx(-2.0)
    assert pending[ModulatorKind.NORADRENALINE] == pytest.approx(2.0 * _NA_SURPRISE_PULSE_SCALE)
    assert pending[ModulatorKind.SEROTONIN] == pytest.approx(0.0)

    pending = {
        ModulatorKind.DOPAMINE: 0.0,
        ModulatorKind.NORADRENALINE: 0.0,
        ModulatorKind.SEROTONIN: 0.0,
    }
    dopamine = _queue_trial_feedback_releases(
        pending_releases=pending,
        modulator_kinds=tuple(pending.keys()),
        modulator_amount=2.0,
        correct=True,
    )
    assert dopamine == pytest.approx(2.0)
    assert pending[ModulatorKind.DOPAMINE] == pytest.approx(2.0)
    assert pending[ModulatorKind.NORADRENALINE] == pytest.approx(0.0)
    assert pending[ModulatorKind.SEROTONIN] == pytest.approx(2.0 * _SEROTONIN_CORRECT_PULSE_SCALE)


def test_build_modulator_releases_grid_uses_output_positions_and_kind_filter() -> None:
    torch = pytest.importorskip("torch")
    output_positions = torch.tensor(
        [[0.2, 0.25, 0.0], [0.8, 0.75, 0.0]],
        device="cpu",
        dtype=torch.float32,
    )
    pending = {
        ModulatorKind.ACETYLCHOLINE: 0.0,
        ModulatorKind.SEROTONIN: 0.0,
    }
    releases, ach_emitted = _build_modulator_releases_for_step(
        pending_releases=pending,
        modulator_kinds=(ModulatorKind.ACETYLCHOLINE, ModulatorKind.SEROTONIN),
        modulator_amount=1.5,
        input_active=True,
        ach_pulse_emitted=False,
        field_type="grid_diffusion_2d",
        world_extent=(1.0, 1.0),
        output_positions=output_positions,
        dopamine_focus_action=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert ach_emitted is True
    assert len(releases) == 2
    assert all(release.positions.shape == output_positions.shape for release in releases)
    assert all(torch.allclose(release.positions, output_positions) for release in releases)
    by_kind = {release.kind: float(release.amount.item()) for release in releases}
    assert by_kind[ModulatorKind.ACETYLCHOLINE] == pytest.approx(1.5 * _ACH_INPUT_PULSE_SCALE)
    assert by_kind[ModulatorKind.SEROTONIN] > 0.0
    assert pending[ModulatorKind.ACETYLCHOLINE] == pytest.approx(0.0)
    assert pending[ModulatorKind.SEROTONIN] == pytest.approx(0.0)


def test_build_modulator_releases_global_uses_dummy_center_position() -> None:
    torch = pytest.importorskip("torch")
    pending = {ModulatorKind.DOPAMINE: 0.75}
    releases, ach_emitted = _build_modulator_releases_for_step(
        pending_releases=pending,
        modulator_kinds=(ModulatorKind.DOPAMINE,),
        modulator_amount=1.0,
        input_active=False,
        ach_pulse_emitted=False,
        field_type="global_scalar",
        world_extent=(2.0, 4.0),
        output_positions=torch.zeros((2, 3), dtype=torch.float32),
        dopamine_focus_action=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert ach_emitted is False
    assert len(releases) == 1
    release = releases[0]
    assert release.kind == ModulatorKind.DOPAMINE
    assert release.positions.shape == (1, 3)
    assert release.positions[0, 0].item() == pytest.approx(1.0)
    assert release.positions[0, 1].item() == pytest.approx(2.0)
    assert release.positions[0, 2].item() == pytest.approx(0.0)
    assert float(release.amount.item()) == pytest.approx(0.75)
