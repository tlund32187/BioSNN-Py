from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from biosnn.contracts.modulators import ModulatorKind
from biosnn.contracts.simulation import ExcitabilityModulationConfig, SimulationConfig
from biosnn.core.torch_utils import resolve_device_dtype
from biosnn.simulation.engine import TorchNetworkEngine
from biosnn.simulation.network import PopulationSpec
from tests.support.test_models import SpikeInputModel

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("compiled_mode", [False, True])
def test_excitability_modulation_bias_changes_drive_and_spikes(
    monkeypatch: pytest.MonkeyPatch,
    compiled_mode: bool,
) -> None:
    pop = PopulationSpec(name="Hidden", model=SpikeInputModel(), n=3, meta={"role": "hidden"})
    engine = TorchNetworkEngine(
        populations=[pop],
        projections=[],
        compiled_mode=compiled_mode,
        fast_mode=compiled_mode,
    )
    cfg = SimulationConfig(
        dt=1e-3,
        device="cpu",
        dtype="float32",
        excitability_modulation=ExcitabilityModulationConfig(
            enabled=True,
            ach_gain=1.0,
            ne_gain=0.5,
            ht_gain=0.5,
            clamp_abs=0.25,
        ),
    )
    engine.reset(config=cfg)

    levels = {
        ModulatorKind.ACETYLCHOLINE: torch.tensor([0.4, 0.1, 0.1], dtype=torch.float32),
        ModulatorKind.NORADRENALINE: torch.tensor([0.2, 0.0, 0.0], dtype=torch.float32),
        ModulatorKind.SEROTONIN: torch.tensor([0.0, 0.2, 1.0], dtype=torch.float32),
    }
    if compiled_mode:
        engine._compiled_mod_by_pop["Hidden"] = dict(levels)
    else:

        def fake_step(**kwargs: Any):
            _ = kwargs
            return {"Hidden": dict(levels)}, {}

        monkeypatch.setattr(engine._modulator_subsystem, "step", fake_step)

    engine.step()

    state = engine._pop_states["Hidden"].state
    expected_drive = torch.tensor([0.25, 0.0, -0.25], dtype=torch.float32)
    torch.testing.assert_close(state.last_drive_soma, expected_drive)
    torch.testing.assert_close(
        engine._pop_states["Hidden"].spikes,
        torch.tensor([True, False, False]),
    )


def test_excitability_modulation_missing_modulators_is_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pop = PopulationSpec(name="Hidden", model=SpikeInputModel(), n=2, meta={"role": "hidden"})
    engine = TorchNetworkEngine(
        populations=[pop],
        projections=[],
        compiled_mode=False,
    )
    cfg = SimulationConfig(
        dt=1e-3,
        device="cpu",
        dtype="float32",
        excitability_modulation=ExcitabilityModulationConfig(
            enabled=True,
            ach_gain=1.0,
            ne_gain=1.0,
            ht_gain=1.0,
            clamp_abs=1.0,
        ),
    )
    engine.reset(config=cfg)

    def fake_step(**kwargs: Any):
        _ = kwargs
        return {}, {}

    monkeypatch.setattr(engine._modulator_subsystem, "step", fake_step)

    engine.step()
    state = engine._pop_states["Hidden"].state
    torch.testing.assert_close(state.last_drive_soma, torch.zeros((2,), dtype=torch.float32))
    torch.testing.assert_close(engine._pop_states["Hidden"].spikes, torch.zeros((2,), dtype=torch.bool))


def test_excitability_modulation_uses_selected_targets_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pop_hidden = PopulationSpec(name="Hidden", model=SpikeInputModel(), n=2, meta={"role": "hidden"})
    pop_input = PopulationSpec(name="Input", model=SpikeInputModel(), n=2, meta={"role": "input"})
    engine = TorchNetworkEngine(
        populations=[pop_hidden, pop_input],
        projections=[],
        compiled_mode=False,
    )
    cfg = SimulationConfig(
        dt=1e-3,
        device="cpu",
        dtype="float32",
        excitability_modulation={
            "enabled": True,
            "targets": ["hidden"],
            "compartment": "soma",
            "ach_gain": 1.0,
            "clamp_abs": 1.0,
        },
    )
    engine.reset(config=cfg)
    device, dtype = resolve_device_dtype(engine._ctx)
    levels = torch.tensor([0.3, 0.4], device=device, dtype=dtype)

    def fake_step(**kwargs: Any) -> tuple[dict[str, Mapping[ModulatorKind, Any]], dict[str, Any]]:
        _ = kwargs
        return {
            "Hidden": {ModulatorKind.ACETYLCHOLINE: levels},
            "Input": {ModulatorKind.ACETYLCHOLINE: levels},
        }, {}

    monkeypatch.setattr(engine._modulator_subsystem, "step", fake_step)

    engine.step()

    hidden_state = engine._pop_states["Hidden"].state
    input_state = engine._pop_states["Input"].state
    torch.testing.assert_close(hidden_state.last_drive_soma, levels)
    torch.testing.assert_close(input_state.last_drive_soma, torch.zeros_like(levels))
