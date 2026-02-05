from __future__ import annotations

import pytest

from biosnn.biophysics.models.glif import GLIFModel
from biosnn.connectivity.builders import build_erdos_renyi_topology
from biosnn.contracts.simulation import SimulationConfig
from biosnn.simulation.engine import TorchNetworkEngine
from biosnn.simulation.network import PopulationSpec, ProjectionSpec
from biosnn.synapses.dynamics.delayed_current import DelayedCurrentParams, DelayedCurrentSynapse

torch = pytest.importorskip("torch")


def _build_engine(*, fast_mode: bool):
    n = 32
    pop = PopulationSpec(name="Pop", model=GLIFModel(), n=n)
    topology = build_erdos_renyi_topology(n=n, p=0.2, allow_self=False, dt=1e-3)
    synapse = DelayedCurrentSynapse(DelayedCurrentParams(init_weight=0.05))
    proj = ProjectionSpec(name="Pop->Pop", synapse=synapse, topology=topology, pre="Pop", post="Pop")
    return TorchNetworkEngine(populations=[pop], projections=[proj], fast_mode=fast_mode)


def test_engine_reuses_drive_buffers_fast_mode():
    engine = _build_engine(fast_mode=True)
    engine.reset(config=SimulationConfig(dt=1e-3, device="cpu"))

    drive_map = engine._drive_buffers["Pop"]
    before_ids = {comp: id(tensor) for comp, tensor in drive_map.items()}
    spikes_id = id(engine._pop_states["Pop"].spikes)

    for _ in range(50):
        engine.step()
        drive_map_step = engine._drive_buffers["Pop"]
        after_ids = {comp: id(tensor) for comp, tensor in drive_map_step.items()}
        assert before_ids == after_ids
        assert id(engine._pop_states["Pop"].spikes) == spikes_id


def test_compiled_mode_reuses_global_buffers():
    engine = TorchNetworkEngine(
        populations=[PopulationSpec(name="Pop", model=GLIFModel(), n=32)],
        projections=[],
        compiled_mode=True,
    )
    engine.reset(config=SimulationConfig(dt=1e-3, device="cpu"))

    assert engine._spikes_global is not None
    spikes_id = id(engine._spikes_global)
    drive_ids = {comp: id(tensor) for comp, tensor in engine._drive_global.items()}

    for _ in range(10):
        engine.step()
        assert id(engine._spikes_global) == spikes_id
        assert {comp: id(tensor) for comp, tensor in engine._drive_global.items()} == drive_ids


def test_projection_runtime_list_built():
    engine = _build_engine(fast_mode=False)
    engine.reset(config=SimulationConfig(dt=1e-3, device="cpu"))

    assert hasattr(engine, "_proj_runtime_list")
    assert len(engine._proj_runtime_list) == 1
    assert engine._proj_runtime_list[0].name == "Pop->Pop"
