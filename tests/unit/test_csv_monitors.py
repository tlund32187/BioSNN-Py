import csv
from pathlib import Path
from typing import cast

import pytest

from biosnn.contracts.monitors import StepEvent
from biosnn.contracts.tensor import Tensor
from biosnn.monitors.csv import (
    AdEx2CompCSVMonitor,
    GLIFCSVMonitor,
    NeuronCSVMonitor,
    SynapseCSVMonitor,
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_tensor(values: list[int] | list[float]) -> Tensor:
    return cast(Tensor, values)


def test_neuron_csv_monitor_writes_stats(artifact_dir: Path) -> None:
    path = artifact_dir / "neurons.csv"
    monitor = NeuronCSVMonitor(path)

    event = StepEvent(
        t=0.1,
        dt=0.01,
        spikes=_as_tensor([0, 1, 1]),
        tensors={"v": _as_tensor([1.0, 2.0, 3.0]), "w": _as_tensor([0.0, 0.5, 1.0])},
        scalars={"loss": 1.25},
    )
    monitor.on_step(event)
    monitor.close()

    rows = _read_rows(path)
    assert len(rows) == 1
    row = rows[0]

    assert float(row["t"]) == pytest.approx(0.1)
    assert float(row["dt"]) == pytest.approx(0.01)
    assert float(row["spike_count"]) == pytest.approx(2.0)
    assert float(row["spike_fraction"]) == pytest.approx(2.0 / 3.0)
    assert float(row["spike_rate_hz"]) == pytest.approx((2.0 / 3.0) / 0.01)
    assert float(row["loss"]) == pytest.approx(1.25)
    assert float(row["v_mean"]) == pytest.approx(2.0)
    assert float(row["v_min"]) == pytest.approx(1.0)
    assert float(row["v_max"]) == pytest.approx(3.0)


def test_glif_csv_monitor_preset(artifact_dir: Path) -> None:
    path = artifact_dir / "glif.csv"
    monitor = GLIFCSVMonitor(path, sample_indices=[1], stats=("mean",))

    event = StepEvent(
        t=0.0,
        dt=0.001,
        spikes=_as_tensor([0, 1, 0]),
        tensors={
            "v_soma": _as_tensor([-0.07, -0.05, -0.06]),
            "refrac_left": _as_tensor([0.0, 0.0, 0.002]),
            "spike_hold_left": _as_tensor([0.0, 0.001, 0.0]),
            "theta": _as_tensor([0.0, 0.003, 0.0]),
        },
    )
    monitor.on_step(event)
    monitor.close()

    row = _read_rows(path)[0]
    assert "v_soma_mean" in row
    assert float(row["v_soma_i1"]) == pytest.approx(-0.05)
    assert float(row["theta_i1"]) == pytest.approx(0.003)


def test_adex_csv_monitor_preset(artifact_dir: Path) -> None:
    path = artifact_dir / "adex.csv"
    monitor = AdEx2CompCSVMonitor(path, stats=("mean",))

    event = StepEvent(
        t=0.2,
        dt=0.001,
        spikes=_as_tensor([1, 0]),
        tensors={
            "v_soma": _as_tensor([-0.06, -0.07]),
            "v_dend": _as_tensor([-0.05, -0.06]),
            "w": _as_tensor([0.0, 0.1]),
            "refrac_left": _as_tensor([0.0, 0.0]),
            "spike_hold_left": _as_tensor([0.001, 0.0]),
        },
    )
    monitor.on_step(event)
    monitor.close()

    row = _read_rows(path)[0]
    assert float(row["v_soma_mean"]) == pytest.approx(-0.065)
    assert float(row["v_dend_mean"]) == pytest.approx(-0.055)
    assert float(row["w_mean"]) == pytest.approx(0.05)


def test_synapse_csv_monitor_preset(artifact_dir: Path) -> None:
    path = artifact_dir / "synapses.csv"
    monitor = SynapseCSVMonitor(path, stats=("mean",))

    event = StepEvent(
        t=0.3,
        dt=0.001,
        tensors={"weights": _as_tensor([0.5, 1.5, 2.0])},
        scalars={"edge_count": 3.0},
    )
    monitor.on_step(event)
    monitor.close()

    row = _read_rows(path)[0]
    assert float(row["weights_mean"]) == pytest.approx(1.3333333333)
    assert float(row["edge_count"]) == pytest.approx(3.0)


def test_neuron_csv_monitor_async_gpu_cuda(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from dataclasses import dataclass

    from biosnn.contracts.neurons import (
        Compartment,
        INeuronModel,
        NeuronInputs,
        NeuronStepResult,
        StepContext,
    )
    from biosnn.contracts.simulation import SimulationConfig
    from biosnn.simulation.engine import TorchNetworkEngine
    from biosnn.simulation.network import PopulationSpec

    @dataclass(slots=True)
    class DummyState:
        v: Tensor

    class DummyNeuron(INeuronModel):
        name = "dummy"
        compartments = frozenset({Compartment.SOMA})

        def init_state(self, n: int, *, ctx: StepContext) -> DummyState:
            return DummyState(v=torch.zeros((n,), device=ctx.device, dtype=torch.float32))

        def reset_state(
            self,
            state: DummyState,
            *,
            ctx: StepContext,
            indices: Tensor | None = None,
        ) -> DummyState:
            _ = (ctx, indices)
            state.v.zero_()
            return state

        def step(
            self,
            state: DummyState,
            inputs: NeuronInputs,
            *,
            dt: float,
            t: float,
            ctx: StepContext,
        ) -> tuple[DummyState, NeuronStepResult]:
            _ = (inputs, dt, t, ctx)
            spikes = torch.zeros_like(state.v, dtype=torch.bool)
            return state, NeuronStepResult(spikes=spikes)

        def state_tensors(self, state: DummyState):
            return {"v": state.v}

    def _no_item(self, *args, **kwargs):
        raise RuntimeError(".item() in hot path")

    monkeypatch.setattr(torch.Tensor, "item", _no_item, raising=False)

    pop = PopulationSpec(name="Pop", model=DummyNeuron(), n=8)
    engine = TorchNetworkEngine(populations=[pop], projections=[], fast_mode=False)

    out_path = tmp_path / "async_gpu.csv"
    monitor = NeuronCSVMonitor(
        out_path,
        tensor_keys=("v",),
        include_spikes=True,
        include_scalars=True,
        flush_every=10,
        every_n_steps=1,
        async_gpu=True,
    )
    engine.attach_monitors([monitor])
    engine.reset(config=SimulationConfig(dt=1e-3, device="cuda"))
    for _ in range(3):
        engine.step()
    monitor.close()

    rows = _read_rows(out_path)
    assert len(rows) == 3
    assert "v_mean" in rows[0]
