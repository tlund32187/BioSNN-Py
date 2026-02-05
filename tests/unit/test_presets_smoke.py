import pytest

from biosnn.api import presets
from biosnn.contracts.simulation import SimulationConfig
from biosnn.simulation.engine import TorchNetworkEngine

pytestmark = pytest.mark.unit

torch = pytest.importorskip("torch")


def _run_network(net, *, steps: int = 3) -> None:
    engine = TorchNetworkEngine(
        populations=net.populations,
        projections=net.projections,
        modulators=net.modulators,
        fast_mode=True,
    )
    engine.reset(config=SimulationConfig(dt=1e-3, device="cpu"))
    engine.run(steps=steps)


def test_presets_two_pop_demo_cpu() -> None:
    net = presets.make_two_pop_delayed_demo(device="cpu", seed=1)
    _run_network(net)


def test_presets_logic_gate_cpu() -> None:
    net = presets.make_minimal_logic_gate_network("xor", device="cpu", seed=7)
    _run_network(net)


def test_default_engine_config() -> None:
    cfg = presets.default_engine_config()
    assert cfg.fast_mode is True
    assert cfg.compiled_mode is False
    assert cfg.monitors_enabled is False


def test_presets_cuda_optional() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    net = presets.make_two_pop_delayed_demo(device="cuda", seed=2)
    engine = TorchNetworkEngine(
        populations=net.populations,
        projections=net.projections,
        modulators=net.modulators,
        fast_mode=True,
    )
    engine.reset(config=SimulationConfig(dt=1e-3, device="cuda"))
    engine.run(steps=2)
