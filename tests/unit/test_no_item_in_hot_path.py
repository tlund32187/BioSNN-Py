from __future__ import annotations

import pytest
pytestmark = pytest.mark.unit


from biosnn.biophysics.models.glif import GLIFModel
from biosnn.contracts.simulation import SimulationConfig
from biosnn.simulation.engine import TorchNetworkEngine
from biosnn.simulation.network import PopulationSpec

torch = pytest.importorskip("torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_no_item_in_hot_path_cuda(monkeypatch):
    def _fail(self, *args, **kwargs):
        raise RuntimeError(
            ".item() in hot path causes CUDA sync; compute stats at stride or lazily."
        )

    monkeypatch.setattr(torch.Tensor, "item", _fail, raising=False)

    pop = PopulationSpec(name="Pop", model=GLIFModel(), n=32)
    engine = TorchNetworkEngine(populations=[pop], projections=[], fast_mode=True)
    engine.reset(config=SimulationConfig(dt=1e-3, device="cuda"))

    for _ in range(5):
        engine.step()
