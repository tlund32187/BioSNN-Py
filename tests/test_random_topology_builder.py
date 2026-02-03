from __future__ import annotations

import pytest

from biosnn.connectivity.builders import build_erdos_renyi_topology

torch = pytest.importorskip("torch")


def test_erdos_renyi_builder_no_self():
    torch.manual_seed(0)
    topo = build_erdos_renyi_topology(n=10, p=0.5, allow_self=False)

    assert topo.pre_idx.dtype == torch.int64
    assert topo.post_idx.dtype == torch.int64
    assert topo.pre_idx.shape == topo.post_idx.shape
    assert not torch.any(topo.pre_idx == topo.post_idx)


def test_erdos_renyi_builder_with_delay():
    torch.manual_seed(0)
    positions = torch.zeros((10, 3), dtype=torch.float32)
    topo = build_erdos_renyi_topology(n=10, p=0.3, allow_self=False, positions=positions, dt=1e-3)

    assert topo.delay_steps is not None
    assert topo.delay_steps.dtype == torch.int32
