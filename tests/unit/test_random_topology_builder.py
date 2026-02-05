from __future__ import annotations

import pytest

from biosnn.connectivity.builders import build_erdos_renyi_edges, build_erdos_renyi_topology

pytestmark = pytest.mark.unit

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


def test_erdos_renyi_edges_count():
    n_pre = 100
    n_post = 80
    p = 0.1
    expected = int(p * n_pre * n_post)
    pre_idx, post_idx = build_erdos_renyi_edges(n_pre=n_pre, n_post=n_post, p=p, allow_self=True)

    assert pre_idx.numel() == expected
    assert pre_idx.shape == post_idx.shape


def test_erdos_renyi_builder_large_no_dense(monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError("torch.rand should not be used for dense masks")

    monkeypatch.setattr(torch, "rand", _fail)

    n = 20_000
    p = 1e-6
    expected = int(p * n * n)
    topo = build_erdos_renyi_topology(n=n, p=p, allow_self=True)

    assert topo.pre_idx.numel() == expected
    assert topo.post_idx.numel() == expected
