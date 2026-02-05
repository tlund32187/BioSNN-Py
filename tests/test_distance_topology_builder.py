from __future__ import annotations

from typing import cast

import pytest

from biosnn.connectivity.builders import build_bipartite_distance_topology
from biosnn.contracts.tensor import Tensor

torch = pytest.importorskip("torch")


def _line_positions(count: int) -> Tensor:
    xs = torch.linspace(0.0, 1.0, count)
    zeros = torch.zeros_like(xs)
    return cast(Tensor, torch.stack([xs, zeros, zeros], dim=1))


def test_distance_builder_locality():
    torch.manual_seed(0)
    n_pre = 40
    n_post = 32
    pre_pos = _line_positions(n_pre)
    post_pos = _line_positions(n_post)

    topo = build_bipartite_distance_topology(
        pre_positions=pre_pos,
        post_positions=post_pos,
        p0=0.6,
        sigma=0.2,
        seed=1,
    )

    assert topo.pre_idx.numel() > 0
    assert topo.pre_idx.shape == topo.post_idx.shape

    edge_dist = torch.abs(pre_pos[topo.pre_idx, 0] - post_pos[topo.post_idx, 0])
    rand_pre = torch.randint(0, n_pre, (edge_dist.numel(),))
    rand_post = torch.randint(0, n_post, (edge_dist.numel(),))
    rand_dist = torch.abs(pre_pos[rand_pre, 0] - post_pos[rand_post, 0])

    assert edge_dist.mean() < rand_dist.mean()


def test_distance_builder_delays_correlate():
    torch.manual_seed(1)
    n_pre = 30
    n_post = 30
    pre_pos = _line_positions(n_pre)
    post_pos = _line_positions(n_post)

    topo = build_bipartite_distance_topology(
        pre_positions=pre_pos,
        post_positions=post_pos,
        p0=0.8,
        sigma=0.25,
        seed=2,
        delay_from_distance=True,
        delay_base_steps=1,
        delay_per_unit_steps=10.0,
        delay_max_steps=20,
    )

    assert topo.delay_steps is not None
    assert topo.delay_steps.dtype == torch.int32
    assert torch.all(topo.delay_steps >= 0)

    dist = torch.abs(pre_pos[topo.pre_idx, 0] - post_pos[topo.post_idx, 0])
    order = torch.argsort(dist)
    mid = max(1, dist.numel() // 2)
    low = topo.delay_steps[order[:mid]].float().mean()
    high = topo.delay_steps[order[mid:]].float().mean()

    assert high >= low
