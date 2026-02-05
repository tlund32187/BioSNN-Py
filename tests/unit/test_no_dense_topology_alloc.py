from __future__ import annotations

import pytest
pytestmark = pytest.mark.unit


from biosnn.connectivity.builders import build_erdos_renyi_edges

torch = pytest.importorskip("torch")


def test_no_dense_mask_allocation(monkeypatch):
    original = torch.rand

    def guarded_rand(*size, **kwargs):
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            shape = tuple(size[0])
        else:
            shape = tuple(size)
        if len(shape) == 2:
            numel = int(shape[0]) * int(shape[1])
            if numel > 2_000_000:
                raise RuntimeError("dense rand allocation detected in topology builder")
        return original(*size, **kwargs)

    monkeypatch.setattr(torch, "rand", guarded_rand)

    n_pre = 20_000
    n_post = 20_000
    p = 1e-6
    pre_idx, post_idx = build_erdos_renyi_edges(n_pre=n_pre, n_post=n_post, p=p, allow_self=True)

    assert pre_idx.numel() == int(p * n_pre * n_post)
    assert pre_idx.shape == post_idx.shape
