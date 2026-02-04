"""Random topology builders."""

from __future__ import annotations

from typing import Any

from biosnn.biophysics.models._torch_utils import require_torch
from biosnn.connectivity.delays.axon import compute_delay_steps
from biosnn.contracts.synapses import SynapseTopology
from biosnn.contracts.tensor import Tensor


def build_erdos_renyi_topology(
    n: int,
    p: float,
    *,
    allow_self: bool = False,
    device: str | None = None,
    dtype: str | None = None,
    dt: float | None = None,
    positions: Tensor | None = None,
    myelin: Tensor | None = None,
    weight_init: float | None = None,
    receptor: Tensor | None = None,
    target_compartments: Tensor | None = None,
) -> SynapseTopology:
    """Build a directed Erdos-Renyi topology with optional delays and weights."""

    if n <= 0:
        raise ValueError("n must be positive")
    if p < 0 or p > 1:
        raise ValueError("p must be within [0,1]")

    torch = require_torch()
    device_obj = torch.device(device) if device else None
    dtype_obj = _resolve_dtype(torch, dtype) if dtype else torch.get_default_dtype()

    pre_idx, post_idx = build_erdos_renyi_edges(
        n_pre=n,
        n_post=n,
        p=p,
        device=device,
        allow_self=allow_self,
    )

    weights = None
    if weight_init is not None:
        weights = torch.full((pre_idx.numel(),), weight_init, device=device_obj, dtype=dtype_obj)

    pre_pos = positions
    post_pos = positions
    delay_steps = None
    myelin_t = _to_device(myelin, device_obj, dtype_obj) if myelin is not None else None
    if positions is not None and dt is not None:
        pre_pos = _to_device(positions, device_obj, dtype_obj)
        if pre_pos is None:
            raise RuntimeError("positions must be a tensor when dt is provided")
        post_pos = pre_pos
        delay_steps = compute_delay_steps(
            pre_pos=pre_pos,
            post_pos=post_pos,
            pre_idx=pre_idx,
            post_idx=post_idx,
            dt=dt,
            myelin=myelin_t,
        )

    receptor = _to_device(receptor, device_obj, None) if receptor is not None else None
    target_compartments = (
        _to_device(target_compartments, device_obj, None) if target_compartments is not None else None
    )

    return SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        delay_steps=delay_steps,
        target_compartments=target_compartments,
        receptor=receptor,
        weights=weights,
        pre_pos=pre_pos,
        post_pos=post_pos,
        myelin=myelin_t,
    )


def build_bipartite_erdos_renyi_topology(
    n_pre: int,
    n_post: int,
    p: float,
    *,
    device: str | None = None,
    dtype: str | None = None,
    dt: float | None = None,
    pre_pos: Tensor | None = None,
    post_pos: Tensor | None = None,
    myelin: Tensor | None = None,
    weight_init: float | None = None,
    delay_steps: Tensor | None = None,
    receptor: Tensor | None = None,
    target_compartments: Tensor | None = None,
) -> SynapseTopology:
    """Build a directed bipartite Erdos-Renyi topology (pre -> post)."""

    if n_pre <= 0 or n_post <= 0:
        raise ValueError("n_pre and n_post must be positive")
    if p < 0 or p > 1:
        raise ValueError("p must be within [0,1]")

    torch = require_torch()
    device_obj = torch.device(device) if device else None
    dtype_obj = _resolve_dtype(torch, dtype) if dtype else torch.get_default_dtype()

    pre_idx, post_idx = build_erdos_renyi_edges(
        n_pre=n_pre,
        n_post=n_post,
        p=p,
        device=device,
        allow_self=True,
    )

    weights = None
    if weight_init is not None:
        weights = torch.full((pre_idx.numel(),), weight_init, device=device_obj, dtype=dtype_obj)

    pre_pos_t = _to_device(pre_pos, device_obj, dtype_obj) if pre_pos is not None else None
    post_pos_t = _to_device(post_pos, device_obj, dtype_obj) if post_pos is not None else None
    myelin_t = _to_device(myelin, device_obj, dtype_obj) if myelin is not None else None

    delay_steps_t = None
    if delay_steps is not None:
        if hasattr(delay_steps, "to"):
            delay_steps_t = delay_steps.to(device=device_obj, dtype=torch.long)
        else:
            delay_steps_t = delay_steps
    elif pre_pos_t is not None and post_pos_t is not None and dt is not None:
        delay_steps_t = compute_delay_steps(
            pre_pos=pre_pos_t,
            post_pos=post_pos_t,
            pre_idx=pre_idx,
            post_idx=post_idx,
            dt=dt,
            myelin=myelin_t,
        )

    receptor = _to_device(receptor, device_obj, None) if receptor is not None else None
    target_compartments = (
        _to_device(target_compartments, device_obj, None) if target_compartments is not None else None
    )

    return SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        delay_steps=delay_steps_t,
        target_compartments=target_compartments,
        receptor=receptor,
        weights=weights,
        pre_pos=pre_pos_t,
        post_pos=post_pos_t,
        myelin=myelin_t,
    )


def build_erdos_renyi_edges(
    n_pre: int,
    n_post: int,
    p: float,
    *,
    device: str | None = None,
    allow_self: bool = False,
    seed: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample Erdos-Renyi edges without allocating a dense mask.

    Edges are sampled with replacement; duplicate pairs may occur.
    """

    if n_pre <= 0 or n_post <= 0:
        raise ValueError("n_pre and n_post must be positive")
    if p < 0 or p > 1:
        raise ValueError("p must be within [0,1]")

    torch = require_torch()
    device_obj = torch.device(device) if device else None

    total = n_pre * n_post
    if total <= 0:
        empty = torch.empty((0,), device=device_obj, dtype=torch.long)
        return empty, empty

    max_edges = total
    if not allow_self and n_pre == n_post:
        max_edges = total - n_pre

    target = int(p * total)
    if target <= 0 or max_edges <= 0:
        empty = torch.empty((0,), device=device_obj, dtype=torch.long)
        return empty, empty
    if target > max_edges:
        target = max_edges

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device_obj) if device_obj is not None else torch.Generator()
        generator.manual_seed(seed)

    def _sample_edges(count: int) -> tuple[Tensor, Tensor]:
        pre = torch.randint(
            0,
            n_pre,
            (count,),
            device=device_obj,
            dtype=torch.long,
            generator=generator,
        )
        post = torch.randint(
            0,
            n_post,
            (count,),
            device=device_obj,
            dtype=torch.long,
            generator=generator,
        )
        if not allow_self and n_pre == n_post:
            mask = pre != post
            if mask.all():
                return pre, post
            if hasattr(mask, "any") and not mask.any():
                empty = pre.new_empty((0,))
                return empty, empty
            return pre[mask], post[mask]
        return pre, post

    pre_parts: list[Tensor] = []
    post_parts: list[Tensor] = []
    remaining = target
    total_kept = 0
    oversample = 1.0
    if not allow_self and n_pre == n_post and n_pre > 1:
        oversample = 1.0 / (1.0 - 1.0 / n_pre)

    for _ in range(20):
        if remaining <= 0:
            break
        sample_size = int(remaining * oversample)
        if sample_size < remaining:
            sample_size = remaining
        pre, post = _sample_edges(sample_size)
        if pre.numel() == 0:
            continue
        pre_parts.append(pre)
        post_parts.append(post)
        total_kept += int(pre.numel())
        remaining = target - total_kept

    if not pre_parts:
        empty = torch.empty((0,), device=device_obj, dtype=torch.long)
        return empty, empty

    if len(pre_parts) == 1:
        pre_idx = pre_parts[0]
        post_idx = post_parts[0]
    else:
        pre_idx = torch.cat(pre_parts)
        post_idx = torch.cat(post_parts)
    if pre_idx.numel() > target:
        pre_idx = pre_idx[:target]
        post_idx = post_idx[:target]

    return pre_idx, post_idx


def _resolve_dtype(torch: Any, dtype: str | None) -> Any:
    if dtype is None:
        return torch.get_default_dtype()
    name = dtype
    if name.startswith("torch."):
        name = name.split(".", 1)[1]
    if not hasattr(torch, name):
        raise ValueError(f"Unknown torch dtype: {dtype}")
    return getattr(torch, name)


def _to_device(tensor: Tensor | None, device: Any, dtype: Any | None) -> Tensor | None:
    if tensor is None:
        return None
    if hasattr(tensor, "to"):
        kwargs = {"device": device} if device is not None else {}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return tensor.to(**kwargs)
    return tensor


__all__ = [
    "build_erdos_renyi_edges",
    "build_erdos_renyi_topology",
    "build_bipartite_erdos_renyi_topology",
]
