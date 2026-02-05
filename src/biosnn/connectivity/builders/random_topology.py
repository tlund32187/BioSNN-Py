"""Random topology builders."""

from __future__ import annotations

import math
from typing import Any, cast

from biosnn.connectivity.delays.axon import compute_delay_steps
from biosnn.contracts.neurons import Compartment
from biosnn.contracts.synapses import SynapseTopology
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch


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
    target_compartment: Compartment | None = None,
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
        target_compartment=target_compartment or Compartment.DENDRITE,
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
    target_compartment: Compartment | None = None,
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
        target_compartment=target_compartment or Compartment.DENDRITE,
        target_compartments=target_compartments,
        receptor=receptor,
        weights=weights,
        pre_pos=pre_pos_t,
        post_pos=post_pos_t,
        myelin=myelin_t,
    )


def build_bipartite_distance_topology(
    pre_positions: Tensor,
    post_positions: Tensor,
    p0: float,
    sigma: float,
    *,
    dist_p_mode: str = "gaussian",
    dist_p_sigma: float | None = None,
    max_fan_in: int | None = None,
    max_fan_out: int | None = None,
    device: str | None = None,
    dtype: str | None = None,
    seed: int | None = None,
    delay_from_distance: bool = False,
    delay_base_steps: int = 0,
    delay_per_unit_steps: float = 0.0,
    delay_max_steps: int | None = None,
) -> SynapseTopology:
    """Build a bipartite distance-aware topology using sparse candidate sampling."""

    if p0 < 0 or p0 > 1:
        raise ValueError("p0 must be within [0,1]")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if max_fan_in is not None and max_fan_in < 0:
        raise ValueError("max_fan_in must be non-negative")
    if max_fan_out is not None and max_fan_out < 0:
        raise ValueError("max_fan_out must be non-negative")
    if delay_max_steps is not None and delay_max_steps < 0:
        raise ValueError("delay_max_steps must be non-negative")

    torch = require_torch()
    device_obj = torch.device(device) if device else None
    dtype_obj = _resolve_dtype(torch, dtype) if dtype else torch.get_default_dtype()

    pre_pos_t = _to_device(pre_positions, device_obj, dtype_obj)
    post_pos_t = _to_device(post_positions, device_obj, dtype_obj)
    if pre_pos_t is None or post_pos_t is None:
        raise ValueError("pre_positions and post_positions are required")
    if pre_pos_t.dim() != 2 or pre_pos_t.shape[1] != 3:
        raise ValueError("pre_positions must have shape [Npre, 3]")
    if post_pos_t.dim() != 2 or post_pos_t.shape[1] != 3:
        raise ValueError("post_positions must have shape [Npost, 3]")

    n_pre = int(pre_pos_t.shape[0])
    n_post = int(post_pos_t.shape[0])
    if n_pre <= 0 or n_post <= 0:
        empty = torch.empty((0,), device=device_obj, dtype=torch.long)
        return SynapseTopology(pre_idx=empty, post_idx=empty, pre_pos=pre_pos_t, post_pos=post_pos_t)
    if p0 <= 0 or max_fan_in == 0 or max_fan_out == 0:
        empty = torch.empty((0,), device=device_obj, dtype=torch.long)
        return SynapseTopology(pre_idx=empty, post_idx=empty, pre_pos=pre_pos_t, post_pos=post_pos_t)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device_obj) if device_obj is not None else torch.Generator()
        generator.manual_seed(seed)

    candidate_factor = 4.0
    candidate_count = min(n_pre, max(1, int(math.ceil(p0 * n_pre * candidate_factor))))

    mode = str(dist_p_mode or "gaussian").lower()
    sigma_prob = float(dist_p_sigma) if dist_p_sigma is not None else float(sigma)
    if mode == "gaussian" and sigma_prob <= 0:
        raise ValueError("dist_p_sigma must be positive when dist_p_mode='gaussian'")
    denom = 2.0 * sigma_prob * sigma_prob if mode == "gaussian" else 1.0
    pre_chunks: list[Tensor] = []
    post_chunks: list[Tensor] = []
    delay_chunks: list[Tensor] = []
    dist_chunks: list[Tensor] = []

    pre_counts = [0] * n_pre if max_fan_out is not None else None
    post_counts = [0] * n_post if max_fan_in is not None else None

    for post_idx in range(n_post):
        if max_fan_in is not None and post_counts is not None and post_counts[post_idx] >= max_fan_in:
            continue

        remaining = None
        if max_fan_in is not None and post_counts is not None:
            remaining = max_fan_in - post_counts[post_idx]
            if remaining <= 0:
                continue

        cand_idx = torch.randint(0, n_pre, (candidate_count,), device=device_obj, generator=generator)
        pre_cand = pre_pos_t.index_select(0, cand_idx)
        post_vec = post_pos_t[post_idx]
        diff = pre_cand - post_vec
        dist2 = (diff * diff).sum(dim=1)
        if mode == "gaussian":
            probs = p0 * torch.exp(-dist2 / denom)
        else:
            probs = torch.full((candidate_count,), p0, device=device_obj, dtype=dist2.dtype)

        draws = torch.rand((candidate_count,), device=device_obj, generator=generator)
        mask = draws < probs
        if not torch.any(mask):
            continue

        cand_idx = cand_idx[mask]
        dist = torch.sqrt(dist2[mask])

        if remaining is not None and cand_idx.numel() > remaining:
            perm = torch.randperm(cand_idx.numel(), device=device_obj, generator=generator)
            perm = perm[:remaining]
            cand_idx = cand_idx.index_select(0, perm)
            dist = dist.index_select(0, perm)

        if cand_idx.numel() == 0:
            continue

        if pre_counts is not None and max_fan_out is not None:
            cand_cpu = cand_idx.detach().cpu().tolist()
            dist_cpu = dist.detach().cpu().tolist()
            kept_pre: list[int] = []
            kept_dist: list[float] = []
            for idx, pre in enumerate(cand_cpu):
                if pre_counts[pre] >= max_fan_out:
                    continue
                pre_counts[pre] += 1
                kept_pre.append(pre)
                kept_dist.append(float(dist_cpu[idx]))
                if max_fan_in is not None and post_counts is not None:
                    post_counts[post_idx] += 1
                    if post_counts[post_idx] >= max_fan_in:
                        break
            if not kept_pre:
                continue
            pre_tensor = torch.tensor(kept_pre, device=device_obj, dtype=torch.long)
            post_tensor = torch.full(
                (len(kept_pre),),
                post_idx,
                device=device_obj,
                dtype=torch.long,
            )
            pre_chunks.append(pre_tensor)
            post_chunks.append(post_tensor)
            dist_chunks.append(torch.tensor(kept_dist, device=device_obj, dtype=dist.dtype))
            if delay_from_distance:
                dist_tensor = torch.tensor(kept_dist, device=device_obj, dtype=dtype_obj)
                delay_chunks.append(
                    _distance_to_delay_steps(
                        dist_tensor,
                        base=delay_base_steps,
                        per_unit=delay_per_unit_steps,
                        max_steps=delay_max_steps,
                    )
                )
        else:
            pre_chunks.append(cand_idx.to(dtype=torch.long))
            post_chunks.append(
                torch.full((cand_idx.numel(),), post_idx, device=device_obj, dtype=torch.long)
            )
            dist_chunks.append(dist)
            if post_counts is not None and max_fan_in is not None:
                post_counts[post_idx] += int(cand_idx.numel())
            if delay_from_distance:
                delay_chunks.append(
                    _distance_to_delay_steps(
                        dist,
                        base=delay_base_steps,
                        per_unit=delay_per_unit_steps,
                        max_steps=delay_max_steps,
                    )
                )

    if not pre_chunks:
        empty = torch.empty((0,), device=device_obj, dtype=torch.long)
        return SynapseTopology(
            pre_idx=empty,
            post_idx=empty,
            delay_steps=None,
            edge_dist=None,
            pre_pos=pre_pos_t,
            post_pos=post_pos_t,
        )

    pre_idx_tensor = pre_chunks[0] if len(pre_chunks) == 1 else torch.cat(pre_chunks)
    post_idx_tensor = post_chunks[0] if len(post_chunks) == 1 else torch.cat(post_chunks)
    delay_steps = None
    if delay_from_distance:
        delay_steps = delay_chunks[0] if len(delay_chunks) == 1 else torch.cat(delay_chunks)
    edge_dist = dist_chunks[0] if len(dist_chunks) == 1 else torch.cat(dist_chunks)

    return SynapseTopology(
        pre_idx=pre_idx_tensor,
        post_idx=post_idx_tensor,
        delay_steps=delay_steps,
        edge_dist=edge_dist,
        pre_pos=pre_pos_t,
        post_pos=post_pos_t,
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


def _distance_to_delay_steps(
    distance: Tensor,
    *,
    base: int,
    per_unit: float,
    max_steps: int | None,
) -> Tensor:
    torch = require_torch()
    steps = distance * float(per_unit) + float(base)
    steps = torch.round(steps)
    if max_steps is not None:
        steps = torch.clamp(steps, min=0.0, max=float(max_steps))
    else:
        steps = torch.clamp(steps, min=0.0)
    return cast(Tensor, steps.to(dtype=torch.int32))


__all__ = [
    "build_erdos_renyi_edges",
    "build_erdos_renyi_topology",
    "build_bipartite_erdos_renyi_topology",
    "build_bipartite_distance_topology",
]
