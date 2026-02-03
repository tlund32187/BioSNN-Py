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

    mask = torch.rand((n, n), device=device_obj) < p
    if not allow_self:
        if hasattr(mask, "fill_diagonal_"):
            mask.fill_diagonal_(False)
        else:
            mask = mask & ~torch.eye(n, dtype=torch.bool, device=device_obj)

    idx = mask.nonzero(as_tuple=False)
    pre_idx = idx[:, 0].to(dtype=torch.long)
    post_idx = idx[:, 1].to(dtype=torch.long)

    if device_obj is not None:
        pre_idx = pre_idx.to(device=device_obj)
        post_idx = post_idx.to(device=device_obj)

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


__all__ = ["build_erdos_renyi_topology"]
