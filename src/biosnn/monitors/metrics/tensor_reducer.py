"""Tensor reduction helpers for monitors."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from biosnn.contracts.tensor import Tensor


def reduce_tensor(
    name: str,
    tensor: Tensor,
    reductions: Sequence[str],
    sample_indices: Sequence[int] | Tensor | None = None,
) -> dict[str, float]:
    """Reduce a tensor into a row mapping of statistics and samples."""

    row: dict[str, float] = {}
    if tensor is None:
        return row

    for stat in reductions:
        row[f"{name}_{stat}"] = reduce_stat(tensor, stat)

    if sample_indices is not None:
        for idx, value in sample_tensor(tensor, sample_indices):
            row[f"{name}_i{idx}"] = value

    return row


def reduce_stat(values: Any, stat: str) -> float:
    """Compute a scalar reduction without moving full tensors to CPU."""

    if values is None:
        return 0.0

    candidate = values
    if hasattr(candidate, "detach"):
        candidate = candidate.detach()
    if hasattr(candidate, "flatten"):
        flat = candidate.flatten()
        if stat == "mean":
            dtype = getattr(flat, "dtype", None)
            dtype_str = str(dtype).lower() if dtype is not None else ""
            if "bool" in dtype_str or (
                "float" not in dtype_str and "complex" not in dtype_str
            ):
                if hasattr(flat, "float"):
                    flat = flat.float()
                elif hasattr(flat, "astype"):
                    flat = flat.astype(float)
        reducer = getattr(flat, stat, None)
        if callable(reducer):
            try:
                return _to_scalar(reducer())
            except Exception:
                pass

    if isinstance(candidate, Iterable):
        flat_list = list(_flatten(candidate))
        if not flat_list:
            return 0.0
        if stat == "mean":
            return float(sum(flat_list) / len(flat_list))
        if stat == "sum":
            return float(sum(flat_list))
        if stat == "min":
            return float(min(flat_list))
        if stat == "max":
            return float(max(flat_list))

    return _to_scalar(candidate)


def sample_tensor(values: Any, indices: Sequence[int] | Tensor) -> list[tuple[int, float]]:
    """Sample values from a tensor or sequence by index."""

    if values is None:
        return []

    index_list = _indices_to_list(indices)
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "flatten"):
        flat = values.flatten()
        try:
            sampled = _index_values(flat, indices)
        except Exception:
            sampled = [flat[idx] for idx in index_list]
        return [(int(idx), _to_scalar(val)) for idx, val in zip(index_list, _to_list(sampled), strict=False)]

    return [(int(idx), _to_scalar(values[idx])) for idx in index_list]


def _index_values(values: Any, indices: Sequence[int] | Tensor) -> Any:
    if hasattr(indices, "device") or hasattr(indices, "dtype"):
        idx_tensor = indices
        if hasattr(values, "device") and hasattr(idx_tensor, "device"):
            if idx_tensor.device != values.device and hasattr(idx_tensor, "to"):
                idx_tensor = idx_tensor.to(device=values.device)
        if hasattr(idx_tensor, "long"):
            idx_tensor = idx_tensor.long()
        if hasattr(values, "index_select"):
            try:
                return values.index_select(0, idx_tensor)
            except Exception:
                pass
        return values[idx_tensor]
    return values[indices]


def _indices_to_list(indices: Sequence[int] | Tensor) -> list[int]:
    if hasattr(indices, "tolist"):
        raw = indices.tolist()
        return [int(value) for value in raw]
    return [int(value) for value in indices]


def _flatten(values: Any) -> Iterable[float]:
    if isinstance(values, (list, tuple)):
        for item in values:
            yield from _flatten(item)
    else:
        yield _to_scalar(values)


def _to_list(values: Any) -> list[Any]:
    if hasattr(values, "tolist"):
        return values.tolist()
    if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
        return list(values)
    return [values]


def _to_scalar(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


__all__ = ["reduce_stat", "reduce_tensor", "sample_tensor"]
