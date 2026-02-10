"""GPU-friendly 2D grid diffusion neuromodulator field."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from biosnn.contracts.modulators import IModulatorField, ModulatorKind, ModulatorRelease
from biosnn.contracts.tensor import Tensor
from biosnn.core.torch_utils import require_torch, resolve_device_dtype


@dataclass(frozen=True, slots=True)
class GridDiffusion2DParams:
    kinds: tuple[ModulatorKind, ...] = (ModulatorKind.DOPAMINE,)
    grid_size: tuple[int, int] = (16, 16)  # (H, W)
    world_origin: tuple[float, float] = (0.0, 0.0)
    world_extent: tuple[float, float] = (1.0, 1.0)
    diffusion: float | Mapping[ModulatorKind, float] = 0.0
    decay_tau: float | Mapping[ModulatorKind, float] = 1.0
    deposit_sigma: float = 0.0
    clamp_min: float | None = None
    clamp_max: float | None = None
    laplacian_mode: Literal["conv"] = "conv"


@dataclass(slots=True)
class GridDiffusion2DState:
    grid: Tensor  # [K, H, W]
    scratch_grid: Tensor  # [K, H, W] reused scratch for release deposits


class GridDiffusion2DField(IModulatorField):
    """2D per-kind diffusion field sampled at xyz positions."""

    name = "grid_diffusion_2d"

    def __init__(self, *, params: GridDiffusion2DParams | None = None) -> None:
        self.params = params or GridDiffusion2DParams()
        if not self.params.kinds:
            raise ValueError("GridDiffusion2DParams.kinds must not be empty.")
        if self.params.laplacian_mode != "conv":
            raise ValueError("GridDiffusion2DField supports laplacian_mode='conv' only.")
        h, w = int(self.params.grid_size[0]), int(self.params.grid_size[1])
        if h <= 0 or w <= 0:
            raise ValueError("grid_size must contain positive dimensions.")

        wx, wy = float(self.params.world_extent[0]), float(self.params.world_extent[1])
        if wx == 0.0 or wy == 0.0:
            raise ValueError("world_extent components must be non-zero.")

        self.kinds = tuple(dict.fromkeys(self.params.kinds))
        self._kind_to_idx = {kind: idx for idx, kind in enumerate(self.kinds)}

        self._laplacian_kernel_cache: dict[tuple[str, str], Tensor] = {}
        self._diffusion_cache: dict[tuple[str, str], Tensor] = {}
        self._inv_decay_cache: dict[tuple[str, str], Tensor] = {}
        self._deposit_kernel_cache: dict[tuple[str, str], Tensor] = {}

    def init_state(self, *, ctx: Any) -> GridDiffusion2DState:
        torch = require_torch()
        device, dtype = resolve_device_dtype(ctx)
        h, w = int(self.params.grid_size[0]), int(self.params.grid_size[1])
        shape = (len(self.kinds), h, w)
        grid = torch.zeros(shape, device=device, dtype=dtype)
        scratch = torch.zeros_like(grid)
        self._ensure_runtime_caches(device=device, dtype=dtype)
        return GridDiffusion2DState(grid=grid, scratch_grid=scratch)

    def step(
        self,
        state: GridDiffusion2DState,
        *,
        releases: Sequence[ModulatorRelease],
        dt: float,
        t: float,
        ctx: Any,
    ) -> GridDiffusion2DState:
        _ = (t, ctx)
        torch = require_torch()
        f = torch.nn.functional

        device = state.grid.device
        dtype = state.grid.dtype
        self._ensure_runtime_caches(device=device, dtype=dtype)

        if releases:
            state.scratch_grid.zero_()
            positions_xy_chunks: list[Tensor] = []
            amount_chunks: list[Tensor] = []
            kind_idx_chunks: list[Tensor] = []
            for release in releases:
                kind_idx = self._kind_to_idx.get(release.kind)
                if kind_idx is None:
                    continue
                positions = cast(Tensor, release.positions)
                amount = cast(Tensor, release.amount)
                if int(positions.numel()) == 0 or int(amount.numel()) == 0:
                    continue

                positions_xy = positions[:, :2].to(device=device, dtype=dtype)
                amount_flat = amount.reshape(-1).to(device=device, dtype=dtype)
                n_pos = int(positions_xy.shape[0])
                n_amt = int(amount_flat.shape[0])
                if n_amt == 1 and n_pos != 1:
                    amount_flat = amount_flat.expand(n_pos)
                elif n_amt != n_pos:
                    raise ValueError(
                        "release.amount must have one value per release position or be scalar."
                    )

                positions_xy_chunks.append(positions_xy)
                amount_chunks.append(amount_flat)
                kind_idx_chunks.append(
                    torch.full((n_pos,), kind_idx, device=device, dtype=torch.long)
                )

            if positions_xy_chunks:
                positions_xy = torch.cat(positions_xy_chunks, dim=0)
                amounts = torch.cat(amount_chunks, dim=0)
                kind_idx = torch.cat(kind_idx_chunks, dim=0)
                flat_idx = self._release_flat_indices(
                    positions_xy=positions_xy,
                    kind_idx=kind_idx,
                    h=int(state.grid.shape[1]),
                    w=int(state.grid.shape[2]),
                )
                state.scratch_grid.view(-1).scatter_add_(0, flat_idx, amounts)

                deposit_kernel = self._deposit_kernel_cache.get(_cache_key(device=device, dtype=dtype))
                if deposit_kernel is None:
                    state.grid.add_(state.scratch_grid)
                else:
                    radius = int((int(deposit_kernel.shape[-1]) - 1) // 2)
                    smoothed = f.conv2d(
                        state.scratch_grid.unsqueeze(0),
                        deposit_kernel,
                        padding=radius,
                        groups=len(self.kinds),
                    ).squeeze(0)
                    state.grid.add_(smoothed)

        laplacian_kernel = self._laplacian_kernel_cache[_cache_key(device=device, dtype=dtype)]
        diffusion = self._diffusion_cache[_cache_key(device=device, dtype=dtype)]
        inv_decay = self._inv_decay_cache[_cache_key(device=device, dtype=dtype)]
        laplacian = f.conv2d(
            state.grid.unsqueeze(0),
            laplacian_kernel,
            padding=1,
            groups=len(self.kinds),
        ).squeeze(0)
        state.grid.add_(dt * (diffusion * laplacian - inv_decay * state.grid))

        if self.params.clamp_min is not None or self.params.clamp_max is not None:
            state.grid.clamp_(min=self.params.clamp_min, max=self.params.clamp_max)
        return state

    def sample_at(
        self,
        state: GridDiffusion2DState,
        *,
        positions: Tensor,
        kind: ModulatorKind,
        ctx: Any,
    ) -> Tensor:
        _ = ctx
        torch = require_torch()
        f = torch.nn.functional
        n = int(positions.shape[0])
        if n == 0:
            return cast(Tensor, torch.zeros((0,), device=positions.device, dtype=positions.dtype))

        idx = self._kind_to_idx.get(kind)
        if idx is None:
            return cast(Tensor, torch.zeros((n,), device=positions.device, dtype=positions.dtype))

        x0, y0 = float(self.params.world_origin[0]), float(self.params.world_origin[1])
        wx, wy = float(self.params.world_extent[0]), float(self.params.world_extent[1])
        wx = wx if wx != 0.0 else 1e-12
        wy = wy if wy != 0.0 else 1e-12

        pos_xy = positions[:, :2].to(device=state.grid.device, dtype=state.grid.dtype)
        u = ((pos_xy[:, 0] - x0) / wx).clamp(0.0, 1.0)
        v = ((pos_xy[:, 1] - y0) / wy).clamp(0.0, 1.0)
        sample_grid = torch.stack((u * 2.0 - 1.0, v * 2.0 - 1.0), dim=-1).view(1, n, 1, 2)

        channel = state.grid[idx : idx + 1].unsqueeze(0)
        sampled = f.grid_sample(
            channel,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        values = sampled.reshape(n).to(device=positions.device, dtype=positions.dtype)
        return cast(Tensor, values)

    def state_tensors(self, state: GridDiffusion2DState) -> Mapping[str, Tensor]:
        return {
            "grid": state.grid,
            "grid_mean": state.grid.mean(),
            "grid_max": state.grid.amax(),
        }

    def _release_flat_indices(self, *, positions_xy: Tensor, kind_idx: Tensor, h: int, w: int) -> Tensor:
        torch = require_torch()
        x0, y0 = float(self.params.world_origin[0]), float(self.params.world_origin[1])
        wx, wy = float(self.params.world_extent[0]), float(self.params.world_extent[1])
        wx = wx if wx != 0.0 else 1e-12
        wy = wy if wy != 0.0 else 1e-12

        u = ((positions_xy[:, 0] - x0) / wx).clamp(0.0, 1.0)
        v = ((positions_xy[:, 1] - y0) / wy).clamp(0.0, 1.0)
        col = torch.round(u * float(w - 1)).to(dtype=torch.long)
        row = torch.round(v * float(h - 1)).to(dtype=torch.long)
        per_kind_offset = kind_idx * (h * w)
        return cast(Tensor, per_kind_offset + row * w + col)

    def _ensure_runtime_caches(self, *, device: Any, dtype: Any) -> None:
        torch = require_torch()
        key = _cache_key(device=device, dtype=dtype)
        if key in self._laplacian_kernel_cache:
            return

        k = len(self.kinds)
        laplace = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            device=device,
            dtype=dtype,
        ).view(1, 1, 3, 3)
        self._laplacian_kernel_cache[key] = laplace.repeat(k, 1, 1, 1)

        diffusion_vals = _resolve_per_kind_values(self.params.diffusion, kinds=self.kinds)
        diffusion = torch.tensor(diffusion_vals, device=device, dtype=dtype).view(k, 1, 1)
        self._diffusion_cache[key] = cast(Tensor, diffusion)

        tau_vals = _resolve_per_kind_values(self.params.decay_tau, kinds=self.kinds)
        tau = torch.tensor(tau_vals, device=device, dtype=dtype).view(k, 1, 1)
        inv_decay = torch.where(tau > 0.0, 1.0 / tau, torch.zeros_like(tau))
        self._inv_decay_cache[key] = cast(Tensor, inv_decay)

        sigma = float(self.params.deposit_sigma)
        if sigma <= 0.0:
            return
        radius = max(1, int(torch.ceil(torch.tensor(3.0 * sigma)).item()))
        offsets = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        kernel_1d = torch.exp(-(offsets**2) / (2.0 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel = kernel_2d.view(1, 1, kernel_2d.shape[0], kernel_2d.shape[1]).repeat(k, 1, 1, 1)
        self._deposit_kernel_cache[key] = cast(Tensor, kernel)


def _resolve_per_kind_values(
    value: float | Mapping[ModulatorKind, float],
    *,
    kinds: Sequence[ModulatorKind],
) -> list[float]:
    if isinstance(value, Mapping):
        resolved: list[float] = []
        for kind in kinds:
            if kind in value:
                resolved.append(float(value[kind]))
            elif kind.value in value:
                resolved.append(float(cast(Any, value)[kind.value]))
            else:
                raise KeyError(f"Missing per-kind value for modulator '{kind.value}'.")
        return resolved
    scalar = float(value)
    return [scalar for _ in kinds]


def _cache_key(*, device: Any, dtype: Any) -> tuple[str, str]:
    return (str(device), str(dtype))


__all__ = [
    "GridDiffusion2DField",
    "GridDiffusion2DParams",
    "GridDiffusion2DState",
]

