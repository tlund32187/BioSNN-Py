"""Beginner-friendly training orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

from biosnn.contracts.monitors import IMonitor, StepEvent
from biosnn.contracts.simulation import SimulationConfig
from biosnn.core.torch_utils import require_torch
from biosnn.monitors.metrics.scalar_utils import scalar_to_float

if TYPE_CHECKING:
    from biosnn.simulation.engine import TorchNetworkEngine


@dataclass(frozen=True, slots=True)
class EngineConfig:
    fast_mode: bool = True
    compiled_mode: bool = False
    monitors_enabled: bool = False
    monitors: tuple[IMonitor, ...] = ()
    spike_stride: int = 2
    spike_cap: int = 5000
    weights_stride: int = 10
    weights_cap: int = 20000
    neuron_sample: int = 32
    synapse_sample: int = 64
    flush_every: int = 25
    device: str | None = None
    dtype: str | None = None


@dataclass(frozen=True, slots=True)
class TrainReport:
    steps: int
    elapsed_s: float
    steps_per_sec: float
    engine_name: str
    device: str | None
    dtype: str | None
    log_every: int


class Trainer:
    """Thin orchestration wrapper for engines + logging."""

    def __init__(
        self,
        network: Any,
        *,
        engine_config: EngineConfig | None = None,
        monitors: tuple[IMonitor, ...] | None = None,
        device: str | None = None,
        dtype: str | None = None,
        seed: int | None = None,
    ) -> None:
        self._network = network
        self._engine_config = engine_config or EngineConfig()
        self._monitors = monitors
        self._device_override = device
        self._dtype_override = dtype
        self._seed_override = seed
        self._engine: TorchNetworkEngine | None = None

    # Future coexistence modes (documented only):
    # (a) evolve structure; online learning adapts behavior
    # (b) evolve plasticity rules (meta-learning)
    # (c) ES slow params + local plasticity fast params

    def build_engine(self):
        from biosnn.simulation.engine import TorchNetworkEngine

        engine = TorchNetworkEngine(
            populations=self._network.populations,
            projections=self._network.projections,
            modulators=self._network.modulators,
            fast_mode=self._engine_config.fast_mode,
            compiled_mode=self._engine_config.compiled_mode,
        )
        self._engine = engine
        return engine

    def compile(self, *, dt: float = 1e-3):
        engine = self._engine or self.build_engine()
        device = self._device_override or self._engine_config.device
        dtype = self._dtype_override or self._engine_config.dtype
        seed = self._seed_override
        if device is None or dtype is None:
            device, dtype = _infer_device_dtype(self._network, device, dtype)

        if self._engine_config.monitors_enabled:
            monitors = self._monitors or self._engine_config.monitors
            if monitors:
                engine.attach_monitors(monitors)

        engine.reset(
            config=SimulationConfig(
                dt=dt,
                seed=seed,
                device=device,
                dtype=dtype,
            )
        )
        return engine

    def run(self, *, steps: int, log_every: int = 100, progress: bool = True, dt: float = 1e-3) -> TrainReport:
        engine = self._engine or self.compile(dt=dt)
        start = perf_counter()
        last_event: StepEvent | None = None
        for step_idx in range(1, steps + 1):
            event = engine.step()
            last_event = event if isinstance(event, StepEvent) else None
            if log_every > 0 and (step_idx == 1 or step_idx % log_every == 0 or step_idx == steps):
                now = perf_counter()
                elapsed = now - start
                steps_per_sec = step_idx / elapsed if elapsed > 0 else 0.0
                if progress:
                    msg = _format_log(step_idx, steps, steps_per_sec, last_event)
                    print(msg)

        total_elapsed = perf_counter() - start
        steps_per_sec = steps / total_elapsed if total_elapsed > 0 else 0.0
        return TrainReport(
            steps=steps,
            elapsed_s=total_elapsed,
            steps_per_sec=steps_per_sec,
            engine_name=engine.name,
            device=getattr(engine, "_device", None),
            dtype=getattr(engine, "_dtype", None),
            log_every=log_every,
        )

    def evaluate(self, *, steps: int, log_every: int = 100, dt: float = 1e-3) -> TrainReport:
        # Stub for future evaluation API.
        return self.run(steps=steps, log_every=log_every, progress=False, dt=dt)

    def save_checkpoint(self, path: str) -> None:
        torch = require_torch()
        if self._engine is None:
            raise RuntimeError("Trainer has no engine. Call compile() before saving.")
        engine = self._engine
        payload: dict[str, Any] = {
            "engine_name": engine.name,
            "step": getattr(engine, "_step", None),
            "t": getattr(engine, "_t", None),
        }
        if hasattr(engine, "state_dict"):
            payload["engine_state"] = engine.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path: str) -> dict[str, Any]:
        torch = require_torch()
        if self._engine is None:
            raise RuntimeError("Trainer has no engine. Call compile() before loading.")
        payload = cast(dict[str, Any], torch.load(path, map_location="cpu"))
        engine = self._engine
        if "engine_state" in payload and hasattr(engine, "load_state_dict"):
            engine.load_state_dict(payload["engine_state"])
        if payload.get("step") is not None and hasattr(engine, "_step"):
            engine._step = payload["step"]
        if payload.get("t") is not None and hasattr(engine, "_t"):
            engine._t = payload["t"]
        return payload


def _infer_device_dtype(network: Any, device: str | None, dtype: str | None) -> tuple[str | None, str | None]:
    device_out = device
    dtype_out = dtype
    for proj in getattr(network, "projections", []):
        topo = proj.topology
        if device_out is None and hasattr(topo.pre_idx, "device"):
            device_out = str(topo.pre_idx.device)
        if dtype_out is None and topo.weights is not None and hasattr(topo.weights, "dtype"):
            dtype_out = str(topo.weights.dtype)
    return device_out, dtype_out


def _format_log(step: int, total: int, steps_per_sec: float, event: StepEvent | None) -> str:
    suffix = ""
    if event and event.scalars:
        try:
            spike_rate = event.scalars.get("spike_rate_hz")
            if spike_rate is not None:
                suffix = f", spike_rate={scalar_to_float(spike_rate):.2f} hz"
        except Exception:
            suffix = ""
    return f"[{step}/{total}] {steps_per_sec:.1f} steps/s{suffix}"


__all__ = ["Trainer", "EngineConfig", "TrainReport"]
