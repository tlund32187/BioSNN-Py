"""Monitor aggregation and dispatch subsystem."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from biosnn.contracts.monitors import IMonitor, MonitorRequirements, StepEvent


class MonitorSubsystem:
    """Owns monitor requirements, compatibility checks, and dispatch lifecycle."""

    def has_active(self, monitors: Sequence[IMonitor]) -> bool:
        if not monitors:
            return False
        return any(getattr(monitor, "enabled", True) for monitor in monitors)

    def collect_requirements(self, monitors: Sequence[IMonitor]) -> MonitorRequirements:
        requirements = MonitorRequirements.none()
        for monitor in monitors:
            if not getattr(monitor, "enabled", True):
                continue
            req_fn = getattr(monitor, "requirements", None)
            if callable(req_fn):
                try:
                    monitor_requirements = req_fn()
                except Exception:
                    monitor_requirements = None
                if isinstance(monitor_requirements, MonitorRequirements):
                    requirements = requirements.merge(monitor_requirements)
                    continue
            requirements = requirements.merge(MonitorRequirements.all())
        return requirements

    def collect_compilation_requirements(self, monitors: Sequence[IMonitor]) -> dict[str, bool]:
        requirements: dict[str, bool] = {}
        monitor_payload_requirements = self.collect_requirements(monitors)
        if monitor_payload_requirements.needs_projection_weights:
            requirements["wants_weights_snapshot_each_step"] = True
        if monitor_payload_requirements.needs_projection_drive:
            requirements["wants_projection_drive_tensor"] = True

        for monitor in monitors:
            if not getattr(monitor, "enabled", True):
                continue
            req_fn = getattr(monitor, "compilation_requirements", None)
            if not callable(req_fn):
                continue
            try:
                monitor_requirements = req_fn()
            except Exception:
                continue
            if not isinstance(monitor_requirements, Mapping):
                continue
            for key, value in monitor_requirements.items():
                requirements[str(key)] = bool(requirements.get(str(key), False) or bool(value))
        return requirements

    def validate_fast_mode_monitors(self, monitors: Sequence[IMonitor]) -> None:
        try:
            from biosnn.monitors.csv import NeuronCSVMonitor, SynapseCSVMonitor
            from biosnn.monitors.raster.spike_events_csv import SpikeEventsCSVMonitor
        except Exception:
            return

        incompatible = [
            monitor
            for monitor in monitors
            if isinstance(monitor, (NeuronCSVMonitor, SynapseCSVMonitor, SpikeEventsCSVMonitor))
        ]
        if incompatible:
            names = ", ".join(type(mon).__name__ for mon in incompatible)
            raise RuntimeError(
                "fast_mode=True does not support monitors that require merged tensors or global spikes. "
                f"Incompatible monitors: {names}"
            )

    def on_step(self, monitors: Sequence[IMonitor], event: StepEvent) -> None:
        for monitor in monitors:
            monitor.on_step(event)

    def flush_and_close(self, monitors: Sequence[IMonitor]) -> None:
        for monitor in monitors:
            monitor.flush()
        for monitor in monitors:
            monitor.close()

    def bind_engine(self, monitors: Sequence[IMonitor], engine: Any) -> None:
        for monitor in monitors:
            bind_engine = getattr(monitor, "bind_engine", None)
            if callable(bind_engine):
                bind_engine(engine)


__all__ = ["MonitorSubsystem"]
