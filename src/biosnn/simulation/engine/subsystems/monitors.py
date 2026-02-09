"""Monitor aggregation and dispatch subsystem."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from biosnn.contracts.monitors import IMonitor, MonitorRequirements, StepEvent
from biosnn.simulation.engine.subsystems.models import (
    STEP_EVENT_KEY_HOMEOSTASIS_STATE,
    STEP_EVENT_KEY_LEARNING_STATE,
    STEP_EVENT_KEY_MODULATORS,
    STEP_EVENT_KEY_POPULATION_SLICES,
    STEP_EVENT_KEY_POPULATION_STATE,
    STEP_EVENT_KEY_PROJECTION_DRIVE,
    STEP_EVENT_KEY_PROJECTION_WEIGHTS,
    STEP_EVENT_KEY_SCALARS,
    STEP_EVENT_KEY_SPIKES,
    STEP_EVENT_KEY_SYNAPSE_STATE,
    STEP_EVENT_KEY_V_SOMA,
    NetworkRequirements,
)


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

    def collect_compilation_requirements(
        self, monitors: Sequence[IMonitor]
    ) -> dict[str, bool | str | None]:
        requirements: dict[str, bool | str | None] = {}
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
                req_key = str(key)
                requirements[req_key] = _merge_compilation_requirement_value(
                    requirements.get(req_key),
                    value,
                )
        return requirements

    def build_network_requirements(
        self,
        monitors: Sequence[IMonitor],
        *,
        monitor_requirements: MonitorRequirements | None = None,
        compilation_requirements: Mapping[str, bool | str | None] | None = None,
    ) -> NetworkRequirements:
        payload_requirements = monitor_requirements or self.collect_requirements(monitors)
        compile_requirements = (
            compilation_requirements if compilation_requirements is not None else self.collect_compilation_requirements(monitors)
        )

        needed_step_event_keys: set[str] = set()
        if payload_requirements.needs_spikes:
            needed_step_event_keys.add(STEP_EVENT_KEY_SPIKES)
        if payload_requirements.needs_v_soma:
            needed_step_event_keys.add(STEP_EVENT_KEY_V_SOMA)
        if payload_requirements.needs_population_state:
            needed_step_event_keys.add(STEP_EVENT_KEY_POPULATION_STATE)
        if payload_requirements.needs_projection_weights:
            needed_step_event_keys.add(STEP_EVENT_KEY_PROJECTION_WEIGHTS)
        if payload_requirements.needs_projection_drive:
            needed_step_event_keys.add(STEP_EVENT_KEY_PROJECTION_DRIVE)
        if payload_requirements.needs_synapse_state:
            needed_step_event_keys.add(STEP_EVENT_KEY_SYNAPSE_STATE)
        if payload_requirements.needs_modulators:
            needed_step_event_keys.add(STEP_EVENT_KEY_MODULATORS)
        if payload_requirements.needs_learning_state:
            needed_step_event_keys.add(STEP_EVENT_KEY_LEARNING_STATE)
        if payload_requirements.needs_homeostasis_state:
            needed_step_event_keys.add(STEP_EVENT_KEY_HOMEOSTASIS_STATE)
        if payload_requirements.needs_scalars:
            needed_step_event_keys.add(STEP_EVENT_KEY_SCALARS)
        if payload_requirements.needs_population_slices:
            needed_step_event_keys.add(STEP_EVENT_KEY_POPULATION_SLICES)

        wants_fused_layout = _as_str(compile_requirements.get("wants_fused_layout")) or "auto"
        if wants_fused_layout == "auto" and _as_bool(compile_requirements.get("wants_fused_csr")):
            wants_fused_layout = "csr"

        ring_strategy = _as_str(compile_requirements.get("ring_strategy")) or "dense"
        ring_dtype = _as_str(compile_requirements.get("ring_dtype"))

        return NetworkRequirements(
            needed_step_event_keys=frozenset(needed_step_event_keys),
            needs_bucket_edge_mapping=_as_bool(
                compile_requirements.get("wants_bucket_edge_mapping")
            ),
            needs_by_delay_sparse=_as_bool(compile_requirements.get("wants_by_delay_sparse")),
            wants_fused_layout=wants_fused_layout,
            ring_strategy=ring_strategy,
            ring_dtype=ring_dtype,
        )

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


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value_norm = value.strip().lower()
        if value_norm in {"1", "true", "yes", "on"}:
            return True
        if value_norm in {"0", "false", "no", "off"}:
            return False
    return False


def _as_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value_str = value.strip()
    if not value_str:
        return None
    return value_str


def _merge_compilation_requirement_value(
    current: bool | str | None,
    incoming: object,
) -> bool | str | None:
    if isinstance(incoming, bool):
        if isinstance(current, bool):
            return bool(current or incoming)
        if isinstance(current, str):
            return current
        return incoming
    if isinstance(incoming, str):
        incoming_str = incoming.strip()
        if not incoming_str:
            return current
        if isinstance(current, str):
            if current.strip() == incoming_str:
                return current
            return incoming_str
        if isinstance(current, bool):
            if current:
                return current
            return incoming_str
        return incoming_str
    return current
