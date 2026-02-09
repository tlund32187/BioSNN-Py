"""Topology compilation subsystem for TorchNetworkEngine."""

from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, SupportsInt, cast

from biosnn.connectivity.topology_compile import compile_topology
from biosnn.contracts.synapses import SynapseTopology
from biosnn.contracts.tensor import Tensor
from biosnn.simulation.network.specs import PopulationSpec, ProjectionSpec

from .models import CompiledNetworkPlan, NetworkRequirements, ProjectionPlan


@dataclass(frozen=True, slots=True)
class ProjectionCompileFlags:
    build_edges_by_delay: bool
    build_pre_adjacency: bool
    build_sparse_delay_mats: bool
    build_bucket_edge_mapping: bool
    fuse_delay_buckets: bool
    store_sparse_by_delay: bool
    build_fused_csr: bool


_TORCH_THREAD_LIMIT_LOCK = threading.Lock()
_TORCH_THREAD_LIMIT_ACTIVE = 0
_TORCH_THREAD_LIMIT_STATE: tuple[int | None, int | None, bool] | None = None


class TopologySubsystem:
    """Encapsulates compile planning and projection-plan construction."""

    def edge_count(self, topology: SynapseTopology) -> int:
        if hasattr(topology.pre_idx, "numel"):
            return int(topology.pre_idx.numel())
        try:
            return len(topology.pre_idx)
        except TypeError:
            return 0

    def ensure_topology_meta(
        self,
        topology: SynapseTopology,
        *,
        n_pre: int | None = None,
        n_post: int | None = None,
    ) -> SynapseTopology:
        meta = dict(topology.meta) if topology.meta else {}
        updated = False

        if n_pre is not None and "n_pre" not in meta:
            meta["n_pre"] = int(n_pre)
            updated = True
        if n_post is not None and "n_post" not in meta:
            meta["n_post"] = int(n_post)
            updated = True

        if topology.delay_steps is not None and "max_delay_steps" not in meta:
            delay_steps = topology.delay_steps
            max_delay = 0
            if hasattr(delay_steps, "numel") and delay_steps.numel():
                max_val = delay_steps.detach()
                if hasattr(max_val, "max"):
                    max_val = max_val.max()
                if hasattr(max_val, "cpu"):
                    max_val = max_val.cpu()
                if hasattr(max_val, "tolist"):
                    max_list = max_val.tolist()
                    if isinstance(max_list, list):
                        scalar = max_list[0] if max_list else 0
                        max_delay = int(cast(SupportsInt, scalar))
                    else:
                        max_delay = int(cast(SupportsInt, max_list))
                else:
                    max_delay = int(max_val)
            meta["max_delay_steps"] = max_delay
            updated = True

        if updated:
            return replace(topology, meta=meta)
        return topology

    def copy_topology_weights(self, state: Any, weights: Tensor | None) -> None:
        if weights is None or not hasattr(state, "weights"):
            if weights is None and getattr(state, "bind_weights_to_topology", False):
                return
            return
        state_weights = state.weights
        if not hasattr(state_weights, "shape"):
            return
        if getattr(state, "bind_weights_to_topology", False):
            state.weights = weights
            return
        if state_weights.shape != weights.shape:
            return
        if hasattr(weights, "to"):
            weights = weights.to(device=state_weights.device, dtype=state_weights.dtype)
        state_weights.copy_(weights)

    def compile_flags_for_projection(
        self,
        proj: ProjectionSpec,
        *,
        requirements: NetworkRequirements,
        device: Any,
    ) -> ProjectionCompileFlags:
        build_edges_by_delay = False
        build_pre_adjacency = False
        build_sparse_delay_mats = False
        build_bucket_edge_mapping = False
        wants_fused_sparse = False
        wants_by_delay_sparse = False
        wants_bucket_edge_mapping = False
        wants_fused_csr: bool | None = None
        store_sparse_by_delay_override: bool | None = None
        reqs = None
        if hasattr(proj.synapse, "compilation_requirements"):
            try:
                reqs = proj.synapse.compilation_requirements()
            except Exception:
                reqs = None
        if isinstance(reqs, Mapping):
            build_edges_by_delay = _as_bool(reqs.get("needs_edges_by_delay"))
            build_pre_adjacency = _as_bool(reqs.get("needs_pre_adjacency"))
            build_sparse_delay_mats = _as_bool(reqs.get("needs_sparse_delay_mats"))
            build_bucket_edge_mapping = _as_bool(reqs.get("needs_bucket_edge_mapping"))
            wants_fused_sparse = _as_bool(reqs.get("wants_fused_sparse"))
            wants_by_delay_sparse = _as_bool(reqs.get("wants_by_delay_sparse"))
            wants_bucket_edge_mapping = _as_bool(reqs.get("wants_bucket_edge_mapping"))
            if "wants_fused_csr" in reqs:
                wants_fused_csr = _as_bool(reqs.get("wants_fused_csr"))
            if "store_sparse_by_delay" in reqs:
                store_sparse_by_delay_override = _as_bool(reqs.get("store_sparse_by_delay"))

        if build_sparse_delay_mats and not wants_fused_sparse and not wants_by_delay_sparse:
            wants_fused_sparse = True

        wants_fused_layout = _normalize_fused_layout(requirements.wants_fused_layout)
        if wants_fused_layout in {"coo", "csr"}:
            wants_fused_sparse = True
        if wants_fused_layout == "csr":
            wants_fused_csr = True
        elif wants_fused_layout == "coo":
            wants_fused_csr = False

        wants_by_delay_sparse = bool(wants_by_delay_sparse or requirements.needs_by_delay_sparse)
        wants_bucket_edge_mapping = bool(
            wants_bucket_edge_mapping or requirements.needs_bucket_edge_mapping
        )

        if wants_fused_csr is None:
            wants_fused_csr = bool(wants_fused_sparse and self.is_cpu_device(device))
        build_fused_csr = bool(wants_fused_sparse and wants_fused_csr)

        build_sparse_delay_mats = bool(
            build_sparse_delay_mats
            or wants_fused_sparse
            or wants_by_delay_sparse
            or wants_bucket_edge_mapping
            or requirements.needs_by_delay_sparse
        )
        build_bucket_edge_mapping = bool(
            build_bucket_edge_mapping
            or wants_bucket_edge_mapping
            or requirements.needs_bucket_edge_mapping
        )

        if proj.learning is not None:
            build_bucket_edge_mapping = True
        else:
            params = getattr(proj.synapse, "params", None)
            if getattr(params, "enable_sparse_updates", False):
                build_bucket_edge_mapping = True

        supports_sparse = bool(getattr(proj.learning, "supports_sparse", False))
        if proj.sparse_learning and supports_sparse:
            build_pre_adjacency = True

        if build_bucket_edge_mapping:
            build_sparse_delay_mats = True

        if store_sparse_by_delay_override:
            build_sparse_delay_mats = True
        store_sparse_by_delay = bool(build_sparse_delay_mats and wants_by_delay_sparse)
        if store_sparse_by_delay_override is not None:
            store_sparse_by_delay = bool(store_sparse_by_delay_override)
        fuse_delay_buckets = bool(
            build_sparse_delay_mats and (wants_fused_sparse or not store_sparse_by_delay)
        )

        return ProjectionCompileFlags(
            build_edges_by_delay=build_edges_by_delay,
            build_pre_adjacency=build_pre_adjacency,
            build_sparse_delay_mats=build_sparse_delay_mats,
            build_bucket_edge_mapping=build_bucket_edge_mapping,
            fuse_delay_buckets=fuse_delay_buckets,
            store_sparse_by_delay=store_sparse_by_delay,
            build_fused_csr=build_fused_csr,
        )

    def is_cpu_device(self, device: Any) -> bool:
        if device is None:
            return True
        device_type = getattr(device, "type", None)
        if isinstance(device_type, str):
            return device_type == "cpu"
        if isinstance(device, str):
            return device.lower().strip().startswith("cpu")
        return False

    def projection_network_requirements(
        self,
        proj: ProjectionSpec,
        *,
        base_requirements: NetworkRequirements,
    ) -> NetworkRequirements:
        merged = base_requirements

        reqs = None
        if hasattr(proj.synapse, "compilation_requirements"):
            try:
                reqs = proj.synapse.compilation_requirements()
            except Exception:
                reqs = None
        if isinstance(reqs, Mapping):
            wants_fused_layout = _normalize_fused_layout(_as_str(reqs.get("wants_fused_layout")))
            if wants_fused_layout == "auto" and _as_bool(reqs.get("wants_fused_csr")):
                wants_fused_layout = "csr"
            ring_strategy = _normalize_ring_strategy(_as_str(reqs.get("ring_strategy")))
            ring_dtype = _as_str(reqs.get("ring_dtype"))
            needs_by_delay_sparse = bool(
                _as_bool(reqs.get("wants_by_delay_sparse"))
                or _as_bool(reqs.get("store_sparse_by_delay"))
            )
            needs_bucket_edge_mapping = bool(
                _as_bool(reqs.get("needs_bucket_edge_mapping"))
                or _as_bool(reqs.get("wants_bucket_edge_mapping"))
            )
            merged = merged.merge(
                NetworkRequirements(
                    needs_bucket_edge_mapping=needs_bucket_edge_mapping,
                    needs_by_delay_sparse=needs_by_delay_sparse,
                    wants_fused_layout=wants_fused_layout,
                    ring_strategy=ring_strategy,
                    ring_dtype=ring_dtype,
                )
            )

        params = getattr(proj.synapse, "params", None)
        if params is not None:
            param_layout = _normalize_fused_layout(_as_str(getattr(params, "fused_layout", None)))
            param_ring_strategy = _normalize_ring_strategy(
                _as_str(getattr(params, "ring_strategy", None))
            )
            param_ring_dtype = _as_str(getattr(params, "ring_dtype", None))
            param_by_delay = _as_bool(getattr(params, "store_sparse_by_delay", False))
            merged = merged.merge(
                NetworkRequirements(
                    needs_by_delay_sparse=param_by_delay,
                    wants_fused_layout=param_layout,
                    ring_strategy=param_ring_strategy,
                    ring_dtype=param_ring_dtype,
                )
            )

        return merged

    def ensure_learning_bucket_mapping(self, proj: ProjectionSpec, topology: SynapseTopology) -> None:
        if proj.learning is None:
            return
        meta = topology.meta or {}
        if (
            meta.get("edge_bucket_comp") is None
            or meta.get("edge_bucket_delay") is None
            or meta.get("edge_bucket_pos") is None
        ):
            raise RuntimeError(
                f"Projection {proj.name} requires edge bucket mappings for learning, "
                "but they were not built. Enable build_bucket_edge_mapping."
            )

    def compile_topology_with_thread_limits(
        self,
        topology: SynapseTopology,
        compile_kwargs: Mapping[str, Any],
        torch_threads: int,
    ) -> SynapseTopology:
        from biosnn.core.torch_utils import require_torch

        torch = require_torch()
        if torch_threads <= 0 or not hasattr(torch, "set_num_threads"):
            return compile_topology(topology, **compile_kwargs)
        self._enter_torch_thread_limits(torch, torch_threads)
        try:
            return compile_topology(topology, **compile_kwargs)
        finally:
            self._exit_torch_thread_limits(torch)

    def _enter_torch_thread_limits(self, torch: Any, torch_threads: int) -> None:
        global _TORCH_THREAD_LIMIT_ACTIVE, _TORCH_THREAD_LIMIT_STATE
        with _TORCH_THREAD_LIMIT_LOCK:
            if _TORCH_THREAD_LIMIT_ACTIVE == 0:
                prev_threads = torch.get_num_threads() if hasattr(torch, "get_num_threads") else None
                prev_interop = (
                    torch.get_num_interop_threads()
                    if hasattr(torch, "get_num_interop_threads")
                    else None
                )
                interop_set = False
                torch.set_num_threads(int(torch_threads))
                if hasattr(torch, "set_num_interop_threads"):
                    try:
                        torch.set_num_interop_threads(1)
                        interop_set = True
                    except RuntimeError:
                        prev_interop = None
                _TORCH_THREAD_LIMIT_STATE = (prev_threads, prev_interop, interop_set)
            _TORCH_THREAD_LIMIT_ACTIVE += 1

    def _exit_torch_thread_limits(self, torch: Any) -> None:
        global _TORCH_THREAD_LIMIT_ACTIVE, _TORCH_THREAD_LIMIT_STATE
        with _TORCH_THREAD_LIMIT_LOCK:
            _TORCH_THREAD_LIMIT_ACTIVE = max(0, _TORCH_THREAD_LIMIT_ACTIVE - 1)
            if _TORCH_THREAD_LIMIT_ACTIVE == 0 and _TORCH_THREAD_LIMIT_STATE is not None:
                prev_threads, prev_interop, interop_set = _TORCH_THREAD_LIMIT_STATE
                if prev_threads is not None:
                    torch.set_num_threads(prev_threads)
                if (
                    interop_set
                    and prev_interop is not None
                    and hasattr(torch, "set_num_interop_threads")
                ):
                    with contextlib.suppress(RuntimeError):
                        torch.set_num_interop_threads(prev_interop)
                _TORCH_THREAD_LIMIT_STATE = None

    def resolve_parallel_compile(
        self,
        mode: str,
        job_count: int,
        device: Any,
        workers: int | None,
    ) -> tuple[bool, int]:
        if device is not None and getattr(device, "type", None) == "cuda":
            return False, 1
        if job_count < 2:
            return False, 1
        mode_norm = mode.lower().strip()
        cpu_count = os.cpu_count() or 1
        if mode_norm == "off":
            return False, 1
        if mode_norm == "auto" and cpu_count < 4:
            return False, 1
        max_workers = workers if workers is not None else min(cpu_count, job_count)
        max_workers = max(1, min(int(max_workers), job_count))
        return True, max_workers

    def check_ring_buffer_budget(
        self,
        *,
        proj_name: str,
        topology: SynapseTopology,
        max_ring_mib: float | None,
        requires_ring: bool,
    ) -> None:
        if not requires_ring:
            return
        if max_ring_mib is None:
            return
        try:
            max_mib = float(max_ring_mib)
        except (TypeError, ValueError):
            return
        if max_mib <= 0:
            return
        meta = topology.meta or {}
        est_mib = meta.get("estimated_ring_mib")
        if est_mib is None:
            return
        try:
            est_val = float(est_mib)
        except (TypeError, ValueError):
            return
        if est_val <= max_mib:
            return
        ring_len = meta.get("ring_len", "unknown")
        n_post = meta.get("n_post", "unknown")
        if topology.weights is not None and hasattr(topology.weights, "dtype"):
            dtype = str(topology.weights.dtype)
        else:
            dtype = str(meta.get("dtype", "unknown"))
        raise RuntimeError(
            f"Projection '{proj_name}' ring buffer estimate {est_val:.2f} MiB exceeds "
            f"max_ring_mib={max_mib:.2f} MiB. "
            f"ring_len={ring_len}, n_post={n_post}, dtype={dtype}. "
            "Reduce max_delay_steps, reduce n_post, use smaller delays, or choose a smaller dtype "
            "to lower ring buffer memory, or increase max_ring_mib."
        )

    def build_projection_plan(
        self,
        proj: ProjectionSpec,
        *,
        fast_mode: bool,
        compiled_mode: bool,
    ) -> ProjectionPlan:
        meta = proj.topology.meta or {}
        needs_bucket = (
            meta.get("edge_bucket_comp") is not None
            and meta.get("edge_bucket_delay") is not None
            and meta.get("edge_bucket_pos") is not None
        )
        use_fused = (
            meta.get("fused_W_by_comp") is not None
            and meta.get("fused_W_delays_by_comp") is not None
            and meta.get("fused_W_n_post_by_comp") is not None
        )
        return ProjectionPlan(
            name=proj.name,
            pre_name=proj.pre,
            post_name=proj.post,
            synapse=proj.synapse,
            topology=proj.topology,
            learning_enabled=proj.learning is not None,
            needs_bucket_mapping=bool(needs_bucket),
            use_fused_sparse=bool(use_fused),
            fast_mode=fast_mode,
            compiled_mode=compiled_mode,
        )

    def build_compiled_network_plan(
        self,
        *,
        compiled_projections: list[ProjectionSpec],
        fast_mode: bool,
        compiled_mode: bool,
    ) -> CompiledNetworkPlan:
        plans = [
            self.build_projection_plan(
                proj,
                fast_mode=fast_mode,
                compiled_mode=compiled_mode,
            )
            for proj in compiled_projections
        ]
        plan_index = {plan.name: plan for plan in plans}
        return CompiledNetworkPlan(
            compiled_projections=compiled_projections,
            projection_plans=plans,
            projection_index=plan_index,
        )

    def compile_projection_jobs(
        self,
        *,
        projection_specs: Sequence[ProjectionSpec],
        jobs: Sequence[tuple[int, ProjectionSpec, SynapseTopology, dict[str, Any], bool]],
        parallel_mode: str,
        parallel_workers: int | None,
        parallel_torch_threads: int,
        device: Any,
        max_ring_mib: float | None,
        executor_factory: Any | None = None,
    ) -> list[ProjectionSpec]:
        compiled_specs: list[ProjectionSpec | None] = [None] * len(projection_specs)
        use_parallel, max_workers = self.resolve_parallel_compile(
            parallel_mode,
            len(jobs),
            device,
            parallel_workers,
        )
        if not use_parallel:
            for idx, proj, topology, compile_kwargs, requires_ring in jobs:
                compiled_topology = compile_topology(topology, **compile_kwargs)
                self.check_ring_buffer_budget(
                    proj_name=proj.name,
                    topology=compiled_topology,
                    max_ring_mib=max_ring_mib,
                    requires_ring=requires_ring,
                )
                self.ensure_learning_bucket_mapping(proj, compiled_topology)
                compiled_specs[idx] = replace(proj, topology=compiled_topology)
        else:
            executor_cls = executor_factory or ThreadPoolExecutor
            with executor_cls(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self.compile_topology_with_thread_limits,
                        topology,
                        compile_kwargs,
                        parallel_torch_threads,
                    ): (idx, proj, requires_ring)
                    for idx, proj, topology, compile_kwargs, requires_ring in jobs
                }
                for future, info in futures.items():
                    compiled_topology = future.result()
                    idx, proj, requires_ring = info
                    self.check_ring_buffer_budget(
                        proj_name=proj.name,
                        topology=compiled_topology,
                        max_ring_mib=max_ring_mib,
                        requires_ring=requires_ring,
                    )
                    self.ensure_learning_bucket_mapping(proj, compiled_topology)
                    compiled_specs[idx] = replace(proj, topology=compiled_topology)

        return [cast(ProjectionSpec, spec) for spec in compiled_specs]

    def validate_specs(
        self,
        populations: Sequence[PopulationSpec],
        projections: Sequence[ProjectionSpec],
    ) -> None:
        names = {spec.name for spec in populations}
        if len(names) != len(populations):
            raise ValueError("Population names must be unique")
        for proj in projections:
            if proj.pre not in names or proj.post not in names:
                raise ValueError(f"Projection {proj.name} references unknown population")


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


def _normalize_fused_layout(value: str | None) -> str:
    if value in {"auto", "coo", "csr"}:
        return value
    return "auto"


def _normalize_ring_strategy(value: str | None) -> str:
    if value in {"dense", "event_bucketed", "event_list_proto"}:
        return value
    return "dense"
