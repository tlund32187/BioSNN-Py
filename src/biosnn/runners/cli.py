"""CLI entrypoint for running demo simulations and opening the dashboard."""

from __future__ import annotations

import argparse
import csv
import os
import socket
import time
import webbrowser
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import quote

from biosnn.contracts.monitors import IMonitor
from biosnn.core.torch_utils import require_torch
from biosnn.experiments.demo_minimal import DemoMinimalConfig, run_demo_minimal
from biosnn.experiments.demo_network import DemoNetworkConfig, build_network_demo
from biosnn.experiments.demo_registry import (
    ALLOWED_DEMOS,
    feature_flags_for_run_spec,
    run_spec_from_cli_args,
)
from biosnn.experiments.demo_runner import run_demo_from_spec
from biosnn.experiments.demo_types import DemoModelSpec, DemoRuntimeConfig
from biosnn.experiments.demos import FEATURE_DEMO_BUILDERS, FeatureDemoConfig, FeatureDemoName
from biosnn.io.export.run_manifest import write_run_config, write_run_features, write_run_status
from biosnn.learning.homeostasis import HomeostasisScope, RateEmaThresholdHomeostasisConfig
from biosnn.runners.dashboard_server import start_dashboard_server
from biosnn.tasks.logic_gates import (
    LogicGateNeuronModel,
    LogicGateRunConfig,
    run_logic_gate,
    run_logic_gate_curriculum,
    run_logic_gate_curriculum_engine,
    run_logic_gate_engine,
)

LOGIC_DEMO_TO_GATE: dict[str, str] = {
    "logic_and": "and",
    "logic_or": "or",
    "logic_xor": "xor",
    "logic_nand": "nand",
    "logic_nor": "nor",
    "logic_xnor": "xnor",
}
LOGIC_CURRICULUM_DEMO = "logic_curriculum"


def main() -> None:
    args = _parse_args()
    torch_threads = _parse_thread_setting(args.torch_threads)
    torch_interop = _parse_thread_setting(args.torch_interop_threads)
    if args.set_omp_env:
        _apply_threading_env(torch_threads, torch_interop)
    torch = require_torch()
    _apply_threading_torch(torch, torch_threads, torch_interop)

    repo_root = Path(__file__).resolve().parents[3]
    artifacts_root = (
        Path(args.artifacts_dir).expanduser().resolve()
        if args.artifacts_dir
        else (repo_root / "artifacts").resolve()
    )
    run_dir = _make_run_dir(artifacts_root, args.run_id)
    device = _resolve_device(torch, args.device)
    _apply_large_network_safety_overrides(args=args, device=device)
    _validate_ring_dtype_for_device(torch=torch, ring_dtype=args.ring_dtype, device=device)
    steps = args.steps
    dt = args.dt

    print(f"Run dir: {run_dir}")
    print(f"Device: {device}")
    print(f"Demo: {args.demo}")

    mode = args.mode

    run_spec: dict[str, Any] | None = None
    if args.demo in ALLOWED_DEMOS:
        run_spec = run_spec_from_cli_args(args=args, device=device)
        run_spec["run_id"] = run_dir.name
        run_spec["run_dir"] = _run_dir_web_path(repo_root, run_dir)
        write_run_config(run_dir, run_spec)
        write_run_features(run_dir, feature_flags_for_run_spec(run_spec))
        write_run_status(
            run_dir,
            {
                "run_id": run_dir.name,
                "state": "running",
                "started_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            },
        )

    try:
        logic_result: dict[str, Any] | None = None
        if args.demo == "minimal":
            run_demo_minimal(
                DemoMinimalConfig(
                    out_dir=run_dir,
                    mode=mode,
                    n_neurons=args.n,
                    p_connect=args.p,
                    steps=steps,
                    dt=dt,
                    seed=args.seed,
                    device=device,
                    max_ring_mib=args.max_ring_mib,
                    profile=args.profile,
                    profile_steps=args.profile_steps,
                    allow_cuda_monitor_sync=args.allow_cuda_monitor_sync,
                    monitor_safe_defaults=bool(args.monitor_safe_defaults),
                    monitor_neuron_sample=args.monitor_neuron_sample,
                    monitor_edge_sample=args.monitor_edge_sample,
                )
            )
        else:
            if args.demo in LOGIC_DEMO_TO_GATE:
                logic_result = _run_logic_gate_demo_from_cli(
                    args=args,
                    run_dir=run_dir,
                    steps=int(steps),
                    dt=float(dt),
                    device=device,
                    run_spec=run_spec,
                )
            elif args.demo == LOGIC_CURRICULUM_DEMO:
                logic_result = _run_logic_curriculum_demo_from_cli(
                    args=args,
                    run_dir=run_dir,
                    steps=int(steps),
                    dt=float(dt),
                    device=device,
                    run_spec=run_spec,
                )
            else:
                builders = _demo_registry()
                builder = builders.get(args.demo)
                if builder is None:
                    raise ValueError(f"Unsupported demo '{args.demo}'")
                model_spec, runtime_cfg, monitors = builder(
                    args=args,
                    run_dir=run_dir,
                    mode=cast(ModeName, mode),
                    steps=int(steps),
                    dt=float(dt),
                    device=device,
                )
                run_demo_from_spec(model_spec, runtime_cfg, monitors)
        if run_spec is not None and logic_result is not None:
            backend = str(logic_result.get("logic_backend", run_spec.get("logic_backend", "harness"))).strip().lower()
            if backend in {"harness", "engine"}:
                run_spec["logic_backend"] = backend
            write_run_config(run_dir, run_spec)
            write_run_features(run_dir, feature_flags_for_run_spec(run_spec))
    except Exception as exc:
        if run_spec is not None:
            write_run_status(
                run_dir,
                {
                    "run_id": run_dir.name,
                    "state": "error",
                    "last_error": str(exc),
                    "finished_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
                },
            )
        raise
    else:
        if run_spec is not None:
            write_run_status(
                run_dir,
                {
                    "run_id": run_dir.name,
                    "state": "done",
                    "last_error": None,
                    "finished_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
                },
            )

    if args.no_server:
        print("No-server mode selected; skipping dashboard server.")
        return

    if not _should_launch_dashboard(mode):
        print("Fast mode selected; skipping dashboard server.")
        return

    port = _find_port(args.port)
    server = start_dashboard_server(
        repo_root=repo_root,
        port=port,
        artifacts_dir=artifacts_root,
        initial_run_dir=run_dir,
        initial_state="done",
    )
    url = _build_dashboard_url(port, run_dir, repo_root, args.refresh_ms)
    print(f"Dashboard URL: {url}")

    if not args.no_open:
        webbrowser.open(url)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        return
    finally:
        controller = getattr(server, "controller", None)
        if controller is not None and hasattr(controller, "shutdown"):
            controller.shutdown()
        server.shutdown()
        server.server_close()


ModeName = Literal["dashboard", "fast"]
ParallelCompileMode = Literal["auto", "on", "off"]
DemoBuildResult = tuple[DemoModelSpec, DemoRuntimeConfig, list[IMonitor]]
DemoBuilder = Callable[..., DemoBuildResult]


def _demo_registry() -> Mapping[str, DemoBuilder]:
    return {
        "network": _build_network_demo_from_cli,
        "vision": _build_vision_demo_from_cli,
        "pruning_sparse": _build_pruning_sparse_demo_from_cli,
        "neurogenesis_sparse": _build_neurogenesis_sparse_demo_from_cli,
        "propagation_impulse": _build_propagation_impulse_demo_from_cli,
        "delay_impulse": _build_delay_impulse_demo_from_cli,
        "learning_gate": _build_learning_gate_demo_from_cli,
        "dopamine_plasticity": _build_dopamine_plasticity_demo_from_cli,
    }


def _run_logic_gate_demo_from_cli(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    steps: int,
    dt: float,
    device: str,
    run_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    spec = cast(Mapping[str, Any], run_spec) if isinstance(run_spec, Mapping) else {}
    demo_gate = LOGIC_DEMO_TO_GATE.get(str(args.demo).strip().lower(), "and")
    gate = str(spec.get("logic_gate", getattr(args, "logic_gate", demo_gate) or demo_gate)).strip().lower()
    if gate not in {"and", "or", "xor", "nand", "nor", "xnor"}:
        gate = demo_gate
    learning_mode = str(spec.get("logic_learning_mode", getattr(args, "logic_learning_mode", "rstdp"))).strip().lower()
    if learning_mode not in {"rstdp", "surrogate", "none"}:
        learning_mode = "rstdp"
    sampling_method = str(spec.get("logic_sampling_method", getattr(args, "logic_sampling_method", "sequential"))).strip().lower()
    if sampling_method not in {"sequential", "random_balanced"}:
        sampling_method = "sequential"
    logic_backend = str(spec.get("logic_backend", "harness")).strip().lower()
    if logic_backend not in {"harness", "engine"}:
        logic_backend = "harness"
    sim_steps_per_trial = max(
        1,
        int(spec.get("logic_sim_steps_per_trial", getattr(args, "logic_sim_steps_per_trial", 10))),
    )
    neuron_model = str(spec.get("logic_neuron_model", _resolve_logic_neuron_model_arg(args))).strip().lower()
    if neuron_model not in {"adex_3c", "lif_3c"}:
        neuron_model = "adex_3c"
    logic_cfg = LogicGateRunConfig(
        gate=gate,
        seed=123 if args.seed is None else int(args.seed),
        steps=int(steps),
        dt=float(dt),
        sim_steps_per_trial=sim_steps_per_trial,
        device=device,
        learning_mode=cast(Any, learning_mode),
        engine_learning_rule=cast(Any, _resolve_logic_engine_learning_rule(spec)),
        learning_modulator_kind=cast(Any, _resolve_logic_learning_modulator_kind(spec)),
        neuron_model=cast(LogicGateNeuronModel, neuron_model),
        debug=bool(spec.get("logic_debug", getattr(args, "logic_debug", False))),
        debug_every=max(1, int(spec.get("logic_debug_every", getattr(args, "logic_debug_every", 25)))),
        export_every=25,
        sampling_method=cast(Any, sampling_method),
        out_dir=run_dir,
        advanced_synapse=cast(Any, spec.get("advanced_synapse", {})),
        modulators=cast(Any, spec.get("modulators", {})),
        wrapper=cast(Any, spec.get("wrapper", {})),
        excitability_modulation=cast(Any, spec.get("excitability_modulation", {})),
    )

    if logic_backend == "engine":
        result = run_logic_gate_engine(logic_cfg, spec)
    else:
        result = run_logic_gate(logic_cfg)
    result = dict(result)
    result["logic_backend"] = logic_backend
    _write_logic_metrics_csv_from_eval(run_dir)
    print(
        "[logic-gate] "
        f"backend={logic_backend} "
        f"gate={result.get('gate')} mode={result.get('learning_mode')} "
        f"eval_acc={result.get('eval_accuracy')} passed={result.get('passed')}"
    )
    return result


def _run_logic_curriculum_demo_from_cli(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    steps: int,
    dt: float,
    device: str,
    run_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    spec = cast(Mapping[str, Any], run_spec) if isinstance(run_spec, Mapping) else {}
    gates_raw = str(spec.get("logic_curriculum_gates", getattr(args, "logic_curriculum_gates", "or,and,nor,nand,xor,xnor")))
    gate_list = [token.strip().lower() for token in gates_raw.split(",") if token.strip()]
    if not gate_list:
        gate_list = ["or", "and", "nor", "nand", "xor", "xnor"]

    learning_mode = str(spec.get("logic_learning_mode", getattr(args, "logic_learning_mode", "rstdp"))).strip().lower()
    logic_backend = str(spec.get("logic_backend", "harness")).strip().lower()
    if logic_backend not in {"harness", "engine"}:
        logic_backend = "harness"
    if logic_backend == "harness" and learning_mode != "rstdp":
        print(
            f"[logic-curriculum] overriding learning_mode={learning_mode!r} to 'rstdp' "
            "for persistent cross-gate learning."
        )
        learning_mode = "rstdp"

    sampling_method = str(spec.get("logic_sampling_method", getattr(args, "logic_sampling_method", "sequential"))).strip().lower()
    if sampling_method not in {"sequential", "random_balanced"}:
        sampling_method = "sequential"
    replay_ratio = float(spec.get("logic_curriculum_replay_ratio", getattr(args, "logic_curriculum_replay_ratio", 0.35)))
    if replay_ratio < 0.0:
        replay_ratio = 0.0
    if replay_ratio > 1.0:
        replay_ratio = 1.0

    sim_steps_per_trial = max(
        1,
        int(spec.get("logic_sim_steps_per_trial", getattr(args, "logic_sim_steps_per_trial", 10))),
    )
    neuron_model = str(spec.get("logic_neuron_model", _resolve_logic_neuron_model_arg(args))).strip().lower()
    if neuron_model not in {"adex_3c", "lif_3c"}:
        neuron_model = "adex_3c"
    logic_cfg = LogicGateRunConfig(
        gate=gate_list[0],
        seed=123 if args.seed is None else int(args.seed),
        steps=int(steps),
        dt=float(dt),
        sim_steps_per_trial=sim_steps_per_trial,
        device=device,
        learning_mode=cast(Any, learning_mode),
        engine_learning_rule=cast(Any, _resolve_logic_engine_learning_rule(spec)),
        learning_modulator_kind=cast(Any, _resolve_logic_learning_modulator_kind(spec)),
        neuron_model=cast(LogicGateNeuronModel, neuron_model),
        debug=bool(spec.get("logic_debug", getattr(args, "logic_debug", False))),
        debug_every=max(1, int(spec.get("logic_debug_every", getattr(args, "logic_debug_every", 25)))),
        export_every=25,
        sampling_method=cast(Any, sampling_method),
        out_dir=run_dir,
        advanced_synapse=cast(Any, spec.get("advanced_synapse", {})),
        modulators=cast(Any, spec.get("modulators", {})),
        wrapper=cast(Any, spec.get("wrapper", {})),
        excitability_modulation=cast(Any, spec.get("excitability_modulation", {})),
    )

    if logic_backend == "engine":
        result = run_logic_gate_curriculum_engine(logic_cfg, spec)
    else:
        result = run_logic_gate_curriculum(
            logic_cfg,
            gates=gate_list,
            phase_steps=int(steps),
            replay_ratio=replay_ratio,
        )
    result = dict(result)
    result["logic_backend"] = logic_backend
    _write_logic_metrics_csv_from_eval(run_dir)
    print(
        "[logic-curriculum] "
        f"backend={logic_backend} "
        f"gates={result.get('gates')} final_eval={result.get('final_eval_by_gate')}"
    )
    return result


def _resolve_logic_engine_learning_rule(spec: Mapping[str, Any]) -> str:
    learning = cast(Mapping[str, Any], spec.get("learning", {}))
    rule = str(learning.get("rule", "three_factor_elig_stdp")).strip().lower()
    if rule in {"three_factor_elig_stdp", "three_factor_eligibility_stdp"}:
        return "three_factor_elig_stdp"
    if rule in {"rstdp", "rstdp_elig", "rstdp_eligibility"}:
        return "rstdp_elig"
    if rule in {"none", "surrogate"}:
        return "none"
    return "three_factor_elig_stdp"


def _resolve_logic_learning_modulator_kind(spec: Mapping[str, Any]) -> str:
    learning = cast(Mapping[str, Any], spec.get("learning", {}))
    token = str(learning.get("modulator_kind", "dopamine")).strip().lower()
    aliases = {
        "da": "dopamine",
        "ach": "acetylcholine",
        "na": "noradrenaline",
        "5ht": "serotonin",
    }
    token = aliases.get(token, token)
    if token in {"dopamine", "acetylcholine", "noradrenaline", "serotonin"}:
        return token
    return "dopamine"


def _write_logic_metrics_csv_from_eval(run_dir: Path) -> None:
    eval_path = run_dir / "eval.csv"
    if not eval_path.exists():
        return
    rows: list[dict[str, str]] = []
    with eval_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            trial_raw = (row.get("trial") or "").strip()
            eval_raw = (row.get("eval_accuracy") or "").strip()
            sample_raw = (row.get("sample_accuracy") or eval_raw).strip()
            global_raw = (row.get("sample_accuracy_global") or "").strip()
            global_eval_raw = (row.get("global_eval_accuracy") or "").strip()
            rolling_raw = (row.get("trial_acc_rolling") or "").strip()
            loss_raw = (row.get("loss") or "").strip()
            if not trial_raw:
                continue
            # Curriculum rows can include global_eval_accuracy (true across-gate
            # evaluation) and sample_accuracy_global (rolling train sampling
            # accuracy). Prefer global_eval_accuracy for dashboard curves.
            train_out = global_eval_raw or global_raw or sample_raw
            eval_out = global_eval_raw or global_raw or eval_raw
            loss_out = loss_raw
            if not loss_out and eval_out:
                try:
                    loss_out = str(max(0.0, 1.0 - float(eval_out)))
                except ValueError:
                    loss_out = ""
            rows.append(
                {
                    "step": trial_raw,
                    "t": trial_raw,
                    "phase": (row.get("phase") or "").strip(),
                    "gate": (row.get("gate") or "").strip(),
                    "train_accuracy": train_out,
                    "eval_accuracy": eval_out,
                    "gate_eval_accuracy": eval_raw,
                    "sample_accuracy_global": global_raw,
                    "global_eval_accuracy": global_eval_raw,
                    "loss": loss_out,
                    "trial_acc_rolling": rolling_raw,
                }
            )
    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "t",
                "phase",
                "gate",
                "train_accuracy",
                "eval_accuracy",
                "gate_eval_accuracy",
                "sample_accuracy_global",
                "global_eval_accuracy",
                "loss",
                "trial_acc_rolling",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_network_demo_from_cli(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    mode: ModeName,
    steps: int,
    dt: float,
    device: str,
) -> DemoBuildResult:
    cfg = _build_network_config_from_args(
        args=args,
        run_dir=run_dir,
        mode=mode,
        steps=steps,
        dt=dt,
        device=device,
    )
    return build_network_demo(cfg)


def _build_propagation_impulse_demo_from_cli(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    mode: ModeName,
    steps: int,
    dt: float,
    device: str,
) -> DemoBuildResult:
    return _build_feature_demo_from_cli(
        demo_name="propagation_impulse",
        args=args,
        run_dir=run_dir,
        mode=mode,
        steps=steps,
        dt=dt,
        device=device,
    )


def _build_vision_demo_from_cli(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    mode: ModeName,
    steps: int,
    dt: float,
    device: str,
) -> DemoBuildResult:
    cfg = _build_network_config_from_args(
        args=args,
        run_dir=run_dir,
        mode=mode,
        steps=steps,
        dt=dt,
        device=device,
    )
    cfg.enable_vision_monitors = mode == "dashboard"
    return build_network_demo(cfg)


def _build_pruning_sparse_demo_from_cli(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    mode: ModeName,
    steps: int,
    dt: float,
    device: str,
) -> DemoBuildResult:
    pruning_steps = 5000 if int(steps) == 500 else int(steps)
    cfg = _build_network_config_from_args(
        args=args,
        run_dir=run_dir,
        mode=mode,
        steps=pruning_steps,
        dt=dt,
        device=device,
    )
    cfg.enable_pruning = True
    cfg.pruning_verbose = True
    cfg.input_to_relay_p = max(float(cfg.input_to_relay_p), 0.6)
    cfg.relay_to_hidden_p = max(float(cfg.relay_to_hidden_p), 0.35)
    cfg.hidden_to_output_p = max(float(cfg.hidden_to_output_p), 0.35)
    print(
        "Pruning demo settings: "
        f"steps={cfg.steps}, "
        f"interval={cfg.prune_interval_steps}, "
        f"w_min={cfg.w_min}, "
        f"usage_min={cfg.usage_min}, "
        f"k_min_out={cfg.k_min_out}, "
        f"k_min_in={cfg.k_min_in}, "
        f"max_fraction={cfg.max_prune_fraction_per_interval}"
    )
    return build_network_demo(cfg)


def _build_neurogenesis_sparse_demo_from_cli(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    mode: ModeName,
    steps: int,
    dt: float,
    device: str,
) -> DemoBuildResult:
    neuro_steps = 2000 if int(steps) == 500 else int(steps)
    cfg = _build_network_config_from_args(
        args=args,
        run_dir=run_dir,
        mode=mode,
        steps=neuro_steps,
        dt=dt,
        device=device,
    )
    cfg.enable_neurogenesis = True
    cfg.neurogenesis_verbose = True
    cfg.input_drive = min(float(cfg.input_drive), 0.35)
    print(
        "Neurogenesis demo settings: "
        f"steps={cfg.steps}, "
        f"interval={cfg.growth_interval_steps}, "
        f"add_neurons={cfg.add_neurons_per_event}, "
        f"newborn_mult={cfg.newborn_plasticity_multiplier}, "
        f"newborn_duration={cfg.newborn_duration_steps}, "
        f"max_total={cfg.max_total_neurons}"
    )
    return build_network_demo(cfg)


def _build_delay_impulse_demo_from_cli(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    mode: ModeName,
    steps: int,
    dt: float,
    device: str,
) -> DemoBuildResult:
    return _build_feature_demo_from_cli(
        demo_name="delay_impulse",
        args=args,
        run_dir=run_dir,
        mode=mode,
        steps=steps,
        dt=dt,
        device=device,
    )


def _build_learning_gate_demo_from_cli(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    mode: ModeName,
    steps: int,
    dt: float,
    device: str,
) -> DemoBuildResult:
    return _build_feature_demo_from_cli(
        demo_name="learning_gate",
        args=args,
        run_dir=run_dir,
        mode=mode,
        steps=steps,
        dt=dt,
        device=device,
    )


def _build_dopamine_plasticity_demo_from_cli(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    mode: ModeName,
    steps: int,
    dt: float,
    device: str,
) -> DemoBuildResult:
    return _build_feature_demo_from_cli(
        demo_name="dopamine_plasticity",
        args=args,
        run_dir=run_dir,
        mode=mode,
        steps=steps,
        dt=dt,
        device=device,
    )


def _build_feature_demo_from_cli(
    *,
    demo_name: str,
    args: argparse.Namespace,
    run_dir: Path,
    mode: ModeName,
    steps: int,
    dt: float,
    device: str,
) -> DemoBuildResult:
    if demo_name not in FEATURE_DEMO_BUILDERS:
        raise ValueError(f"Unknown feature demo '{demo_name}'")
    typed_demo_name = cast(FeatureDemoName, demo_name)
    cfg = _build_feature_config_from_args(
        args=args,
        run_dir=run_dir,
        mode=mode,
        steps=steps,
        dt=dt,
        device=device,
    )
    builder = cast(Callable[[FeatureDemoConfig], DemoBuildResult], FEATURE_DEMO_BUILDERS[typed_demo_name])
    return builder(cfg)


def _build_feature_config_from_args(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    mode: ModeName,
    steps: int,
    dt: float,
    device: str,
) -> FeatureDemoConfig:
    return FeatureDemoConfig(
        out_dir=run_dir,
        mode=mode,
        steps=steps,
        dt=dt,
        seed=args.seed,
        device=device,
        max_ring_mib=args.max_ring_mib,
        profile=args.profile,
        profile_steps=args.profile_steps,
        allow_cuda_monitor_sync=args.allow_cuda_monitor_sync,
        parallel_compile=cast(ParallelCompileMode, args.parallel_compile),
        parallel_compile_workers=_parse_thread_setting(args.parallel_compile_workers),
        parallel_compile_torch_threads=int(args.parallel_compile_torch_threads),
        delay_steps=int(args.delay_steps),
        learning_lr=float(args.learning_lr),
        da_amount=float(args.da_amount),
        da_step=int(args.da_step),
        fused_layout=cast(Literal["auto", "coo", "csr"], args.fused_layout),
        ring_dtype=cast(str | None, args.ring_dtype),
        ring_strategy=cast(Literal["dense", "event_bucketed"], args.ring_strategy),
        store_sparse_by_delay=cast(bool | None, args.store_sparse_by_delay),
    )


def _build_network_config_from_args(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    mode: ModeName,
    steps: int,
    dt: float,
    device: str,
) -> DemoNetworkConfig:
    input_to_relay_p = args.input_to_relay_p
    relay_to_hidden_p = args.relay_to_hidden_p
    hidden_to_output_p = args.hidden_to_output_p
    if args.p_in_hidden is not None:
        input_to_relay_p = args.p_in_hidden
        relay_to_hidden_p = args.p_in_hidden
    if args.p_hidden_out is not None:
        hidden_to_output_p = args.p_hidden_out
    return DemoNetworkConfig(
        out_dir=run_dir,
        mode=mode,
        steps=steps,
        dt=dt,
        seed=args.seed,
        device=device,
        max_ring_mib=args.max_ring_mib,
        profile=args.profile,
        profile_steps=args.profile_steps,
        allow_cuda_monitor_sync=args.allow_cuda_monitor_sync,
        parallel_compile=cast(ParallelCompileMode, args.parallel_compile),
        parallel_compile_workers=_parse_thread_setting(args.parallel_compile_workers),
        parallel_compile_torch_threads=int(args.parallel_compile_torch_threads),
        monitor_safe_defaults=bool(args.monitor_safe_defaults),
        monitor_neuron_sample=args.monitor_neuron_sample,
        monitor_edge_sample=args.monitor_edge_sample,
        n_in=args.n_in,
        n_hidden=args.n_hidden,
        n_out=args.n_out,
        input_pops=args.input_pops,
        input_depth=args.input_depth,
        hidden_layers=args.hidden_layers,
        hidden_pops_per_layer=args.hidden_pops_per_layer,
        output_pops=args.output_pops,
        input_cross=args.input_cross,
        input_to_relay_p=input_to_relay_p,
        input_to_relay_weight_scale=args.input_to_relay_weight_scale,
        relay_to_hidden_p=relay_to_hidden_p,
        relay_to_hidden_weight_scale=args.relay_to_hidden_weight_scale,
        hidden_to_output_p=hidden_to_output_p,
        hidden_to_output_weight_scale=args.hidden_to_output_weight_scale,
        input_skip_to_hidden=args.input_skip_to_hidden,
        input_skip_p=args.input_skip_p,
        input_skip_weight_scale=args.input_skip_weight_scale,
        relay_cross=args.relay_cross,
        relay_cross_p=args.relay_cross_p,
        relay_cross_weight_scale=args.relay_cross_weight_scale,
        relay_lateral=args.relay_lateral,
        hidden_lateral=args.hidden_lateral,
        weight_init=args.weight_init,
        enable_homeostasis=bool(args.enable_homeostasis),
        homeostasis=RateEmaThresholdHomeostasisConfig(
            alpha=float(args.homeostasis_alpha),
            eta=float(args.homeostasis_eta),
            r_target=float(args.homeostasis_r_target),
            clamp_min=float(args.homeostasis_clamp_min),
            clamp_max=float(args.homeostasis_clamp_max),
            scope=cast(HomeostasisScope, args.homeostasis_scope),
        ),
        homeostasis_export_every=int(args.homeostasis_export_every),
        enable_pruning=bool(args.enable_pruning),
        prune_interval_steps=max(1, int(args.prune_interval_steps)),
        usage_alpha=float(args.prune_usage_alpha),
        w_min=float(args.prune_w_min),
        usage_min=float(args.prune_usage_min),
        k_min_out=max(0, int(args.prune_k_min_out)),
        k_min_in=max(0, int(args.prune_k_min_in)),
        max_prune_fraction_per_interval=float(args.prune_max_fraction),
        pruning_verbose=bool(args.prune_verbose),
        enable_neurogenesis=bool(args.enable_neurogenesis),
        growth_interval_steps=max(1, int(args.growth_interval_steps)),
        add_neurons_per_event=max(1, int(args.add_neurons_per_event)),
        newborn_plasticity_multiplier=float(args.newborn_plasticity_multiplier),
        newborn_duration_steps=max(1, int(args.newborn_duration_steps)),
        max_total_neurons=max(1, int(args.max_total_neurons)),
        neurogenesis_verbose=bool(args.neurogenesis_verbose),
        feedforward_delay_from_distance=args.feedforward_delay_from_distance,
        feedforward_delay_base_velocity=args.feedforward_delay_base_velocity,
        feedforward_delay_myelin_scale=args.feedforward_delay_myelin_scale,
        feedforward_delay_myelin_mean=args.feedforward_delay_myelin_mean,
        feedforward_delay_myelin_std=args.feedforward_delay_myelin_std,
        feedforward_delay_distance_scale=args.feedforward_delay_distance_scale,
        feedforward_delay_min=args.feedforward_delay_min,
        feedforward_delay_max=args.feedforward_delay_max,
        feedforward_delay_use_ceil=args.feedforward_delay_use_ceil,
        input_drive=args.input_drive,
        drive_monitor=args.drive_monitor,
        synapse_backend=cast(Literal["spmm_fused", "event_driven"], args.synapse_backend),
        fused_layout=cast(Literal["auto", "coo", "csr"], args.fused_layout),
        ring_dtype=cast(str | None, args.ring_dtype),
        receptor_state_dtype=cast(str | None, args.receptor_state_dtype),
        ring_strategy=cast(Literal["dense", "event_bucketed"], args.ring_strategy),
        store_sparse_by_delay=cast(bool | None, args.store_sparse_by_delay),
        vision_compile=bool(args.vision_compile),
        vision_max_side=max(1, int(args.vision_max_side)),
        vision_max_elements=max(1, int(args.vision_max_elements)),
        modgrid_max_side=max(1, int(args.modgrid_max_side)),
        modgrid_max_elements=max(1, int(args.modgrid_max_elements)),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BioSNN demo and open dashboard")
    parser.add_argument(
        "--demo",
        choices=[
            "minimal",
            "network",
            "vision",
            "pruning_sparse",
            "neurogenesis_sparse",
            "propagation_impulse",
            "delay_impulse",
            "learning_gate",
            "dopamine_plasticity",
            "logic_curriculum",
            "logic_and",
            "logic_or",
            "logic_xor",
            "logic_nand",
            "logic_nor",
            "logic_xnor",
        ],
        default=_default_demo(),
        help=(
            "demo to run: minimal, network, vision, pruning_sparse, neurogenesis_sparse, "
            "propagation_impulse, delay_impulse, learning_gate, dopamine_plasticity, logic_curriculum, logic_and, logic_or, "
            "logic_xor, logic_nand, logic_nor, logic_xnor"
        ),
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="run device; CUDA defaults to DelayedSparseMatmulSynapse in the demo",
    )
    parser.add_argument(
        "--mode",
        choices=["dashboard", "fast"],
        default="dashboard",
        help="run mode: dashboard (full artifacts) or fast (throughput-oriented)",
    )
    parser.add_argument(
        "--max-ring-mib",
        type=float,
        default=2048.0,
        help="max ring buffer size per projection in MiB (set <=0 to disable)",
    )
    parser.add_argument(
        "--fused-layout",
        choices=["auto", "coo", "csr"],
        default="auto",
        help="fused sparse layout preference for delayed_sparse_matmul synapses",
    )
    parser.add_argument(
        "--synapse-backend",
        choices=["spmm_fused", "event_driven"],
        default="spmm_fused",
        help="synapse backend preference for delayed_sparse_matmul",
    )
    parser.add_argument(
        "--receptor-mode",
        choices=["exc_only", "ei_ampa_nmda_gabaa", "ei_ampa_nmda_gabaa_gabab"],
        default="exc_only",
        help="receptor profile preset used by schema/registry-level run specs",
    )
    parser.add_argument(
        "--ring-dtype",
        type=_parse_ring_dtype,
        default=None,
        help="ring buffer dtype: none, float32, float16, or bfloat16",
    )
    parser.add_argument(
        "--receptor-state-dtype",
        type=_parse_ring_dtype,
        default=None,
        help=(
            "receptor state dtype for multi-receptor sparse synapses: "
            "none, float32, float16, or bfloat16"
        ),
    )
    parser.add_argument(
        "--ring-strategy",
        choices=["dense", "event_bucketed"],
        default="dense",
        help="ring buffer strategy for delayed_current synapses",
    )
    parser.add_argument(
        "--store-sparse-by-delay",
        type=_parse_true_false,
        default=None,
        metavar="{true,false}",
        help="override sparse by-delay matrix storage for delayed_sparse_matmul",
    )
    parser.add_argument(
        "--monitor-safe-defaults",
        dest="monitor_safe_defaults",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable safety rails for large CSV monitors (default: enabled)",
    )
    parser.add_argument(
        "--large-network-safety",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "enable conservative large-network safety overrides "
            "(monitor sync off, bounded monitor payloads, fp16 receptor state on CUDA)"
        ),
    )
    parser.add_argument(
        "--monitor-neuron-sample",
        type=int,
        default=512,
        help="sample size for neuron/spike monitors when safety rails are enabled",
    )
    parser.add_argument(
        "--monitor-edge-sample",
        type=int,
        default=20000,
        help="max edge sample for weight monitors when safety rails are enabled",
    )
    parser.add_argument(
        "--allow-cuda-monitor-sync",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="allow CUDA monitors that require CPU sync (e.g., spike events)",
    )
    parser.add_argument(
        "--vision-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable torch.compile for vision monitor conversion pipeline",
    )
    parser.add_argument(
        "--vision-max-side",
        type=int,
        default=64,
        help="max side length for vision monitor exports",
    )
    parser.add_argument(
        "--vision-max-elements",
        type=int,
        default=16384,
        help="max element count for vision monitor exports",
    )
    parser.add_argument(
        "--modgrid-max-side",
        type=int,
        default=64,
        help="max side length for modulator grid monitor exports",
    )
    parser.add_argument(
        "--modgrid-max-elements",
        type=int,
        default=16384,
        help="max element count for modulator grid monitor exports",
    )
    parser.add_argument(
        "--parallel-compile",
        choices=["auto", "on", "off"],
        default="auto",
        help="parallelize projection compilation on CPU",
    )
    parser.add_argument(
        "--parallel-compile-workers",
        default="auto",
        help="worker count for parallel compile (int or 'auto')",
    )
    parser.add_argument(
        "--parallel-compile-torch-threads",
        type=int,
        default=1,
        help="torch.set_num_threads value used inside compile workers",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="run a short torch.profiler trace after the demo (writes profile.json)",
    )
    parser.add_argument("--profile-steps", type=int, default=20)
    parser.add_argument(
        "--torch-threads",
        default="auto",
        help="set torch.set_num_threads (int or 'auto')",
    )
    parser.add_argument(
        "--torch-interop-threads",
        default="auto",
        help="set torch.set_num_interop_threads (int or 'auto')",
    )
    parser.add_argument(
        "--set-omp-env",
        action="store_true",
        help="set OMP_NUM_THREADS and MKL_NUM_THREADS to match --torch-threads",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--delay_steps", type=int, default=3)
    parser.add_argument("--learning_lr", type=float, default=0.1)
    parser.add_argument("--da_amount", type=float, default=1.0)
    parser.add_argument("--da_step", type=int, default=10)
    parser.add_argument(
        "--modulators-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable modulators from run-spec plumbing",
    )
    parser.add_argument(
        "--modulator-kinds",
        type=str,
        default="",
        help="comma-separated modulator kinds (dopamine,acetylcholine,noradrenaline,serotonin)",
    )
    parser.add_argument(
        "--modulator-field-type",
        choices=["global_scalar", "grid_diffusion_2d"],
        default="global_scalar",
        help="modulator field type for schema/registry-level run specs",
    )
    parser.add_argument(
        "--modulator-grid-size",
        type=str,
        default="16x16",
        help="modulator grid size as 'HxW' (e.g., 16x16)",
    )
    parser.add_argument(
        "--modulator-world-extent",
        type=str,
        default="1.0,1.0",
        help="modulator world extent as 'x,y'",
    )
    parser.add_argument("--modulator-diffusion", type=float, default=0.0)
    parser.add_argument("--modulator-decay-tau", type=float, default=1.0)
    parser.add_argument("--modulator-deposit-sigma", type=float, default=0.0)
    parser.add_argument(
        "--logic-adv-synapse-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--logic-adv-synapse-conductance-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--logic-adv-synapse-nmda-block",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--logic-adv-synapse-stp-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--logic-wrapper-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--logic-wrapper-ach-lr-gain", type=float, default=0.0)
    parser.add_argument("--logic-wrapper-ne-lr-gain", type=float, default=0.0)
    parser.add_argument("--logic-wrapper-ht-extra-weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--logic-excitability-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--logic-excitability-ach-gain", type=float, default=0.0)
    parser.add_argument("--logic-excitability-ne-gain", type=float, default=0.0)
    parser.add_argument("--logic-excitability-ht-gain", type=float, default=0.0)
    parser.add_argument(
        "--logic-gate",
        choices=["and", "or", "xor", "nand", "nor", "xnor"],
        default=None,
        help="logic gate override for logic_* demos (default follows selected demo)",
    )
    parser.add_argument(
        "--logic-backend",
        choices=["harness", "engine"],
        default="harness",
        help="execution backend tag for logic_* demos (harness or engine)",
    )
    parser.add_argument(
        "--logic-learning-mode",
        choices=["rstdp", "surrogate", "none"],
        default="rstdp",
        help="learning mode for logic_* demos",
    )
    parser.add_argument(
        "--logic-neuron-model",
        choices=["adex_3c", "lif_3c"],
        default="adex_3c",
        help="neuron model for logic_* demos",
    )
    parser.add_argument("--logic-sim-steps-per-trial", type=int, default=10)
    parser.add_argument(
        "--logic-curriculum-gates",
        type=str,
        default="or,and,nor,nand,xor,xnor",
        help="comma-separated gate sequence for logic_curriculum demo",
    )
    parser.add_argument(
        "--logic-curriculum-replay-ratio",
        type=float,
        default=0.35,
        help="fraction of curriculum trials per phase used for replay of previous gates (0..1)",
    )
    parser.add_argument(
        "--logic-sampling-method",
        choices=["sequential", "random_balanced"],
        default="sequential",
    )
    parser.add_argument(
        "--logic-debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable per-trial debug logs for logic_* demos",
    )
    parser.add_argument("--logic-debug-every", type=int, default=25)
    parser.add_argument("--n", type=int, default=100, help="number of neurons")
    parser.add_argument("--p", type=float, default=0.05, help="connection probability")
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n-in", type=int, default=16)
    parser.add_argument("--n-hidden", type=int, default=64)
    parser.add_argument("--n-out", type=int, default=10)
    parser.add_argument("--input-pops", type=int, default=2)
    parser.add_argument("--input-depth", type=int, default=2)
    parser.add_argument("--hidden-layers", type=int, default=1)
    parser.add_argument("--hidden-pops-per-layer", type=int, default=1)
    parser.add_argument("--output-pops", type=int, default=1)
    parser.add_argument("--input-cross", action="store_true")
    parser.add_argument("--p-in-hidden", type=float, default=None, help="legacy: use --relay-to-hidden-p")
    parser.add_argument("--p-hidden-out", type=float, default=None, help="legacy: use --hidden-to-output-p")
    parser.add_argument("--input-to-relay-p", type=float, default=0.35)
    parser.add_argument("--input-to-relay-weight-scale", type=float, default=1.5)
    parser.add_argument("--relay-to-hidden-p", type=float, default=0.20)
    parser.add_argument("--relay-to-hidden-weight-scale", type=float, default=1.0)
    parser.add_argument("--hidden-to-output-p", type=float, default=0.20)
    parser.add_argument("--hidden-to-output-weight-scale", type=float, default=1.0)
    parser.add_argument("--input-skip-to-hidden", action="store_true")
    parser.add_argument("--input-skip-p", type=float, default=0.03)
    parser.add_argument("--input-skip-weight-scale", type=float, default=0.5)
    parser.add_argument("--relay-cross", action="store_true")
    parser.add_argument("--relay-cross-p", type=float, default=0.05)
    parser.add_argument("--relay-cross-weight-scale", type=float, default=0.2)
    parser.add_argument("--relay-lateral", action="store_true")
    parser.add_argument("--hidden-lateral", action="store_true")
    parser.add_argument("--weight-init", type=float, default=0.05)
    parser.add_argument(
        "--feedforward-delay-from-distance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable distance-based delays on feedforward projections",
    )
    parser.add_argument(
        "--feedforward-delay-base-velocity",
        type=float,
        default=1.0,
        help="base conduction velocity in length units per second",
    )
    parser.add_argument(
        "--feedforward-delay-myelin-scale",
        type=float,
        default=5.0,
        help="scale for myelin speedup factor",
    )
    parser.add_argument(
        "--feedforward-delay-myelin-mean",
        type=float,
        default=0.6,
        help="mean myelin factor (0..1)",
    )
    parser.add_argument(
        "--feedforward-delay-myelin-std",
        type=float,
        default=0.2,
        help="std dev for myelin factor",
    )
    parser.add_argument(
        "--feedforward-delay-distance-scale",
        type=float,
        default=1.0,
        help="multiplier to convert coordinate units to meters",
    )
    parser.add_argument(
        "--feedforward-delay-min",
        type=float,
        default=0.0,
        help="minimum axonal delay (seconds)",
    )
    parser.add_argument(
        "--feedforward-delay-max",
        type=float,
        default=None,
        help="maximum axonal delay (seconds); omit for no max",
    )
    parser.add_argument(
        "--feedforward-delay-use-ceil",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use ceil when converting delay seconds to steps",
    )
    parser.add_argument("--input-drive", type=float, default=1.0)
    parser.add_argument("--drive-monitor", action="store_true", help="write drive.csv diagnostics")
    parser.add_argument(
        "--enable-homeostasis",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable EMA firing-rate homeostasis (threshold adaptation)",
    )
    parser.add_argument("--homeostasis-alpha", type=float, default=0.01)
    parser.add_argument("--homeostasis-eta", type=float, default=1e-3)
    parser.add_argument("--homeostasis-r-target", type=float, default=0.05)
    parser.add_argument("--homeostasis-clamp-min", type=float, default=0.0)
    parser.add_argument("--homeostasis-clamp-max", type=float, default=0.050)
    parser.add_argument(
        "--homeostasis-scope",
        choices=["per_population", "per_neuron"],
        default="per_neuron",
    )
    parser.add_argument("--homeostasis-export-every", type=int, default=10)
    parser.add_argument(
        "--enable-pruning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable activity-based structural pruning",
    )
    parser.add_argument("--prune-interval-steps", type=int, default=250)
    parser.add_argument("--prune-usage-alpha", type=float, default=0.01)
    parser.add_argument("--prune-w-min", type=float, default=0.05)
    parser.add_argument("--prune-usage-min", type=float, default=0.01)
    parser.add_argument("--prune-k-min-out", type=int, default=1)
    parser.add_argument("--prune-k-min-in", type=int, default=1)
    parser.add_argument("--prune-max-fraction", type=float, default=0.10)
    parser.add_argument(
        "--prune-verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="print prune summaries at prune boundaries",
    )
    parser.add_argument(
        "--enable-neurogenesis",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable online neurogenesis for hidden populations",
    )
    parser.add_argument("--growth-interval-steps", type=int, default=500)
    parser.add_argument("--add-neurons-per-event", type=int, default=4)
    parser.add_argument("--newborn-plasticity-multiplier", type=float, default=1.5)
    parser.add_argument("--newborn-duration-steps", type=int, default=250)
    parser.add_argument("--max-total-neurons", type=int, default=20000)
    parser.add_argument(
        "--neurogenesis-verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="print growth summaries at neurogenesis boundaries",
    )
    parser.add_argument("--port", type=int)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="run simulation and exit without starting the dashboard HTTP server",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="optional explicit run folder name (used by dashboard API subprocess runs)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=None,
        help="override artifacts root directory (default: <repo>/artifacts)",
    )
    parser.add_argument("--refresh-ms", type=int, default=1200)
    return parser.parse_args(argv)


def _make_run_dir(base: Path, run_id: str | None = None) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    if run_id:
        safe_id = "".join(ch if ch.isalnum() or ch in "_-." else "_" for ch in run_id).strip("._")
        if not safe_id:
            raise ValueError("run_id must contain at least one valid character")
        run_dir = base / safe_id
        if run_dir.exists():
            if not run_dir.is_dir():
                raise FileExistsError(f"Run path exists and is not a directory: {run_dir}")
            return run_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base / f"run_{stamp}"
        if run_dir.exists():
            run_dir = base / f"run_{stamp}_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _resolve_device(torch: Any, device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _default_demo() -> str:
    try:
        from biosnn.simulation.engine import TorchNetworkEngine

        _ = TorchNetworkEngine
        return "network"
    except Exception:
        return "minimal"


def _should_launch_dashboard(mode: str) -> bool:
    return mode.lower().strip() != "fast"


def _parse_thread_setting(value: str | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower().strip() == "auto":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid thread count: {value}") from exc
    if parsed <= 0:
        raise ValueError(f"Thread count must be positive: {parsed}")
    return parsed


def _parse_ring_dtype(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized == "none":
        return None
    if normalized in {"float32", "float16", "bfloat16"}:
        return normalized
    raise ValueError(f"Invalid ring dtype: {value}")


def _parse_true_false(value: str | None) -> bool:
    if value is None:
        raise ValueError("Expected true or false")
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean literal: {value}")


def _resolve_logic_neuron_model_arg(args: argparse.Namespace) -> LogicGateNeuronModel:
    model = str(getattr(args, "logic_neuron_model", "adex_3c")).strip().lower()
    if model == "lif_3c":
        return "lif_3c"
    return "adex_3c"


def _apply_large_network_safety_overrides(*, args: argparse.Namespace, device: str) -> None:
    if not bool(getattr(args, "large_network_safety", False)):
        return

    args.monitor_safe_defaults = True

    if getattr(args, "allow_cuda_monitor_sync", None) is None:
        args.allow_cuda_monitor_sync = False

    args.monitor_neuron_sample = max(1, min(int(args.monitor_neuron_sample), 512))
    args.monitor_edge_sample = max(1, min(int(args.monitor_edge_sample), 20_000))

    args.vision_max_side = max(1, min(int(args.vision_max_side), 64))
    args.modgrid_max_side = max(1, min(int(args.modgrid_max_side), 64))
    args.vision_max_elements = max(1, min(int(args.vision_max_elements), 16_384))
    args.modgrid_max_elements = max(1, min(int(args.modgrid_max_elements), 16_384))

    if device == "cuda" and getattr(args, "receptor_state_dtype", None) is None:
        args.receptor_state_dtype = "float16"


def _validate_ring_dtype_for_device(*, torch: Any, ring_dtype: str | None, device: str) -> None:
    if ring_dtype is None:
        return
    dtype = getattr(torch, ring_dtype, None)
    if dtype is None:
        raise ValueError(f"Unsupported ring dtype: {ring_dtype}")
    if device != "cpu":
        return
    try:
        probe = torch.zeros((2, 2), device="cpu", dtype=dtype)
        probe.add_(1)
        idx = torch.tensor([0, 1], device="cpu", dtype=torch.long)
        probe.index_add_(0, idx, probe)
    except Exception as exc:
        raise ValueError(
            f"--ring-dtype {ring_dtype} is not supported on CPU in this runtime."
        ) from exc


def _apply_threading_env(threads: int | None, interop: int | None) -> None:
    env_threads = threads if threads is not None else interop
    if env_threads is None:
        return
    os.environ["OMP_NUM_THREADS"] = str(env_threads)
    os.environ["MKL_NUM_THREADS"] = str(env_threads)


def _apply_threading_torch(torch: Any, threads: int | None, interop: int | None) -> None:
    if threads is not None and hasattr(torch, "set_num_threads"):
        torch.set_num_threads(threads)
    if interop is not None and hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(interop)


def _build_dashboard_url(port: int, run_dir: Path, repo_root: Path, refresh_ms: int) -> str:
    base = f"http://localhost:{port}/docs/dashboard/"
    def param_or_none(filename: str) -> str:
        path = run_dir / filename
        if not path.exists():
            return "none"
        return _dashboard_param(repo_root, run_dir, filename)

    query = {
        "run": _run_dir_web_path(repo_root, run_dir),
        "topology": param_or_none("topology.json"),
        "neuron": param_or_none("neuron.csv"),
        "synapse": param_or_none("synapse.csv"),
        "spikes": param_or_none("spikes.csv"),
        "metrics": param_or_none("metrics.csv"),
        "weights": param_or_none("weights.csv"),
        "modgrid": param_or_none("modgrid.json"),
        "receptors": param_or_none("receptors.csv"),
        "vision": param_or_none("vision.json"),
        "refresh": str(refresh_ms),
    }

    query_str = "&".join(f"{key}={quote(value, safe='/:')}" for key, value in query.items())
    return f"{base}?{query_str}"


def _dashboard_param(repo_root: Path, run_dir: Path, filename: str) -> str:
    try:
        rel = run_dir.relative_to(repo_root)
    except ValueError:
        return (run_dir / filename).as_posix()
    return (Path("/") / rel / filename).as_posix()


def _run_dir_web_path(repo_root: Path, run_dir: Path) -> str:
    return _dashboard_param(repo_root, run_dir, "").rstrip("/")


def _find_port(requested: int | None) -> int:
    if requested is not None:
        if _port_available(requested):
            return requested
        raise RuntimeError(f"Port {requested} is not available")

    for port in range(8000, 8011):
        if _port_available(port):
            return port
    raise RuntimeError("No available port in range 8000-8010")


def _port_available(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            sock.bind(("localhost", port))
        return True
    except OSError:
        return False


if __name__ == "__main__":
    main()
