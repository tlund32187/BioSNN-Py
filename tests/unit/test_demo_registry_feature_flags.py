from __future__ import annotations

import pytest

from biosnn.experiments.demo_registry import (
    feature_flags_for_run_spec,
    list_demo_definitions,
    resolve_run_spec,
    run_spec_from_cli_args,
    run_spec_to_cli_args,
)
from biosnn.runners import cli

pytestmark = pytest.mark.unit


def test_logic_curriculum_feature_flags_differ_between_harness_and_engine() -> None:
    harness_features = feature_flags_for_run_spec(
        {
            "demo_id": "logic_curriculum",
            "logic_backend": "harness",
            "delay_steps": 3,
            "modulators": {"enabled": True, "kinds": ["dopamine"]},
        }
    )
    engine_features = feature_flags_for_run_spec(
        {
            "demo_id": "logic_curriculum",
            "logic_backend": "engine",
            "delay_steps": 3,
            "modulators": {"enabled": True, "kinds": ["dopamine"]},
        }
    )

    assert harness_features["logic_backend"] == "harness"
    assert harness_features["delays"]["enabled"] is False
    assert harness_features["delays"]["max_delay_steps"] is None
    assert harness_features["delays"]["ring_len"] is None
    assert harness_features["synapse"]["backend"] is None
    assert harness_features["synapse"]["ring_strategy"] is None
    assert harness_features["modulators"]["enabled"] is True
    assert harness_features["modulators"]["kinds"] == ["dopamine"]
    assert harness_features["modulators"]["pulse_step"] is None
    assert harness_features["modulators"]["amount"] is None

    assert engine_features["logic_backend"] == "engine"
    assert engine_features["delays"]["enabled"] is True
    assert engine_features["delays"]["max_delay_steps"] == 3
    assert engine_features["delays"]["ring_len"] == 4
    assert engine_features["synapse"]["backend"] == "spmm_fused"
    assert engine_features["synapse"]["ring_strategy"] == "dense"
    assert engine_features["modulators"]["enabled"] is True
    assert engine_features["modulators"]["kinds"] == ["dopamine"]
    assert engine_features["modulators"]["pulse_step"] == 50
    assert engine_features["modulators"]["amount"] == 0.1


def test_logic_curriculum_rstdp_reports_internal_dopamine_modulator() -> None:
    features = feature_flags_for_run_spec({"demo_id": "logic_curriculum"})

    assert features["logic_backend"] == "harness"
    assert features["learning"]["enabled"] is True
    assert features["learning"]["rule"] == "rstdp"
    assert features["delays"]["enabled"] is False
    assert features["synapse"]["ring_strategy"] is None
    assert features["modulators"]["enabled"] is True
    assert "dopamine" in features["modulators"]["kinds"]


def test_logic_curriculum_can_disable_learning_in_engine_mode() -> None:
    resolved = resolve_run_spec(
        {
            "demo_id": "logic_curriculum",
            "logic_backend": "engine",
            "learning": {"enabled": False},
        }
    )
    features = feature_flags_for_run_spec(resolved)

    assert resolved["logic_learning_mode"] == "none"
    assert resolved["learning"]["enabled"] is False
    assert features["learning"]["enabled"] is False
    assert features["learning"]["rule"] is None


def test_logic_engine_feature_flags_include_exploration_and_reward_window() -> None:
    features = feature_flags_for_run_spec(
        {
            "demo_id": "logic_curriculum",
            "logic_backend": "engine",
            "logic": {
                "reward_delivery_steps": 2,
                "reward_delivery_clamp_input": True,
                "exploration": {
                    "enabled": True,
                    "mode": "epsilon_greedy",
                    "epsilon_start": 0.25,
                    "epsilon_end": 0.05,
                    "epsilon_decay_trials": 1000,
                    "tie_break": "alternate",
                    "seed": 7,
                },
                "gate_context": {
                    "enabled": True,
                    "amplitude": 0.3,
                    "compartment": "dendrite",
                    "targets": ["hidden"],
                },
                "action_force": {
                    "enabled": True,
                    "window": "reward_window",
                    "steps": 2,
                    "amplitude": 0.8,
                    "compartment": "soma",
                },
            },
        }
    )

    assert features["exploration"]["enabled"] is True
    assert features["exploration"]["mode"] == "epsilon_greedy"
    assert features["exploration"]["tie_break"] == "alternate"
    assert features["reward_window"]["steps"] == 2
    assert features["reward_window"]["clamp_input"] is True
    assert features["action_force"]["enabled"] is True
    assert features["action_force"]["window"] == "reward_window"
    assert features["action_force"]["steps"] == 2
    assert features["action_force"]["amplitude"] == pytest.approx(0.8)
    assert features["gate_context"]["enabled"] is True
    assert features["gate_context"]["amplitude"] == pytest.approx(0.3)


def test_feature_flags_reflect_monitor_toggle() -> None:
    features = feature_flags_for_run_spec(
        {"demo_id": "network", "monitor_mode": "dashboard", "monitors_enabled": False}
    )

    assert features["monitor"]["enabled"] is False
    assert features["monitor"]["sync_policy"] == "disabled"


def test_resolve_run_spec_accepts_logic_backend_and_coerces_invalid_values() -> None:
    resolved_engine = resolve_run_spec({"demo_id": "logic_xor", "logic_backend": "engine"})
    resolved_invalid = resolve_run_spec({"demo_id": "logic_xor", "logic_backend": "bad-value"})

    assert resolved_engine["logic_backend"] == "engine"
    assert resolved_invalid["logic_backend"] == "harness"


def test_logic_demo_definitions_default_to_harness_backend() -> None:
    by_id = {str(entry["id"]): entry["defaults"] for entry in list_demo_definitions()}

    for demo_id in (
        "logic_curriculum",
        "logic_and",
        "logic_or",
        "logic_xor",
        "logic_nand",
        "logic_nor",
        "logic_xnor",
    ):
        assert by_id[demo_id]["logic_backend"] == "harness"


def test_resolve_run_spec_coerces_extended_schema_sections() -> None:
    resolved = resolve_run_spec(
        {
            "demo_id": "network",
            "synapse": {"receptor_mode": "ei_ampa_nmda_gabaa_gabab"},
            "modulators": {
                "field_type": "grid_diffusion_2d",
                "grid_size": "24x12",
                "world_extent": [2.5, 1.5],
                "diffusion": -1.0,
                "decay_tau": 0.0,
                "deposit_sigma": -0.5,
            },
            "homeostasis": {
                "enabled": True,
                "rule": "rate_ema_threshold",
                "alpha": 2.0,
                "eta": -1.0,
                "r_target": 0.02,
                "clamp_min": 0.4,
                "clamp_max": 0.1,
                "scope": "per_population",
            },
            "pruning": {
                "enabled": True,
                "prune_interval_steps": 0,
                "usage_alpha": -1.0,
                "w_min": -1.0,
                "usage_min": -1.0,
                "k_min_out": -3,
                "k_min_in": -5,
                "max_prune_fraction_per_interval": 2.0,
                "verbose": True,
            },
            "neurogenesis": {
                "enabled": True,
                "growth_interval_steps": 0,
                "add_neurons_per_event": 0,
                "newborn_plasticity_multiplier": 0.0,
                "newborn_duration_steps": 0,
                "max_total_neurons": 0,
                "verbose": True,
            },
        }
    )

    assert resolved["synapse"]["receptor_mode"] == "ei_ampa_nmda_gabaa_gabab"
    assert resolved["modulators"]["field_type"] == "grid_diffusion_2d"
    assert resolved["modulators"]["grid_size"] == [24, 12]
    assert resolved["modulators"]["world_extent"] == [2.5, 1.5]
    assert resolved["modulators"]["diffusion"] == 0.0
    assert resolved["modulators"]["decay_tau"] == 1.0
    assert resolved["modulators"]["deposit_sigma"] == 0.0
    assert resolved["homeostasis"]["enabled"] is True
    assert resolved["homeostasis"]["scope"] == "per_population"
    assert resolved["homeostasis"]["alpha"] == 0.01
    assert resolved["homeostasis"]["eta"] == 1e-3
    assert resolved["homeostasis"]["clamp_max"] == resolved["homeostasis"]["clamp_min"]
    assert resolved["pruning"]["enabled"] is True
    assert resolved["pruning"]["prune_interval_steps"] == 250
    assert resolved["pruning"]["usage_alpha"] == 0.01
    assert resolved["pruning"]["w_min"] == 0.05
    assert resolved["pruning"]["usage_min"] == 0.01
    assert resolved["pruning"]["k_min_out"] == 1
    assert resolved["pruning"]["k_min_in"] == 1
    assert resolved["pruning"]["max_prune_fraction_per_interval"] == 1.0
    assert resolved["neurogenesis"]["enabled"] is True
    assert resolved["neurogenesis"]["growth_interval_steps"] == 500
    assert resolved["neurogenesis"]["add_neurons_per_event"] == 4
    assert resolved["neurogenesis"]["newborn_plasticity_multiplier"] == 1.5
    assert resolved["neurogenesis"]["newborn_duration_steps"] == 250
    assert resolved["neurogenesis"]["max_total_neurons"] == 20000


def test_run_spec_cli_round_trip_extended_fields(tmp_path) -> None:
    start = resolve_run_spec(
        {
            "demo_id": "logic_xor",
            "monitors_enabled": False,
            "logic_backend": "engine",
            "synapse": {
                "backend": "event_driven",
                "fused_layout": "csr",
                "ring_strategy": "event_bucketed",
                "ring_dtype": "float16",
                "store_sparse_by_delay": True,
                "receptor_mode": "ei_ampa_nmda_gabaa",
            },
            "modulators": {
                "enabled": True,
                "kinds": ["dopamine", "acetylcholine", "noradrenaline", "serotonin"],
                "pulse_step": 17,
                "amount": 2.5,
                "field_type": "grid_diffusion_2d",
                "grid_size": [20, 22],
                "world_extent": [1.5, 2.5],
                "diffusion": 0.2,
                "decay_tau": 2.0,
                "deposit_sigma": 0.8,
            },
            "advanced_synapse": {
                "enabled": True,
                "conductance_mode": True,
                "nmda_voltage_block": True,
                "stp_enabled": True,
            },
            "wrapper": {
                "enabled": True,
                "ach_lr_gain": 0.4,
                "ne_lr_gain": 0.3,
                "ht_extra_weight_decay": 0.2,
            },
            "excitability_modulation": {
                "enabled": True,
                "ach_gain": 0.15,
                "ne_gain": 0.25,
                "ht_gain": 0.05,
            },
            "homeostasis": {
                "enabled": True,
                "rule": "rate_ema_threshold",
                "alpha": 0.03,
                "eta": 0.002,
                "r_target": 0.08,
                "clamp_min": 0.01,
                "clamp_max": 0.2,
                "scope": "per_neuron",
            },
            "pruning": {
                "enabled": True,
                "prune_interval_steps": 111,
                "usage_alpha": 0.02,
                "w_min": 0.03,
                "usage_min": 0.01,
                "k_min_out": 2,
                "k_min_in": 3,
                "max_prune_fraction_per_interval": 0.33,
                "verbose": True,
            },
            "neurogenesis": {
                "enabled": True,
                "growth_interval_steps": 333,
                "add_neurons_per_event": 5,
                "newborn_plasticity_multiplier": 1.8,
                "newborn_duration_steps": 444,
                "max_total_neurons": 12345,
                "verbose": True,
            },
            "logic": {
                "learn_every": 3,
                "reward_delivery_steps": 2,
                "reward_delivery_clamp_input": True,
                "gate_context": {
                    "enabled": True,
                    "amplitude": 0.42,
                    "compartment": "soma",
                    "targets": ["hidden", "out"],
                },
                "action_force": {
                    "enabled": True,
                    "window": "post_decision",
                    "steps": 1,
                    "amplitude": 0.9,
                    "compartment": "dendrite",
                },
                "exploration": {
                    "enabled": True,
                    "mode": "epsilon_greedy",
                    "epsilon_start": 0.3,
                    "epsilon_end": 0.05,
                    "epsilon_decay_trials": 250,
                    "tie_break": "prefer_last",
                    "seed": 13,
                },
            },
        }
    )

    cli_args = run_spec_to_cli_args(run_spec=start, run_id="rt", artifacts_dir=tmp_path)
    parsed = cli._parse_args(cli_args)
    rt = run_spec_from_cli_args(args=parsed, device=str(parsed.device or "cpu"))

    assert rt["monitors_enabled"] is False
    assert rt["logic_backend"] == "engine"
    assert rt["synapse"]["backend"] == "event_driven"
    assert rt["synapse"]["fused_layout"] == "csr"
    assert rt["synapse"]["ring_strategy"] == "event_bucketed"
    assert rt["synapse"]["ring_dtype"] == "float16"
    assert rt["synapse"]["store_sparse_by_delay"] is True
    assert rt["synapse"]["receptor_mode"] == "ei_ampa_nmda_gabaa"
    assert rt["modulators"]["field_type"] == "grid_diffusion_2d"
    assert rt["modulators"]["grid_size"] == [20, 22]
    assert rt["modulators"]["world_extent"] == [1.5, 2.5]
    assert rt["modulators"]["diffusion"] == 0.2
    assert rt["modulators"]["decay_tau"] == 2.0
    assert rt["modulators"]["deposit_sigma"] == 0.8
    assert rt["modulators"]["kinds"] == [
        "dopamine",
        "acetylcholine",
        "noradrenaline",
        "serotonin",
    ]
    assert rt["advanced_synapse"]["enabled"] is True
    assert rt["advanced_synapse"]["conductance_mode"] is True
    assert rt["advanced_synapse"]["nmda_voltage_block"] is True
    assert rt["advanced_synapse"]["stp_enabled"] is True
    assert rt["wrapper"]["enabled"] is True
    assert rt["wrapper"]["ach_lr_gain"] == 0.4
    assert rt["wrapper"]["ne_lr_gain"] == 0.3
    assert rt["wrapper"]["ht_extra_weight_decay"] == 0.2
    assert rt["excitability_modulation"]["enabled"] is True
    assert rt["excitability_modulation"]["ach_gain"] == 0.15
    assert rt["excitability_modulation"]["ne_gain"] == 0.25
    assert rt["excitability_modulation"]["ht_gain"] == 0.05
    assert rt["homeostasis"]["enabled"] is True
    assert rt["homeostasis"]["alpha"] == 0.03
    assert rt["pruning"]["enabled"] is True
    assert rt["pruning"]["prune_interval_steps"] == 111
    assert rt["neurogenesis"]["enabled"] is True
    assert rt["neurogenesis"]["growth_interval_steps"] == 333
    assert rt["logic"]["learn_every"] == 3
    assert rt["logic"]["reward_delivery_steps"] == 2
    assert rt["logic"]["reward_delivery_clamp_input"] is True
    assert rt["logic"]["action_force"]["enabled"] is True
    assert rt["logic"]["action_force"]["window"] == "post_decision"
    assert rt["logic"]["action_force"]["steps"] == 1
    assert rt["logic"]["action_force"]["amplitude"] == pytest.approx(0.9)
    assert rt["logic"]["action_force"]["compartment"] == "dendrite"
    assert rt["logic"]["gate_context"]["enabled"] is True
    assert rt["logic"]["gate_context"]["amplitude"] == pytest.approx(0.42)
    assert rt["logic"]["gate_context"]["compartment"] == "soma"
    assert rt["logic"]["gate_context"]["targets"] == ["hidden", "out"]
    assert rt["logic"]["exploration"]["enabled"] is True
    assert rt["logic"]["exploration"]["epsilon_start"] == 0.3
    assert rt["logic"]["exploration"]["epsilon_end"] == 0.05
    assert rt["logic"]["exploration"]["epsilon_decay_trials"] == 250
    assert rt["logic"]["exploration"]["tie_break"] == "prefer_last"


def test_logic_curriculum_cli_round_trip_preserves_disabled_learning(tmp_path) -> None:
    start = resolve_run_spec(
        {
            "demo_id": "logic_curriculum",
            "logic_backend": "engine",
            "learning": {"enabled": False, "rule": "rstdp"},
            "logic_curriculum_gates": "or,and,xor",
            "logic_curriculum_replay_ratio": 0.5,
        }
    )

    cli_args = run_spec_to_cli_args(run_spec=start, run_id="rt", artifacts_dir=tmp_path)
    parsed = cli._parse_args(cli_args)
    rt = run_spec_from_cli_args(args=parsed, device=str(parsed.device or "cpu"))

    assert rt["demo_id"] == "logic_curriculum"
    assert rt["logic_backend"] == "engine"
    assert rt["logic_learning_mode"] == "none"
    assert rt["learning"]["enabled"] is False
