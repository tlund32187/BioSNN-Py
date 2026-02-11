from __future__ import annotations

import random

import pytest

from biosnn.tasks.logic_gates.configs import ExplorationConfig
from biosnn.tasks.logic_gates.engine_runner import select_action_wta

pytestmark = pytest.mark.unit


def test_select_action_wta_tie_break_random_among_max_is_seeded() -> None:
    torch = pytest.importorskip("torch")
    counts = torch.tensor([3.0, 3.0], dtype=torch.float32)
    exploration = ExplorationConfig(
        enabled=True,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay_trials=100,
        tie_break="random_among_max",
        seed=11,
    )

    rng_a = random.Random(11)
    action_a, info_a = select_action_wta(
        counts,
        last_action=0,
        exploration=exploration,
        trial_idx=0,
        is_eval=False,
        rng=rng_a,
    )
    rng_b = random.Random(11)
    action_b, info_b = select_action_wta(
        counts,
        last_action=0,
        exploration=exploration,
        trial_idx=0,
        is_eval=False,
        rng=rng_b,
    )

    assert action_a in {0, 1}
    assert action_a == action_b
    assert info_a["tie"] is True
    assert info_a["explored"] is False
    assert info_b["greedy"] == info_a["greedy"]


def test_select_action_wta_epsilon_one_forces_exploration() -> None:
    torch = pytest.importorskip("torch")
    counts = torch.tensor([10.0, 0.0], dtype=torch.float32)
    exploration = ExplorationConfig(
        enabled=True,
        epsilon_start=1.0,
        epsilon_end=1.0,
        epsilon_decay_trials=1,
        tie_break="prefer_last",
        seed=7,
    )
    action, info = select_action_wta(
        counts,
        last_action=0,
        exploration=exploration,
        trial_idx=0,
        is_eval=False,
        rng=random.Random(7),
    )

    assert action in {0, 1}
    assert info["epsilon"] == pytest.approx(1.0)
    assert info["explored"] is True


def test_select_action_wta_eval_mode_is_deterministic() -> None:
    torch = pytest.importorskip("torch")
    counts = torch.tensor([1.0, 4.0], dtype=torch.float32)
    exploration = ExplorationConfig(
        enabled=True,
        epsilon_start=0.9,
        epsilon_end=0.1,
        epsilon_decay_trials=10,
        tie_break="random_among_max",
        seed=99,
    )
    action, info = select_action_wta(
        counts,
        last_action=0,
        exploration=exploration,
        trial_idx=5,
        is_eval=True,
        rng=random.Random(99),
    )

    assert action == 1
    assert info["epsilon"] == pytest.approx(0.0)
    assert info["explored"] is False
