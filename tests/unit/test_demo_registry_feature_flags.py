from __future__ import annotations

import pytest

from biosnn.experiments.demo_registry import feature_flags_for_run_spec

pytestmark = pytest.mark.unit


def test_logic_curriculum_feature_flags_report_delay_and_modulators() -> None:
    features = feature_flags_for_run_spec(
        {
            "demo_id": "logic_curriculum",
            "delay_steps": 3,
            "modulators": {"enabled": True, "kinds": ["dopamine"]},
        }
    )

    assert features["delays"]["enabled"] is True
    assert features["delays"]["max_delay_steps"] == 3
    assert features["delays"]["ring_len"] == 4
    assert features["modulators"]["enabled"] is True
    assert features["modulators"]["kinds"] == ["dopamine"]


def test_logic_curriculum_rstdp_reports_internal_dopamine_modulator() -> None:
    features = feature_flags_for_run_spec({"demo_id": "logic_curriculum"})

    assert features["learning"]["enabled"] is True
    assert features["learning"]["rule"] == "rstdp"
    assert features["modulators"]["enabled"] is True
    assert "dopamine" in features["modulators"]["kinds"]
