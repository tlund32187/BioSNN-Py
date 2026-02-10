from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

import pytest

from biosnn.experiments.demo_registry import ALLOWED_DEMOS
from biosnn.runners import cli

pytestmark = pytest.mark.acceptance

_LOGIC_DEMOS = {
    "logic_and",
    "logic_or",
    "logic_xor",
    "logic_nand",
    "logic_nor",
    "logic_xnor",
}


class _RunDemoSelectParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_run_demo_select = False
        self.values: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "select":
            attrs_map = dict(attrs)
            if attrs_map.get("id") == "runDemoSelect":
                self._in_run_demo_select = True
            return
        if tag == "option" and self._in_run_demo_select:
            attrs_map = dict(attrs)
            value = attrs_map.get("value")
            if value:
                self.values.append(value)

    def handle_endtag(self, tag: str) -> None:
        if tag == "select" and self._in_run_demo_select:
            self._in_run_demo_select = False


def _dashboard_dropdown_demos() -> tuple[str, ...]:
    dashboard_html_path = Path("docs/dashboard/index.html")
    source = dashboard_html_path.read_text(encoding="utf-8")
    parser = _RunDemoSelectParser()
    parser.feed(source)
    return tuple(parser.values)


_DROPDOWN_DEMOS = _dashboard_dropdown_demos()


def _extra_args_for_demo(demo_name: str) -> list[str]:
    if demo_name in _LOGIC_DEMOS:
        # Keep smoke tests quick while still exercising the demo path.
        return ["--logic-learning-mode", "none"]
    if demo_name == "logic_curriculum":
        return [
            "--logic-curriculum-gates",
            "or,and,xor",
            "--logic-curriculum-replay-ratio",
            "0.25",
            "--logic-learning-mode",
            "rstdp",
        ]
    return []


def test_dashboard_dropdown_demo_list_matches_registry() -> None:
    assert _DROPDOWN_DEMOS, "Dashboard demo dropdown is empty"
    assert tuple(ALLOWED_DEMOS) == _DROPDOWN_DEMOS


@pytest.mark.parametrize("demo_name", _DROPDOWN_DEMOS)
def test_dashboard_dropdown_demo_writes_topology(
    monkeypatch,
    tmp_path: Path,
    demo_name: str,
) -> None:
    pytest.importorskip("torch")
    run_dir = tmp_path / demo_name

    args = cli._parse_args(
        [
            "--demo",
            demo_name,
            "--mode",
            "dashboard",
            "--device",
            "cpu",
            "--steps",
            "20",
            "--no-open",
            *_extra_args_for_demo(demo_name),
        ]
    )

    monkeypatch.setattr(cli, "_parse_args", lambda *_: args)
    monkeypatch.setattr(cli, "_make_run_dir", lambda *_: run_dir)
    monkeypatch.setattr(cli, "_should_launch_dashboard", lambda *_: False)

    cli.main()

    topology_path = run_dir / "topology.json"
    assert topology_path.exists(), f"Missing topology.json for demo={demo_name}"
    assert topology_path.stat().st_size > 0, f"Empty topology.json for demo={demo_name}"

    with topology_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert isinstance(payload, dict), f"Invalid topology payload for demo={demo_name}"
    assert isinstance(payload.get("nodes"), list), f"Missing nodes list for demo={demo_name}"
    assert isinstance(payload.get("edges"), list), f"Missing edges list for demo={demo_name}"
    assert payload.get("nodes"), f"No topology nodes for demo={demo_name}"
