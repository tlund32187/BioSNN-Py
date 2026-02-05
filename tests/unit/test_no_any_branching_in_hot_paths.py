from __future__ import annotations

import re
from pathlib import Path


def test_no_tensor_any_branching_in_hot_paths():
    pattern = re.compile(r"^\s*if\s+.*\.any\(\)\s*:", re.MULTILINE)
    repo_root = Path(__file__).resolve().parents[2]
    targets = [
        repo_root / "src/biosnn/synapses/dynamics/delayed_current.py",
        repo_root / "src/biosnn/synapses/dynamics/delayed_sparse_matmul.py",
        repo_root / "src/biosnn/connectivity/topology_compile.py",
        repo_root / "src/biosnn/experiments/demo_network.py",
        repo_root / "src/biosnn/connectivity/builders/random_topology.py",
    ]

    for path in targets:
        text = path.read_text(encoding="utf-8")
        matches = pattern.findall(text)
        assert not matches, f"Found tensor any-branching in {path}"
