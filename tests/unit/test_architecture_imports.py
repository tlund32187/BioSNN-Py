from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

BANNED_PREFIXES = ("biosnn.monitors", "biosnn.io")
BANNED_MODULES = ("csv", "pandas", "pathlib")
ROOT = Path(__file__).resolve().parents[2]


def test_architecture_imports() -> None:
    roots = [
        ROOT / "src" / "biosnn" / "synapses",
        ROOT / "src" / "biosnn" / "biophysics",
    ]
    violations: list[str] = []

    for root in roots:
        for path in root.rglob("*.py"):
            for line in _iter_import_lines(path):
                for module in _extract_modules(line):
                    if not module or module.startswith("."):
                        continue
                    if module.startswith(BANNED_PREFIXES):
                        violations.append(f"{path}: {line}")
                        continue
                    top = module.split(".", 1)[0]
                    if top in BANNED_MODULES:
                        violations.append(f"{path}: {line}")

    assert not violations, "Disallowed imports detected:\n" + "\n".join(violations)


def _iter_import_lines(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    lines: list[str] = []
    for raw in text.splitlines():
        stripped = raw.lstrip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            line = stripped.split("#", 1)[0].strip()
            if line:
                lines.append(line)
    return lines


def _extract_modules(line: str) -> list[str]:
    if line.startswith("from "):
        parts = line.split()
        if len(parts) >= 2:
            return [parts[1]]
        return []
    if line.startswith("import "):
        modules: list[str] = []
        for part in line[len("import ") :].split(","):
            module = part.strip().split()[0]
            if module:
                modules.append(module)
        return modules
    return []
