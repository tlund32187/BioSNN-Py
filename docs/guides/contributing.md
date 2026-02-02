# Contributing

## Dev setup
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[dev]"
```

## Checks
```powershell
ruff check .
mypy .
pytest
```

## Style rules (high-level)
- Add types where practical; prefer `Protocol` for interfaces used across layers.
- Public API changes must update `docs/public_api.md` and `CHANGELOG.md`.
- Avoid circular imports: contracts live in `biosnn.contracts`.
