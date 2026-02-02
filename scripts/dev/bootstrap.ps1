# Bootstrap dev environment (Windows PowerShell)
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[dev]"
Write-Host "Done. Try: ruff check . ; mypy . ; pytest" -ForegroundColor Green
