$ErrorActionPreference = "Stop"
python -m pytest -m "unit and not cuda and not slow"
