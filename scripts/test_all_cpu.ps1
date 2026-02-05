$ErrorActionPreference = "Stop"
python -m pytest -m "not cuda and not slow"
