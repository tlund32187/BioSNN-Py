$ErrorActionPreference = "Stop"
python -m pytest -m "acceptance and not cuda and not slow"
