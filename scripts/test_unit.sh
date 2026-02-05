#!/usr/bin/env bash
set -euo pipefail
python -m pytest -m "unit and not cuda and not slow"
