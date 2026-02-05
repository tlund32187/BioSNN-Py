#!/usr/bin/env bash
set -euo pipefail
python -m pytest -m "not cuda and not slow"
