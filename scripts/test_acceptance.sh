#!/usr/bin/env bash
set -euo pipefail
python -m pytest -m "acceptance and not cuda and not slow"
