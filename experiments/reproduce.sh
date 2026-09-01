#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
modal run modal_app.py::controlled_experiment "$@"
python3 experiments/report.py
