#!/usr/bin/env bash
# Regenerates summaries/, plots/, and artifact-manifest.json from raw/ alone.
# Does NOT run the GPU pilot itself -- that is a separate, billable step:
#   modal run modal_app.py::sentinel_pilot
set -euo pipefail

cd "$(dirname "$0")/../.."
python3 -m experiments.sentinel_report experiments/sentinel-pilot
