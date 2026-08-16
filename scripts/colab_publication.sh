#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
RUN_CPU_DEMO="${RUN_CPU_DEMO:-0}"
cd "$ROOT"

"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -e .

"$PYTHON" -m gpt2_nano_parallel.cli report configs/publication.toml
"$PYTHON" scripts/generate_artifacts.py

if [[ "$RUN_CPU_DEMO" == "1" ]]; then
  "$PYTHON" -m pip install -e '.[cpu-demo]'
  "$PYTHON" scripts/run_cpu_demo.py
  "$PYTHON" scripts/generate_artifacts.py
fi
