#!/usr/bin/env python3
"""Run and persist the optional measured CPU collective."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpt2_nano_parallel.cpu_demo import write_demo


if __name__ == "__main__":
    write_demo(ROOT / "artifacts" / "cpu_allreduce_measured.json")
    print("wrote artifacts/cpu_allreduce_measured.json")
