#!/usr/bin/env python3
"""Generate reproducible analytical, simulated, and measured artifact metadata."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpt2_nano_parallel.cli import _report
from gpt2_nano_parallel.memory import zero_memory


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.new")
    try:
        temporary.write_text(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _validate_cpu_measurement(raw: bytes) -> dict[str, Any]:
    """Validate that an artifact was produced by the local Gloo demo."""

    try:
        measurement = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("measurement is not valid UTF-8 JSON") from error
    if not isinstance(measurement, dict):
        raise ValueError("measurement must be a JSON object")
    expected_metadata = {
        "schema_version": 1,
        "generated_by": "gpt2_nano_parallel.cpu_demo.run_demo",
        "evidence": "measured",
        "scope": "local CPU processes using Gloo; not a GPU scaling result",
        "backend": "gloo",
        "device": "cpu",
    }
    for key, expected in expected_metadata.items():
        if measurement.get(key) != expected:
            raise ValueError(f"measurement has invalid {key}")
    torch_version = measurement.get("torch_version")
    if not isinstance(torch_version, str) or not torch_version:
        raise ValueError("measurement has invalid torch_version")
    world_size = measurement.get("world_size")
    if (
        not isinstance(world_size, int)
        or isinstance(world_size, bool)
        or world_size <= 1
    ):
        raise ValueError("measurement has invalid world_size")
    expected_sum = world_size * (world_size + 1) / 2
    if not _finite_number(measurement.get("expected_sum")) or not math.isclose(
        float(measurement["expected_sum"]), expected_sum
    ):
        raise ValueError("measurement has invalid expected_sum")
    if not _finite_number(measurement.get("wall_seconds")) or float(
        measurement["wall_seconds"]
    ) < 0:
        raise ValueError("measurement has invalid wall_seconds")
    ranks = measurement.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != world_size:
        raise ValueError("measurement has invalid rank results")
    seen: set[int] = set()
    for result in ranks:
        if not isinstance(result, dict):
            raise ValueError("measurement rank result must be an object")
        rank = result.get("rank")
        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or rank not in range(world_size)
            or rank in seen
        ):
            raise ValueError("measurement has invalid or duplicate rank")
        seen.add(rank)
        if not _finite_number(result.get("sum")) or not math.isclose(
            float(result["sum"]), expected_sum
        ):
            raise ValueError(f"measurement rank {rank} has an invalid sum")
        if not _finite_number(result.get("elapsed_seconds")) or float(
            result["elapsed_seconds"]
        ) < 0:
            raise ValueError(f"measurement rank {rank} has invalid timing")
    return measurement


def _validate_allreduce_validation(raw: bytes) -> dict[str, Any]:
    """Validate the local Gloo all-reduce model validation artifact."""

    try:
        measurement = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("validation artifact is not valid UTF-8 JSON") from error
    if not isinstance(measurement, dict):
        raise ValueError("validation artifact must be a JSON object")
    expected_metadata = {
        "schema_version": 1,
        "generated_by": (
            "gpt2_nano_parallel.allreduce_validation.run_gloo_allreduce_sweep"
        ),
        "evidence": "measured",
        "scope": (
            "local CPU processes using Gloo on 127.0.0.1; "
            "validates the communication cost model only; "
            "not a GPU scaling result"
        ),
        "backend": "gloo",
        "device": "cpu",
    }
    for key, expected in expected_metadata.items():
        if measurement.get(key) != expected:
            raise ValueError(f"validation artifact has invalid {key}")
    for key in ("torch_version", "python_version", "platform"):
        value = measurement.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"validation artifact has invalid {key}")
    if not _finite_number(measurement.get("wall_seconds")) or float(
        measurement["wall_seconds"]
    ) <= 0:
        raise ValueError("validation artifact has invalid wall_seconds")
    config = measurement.get("config")
    if not isinstance(config, dict):
        raise ValueError("validation artifact has invalid config")
    world_sizes = config.get("world_sizes")
    message_bytes = config.get("message_bytes")
    if (
        not isinstance(world_sizes, list)
        or not world_sizes
        or not all(isinstance(value, int) and value > 1 for value in world_sizes)
    ):
        raise ValueError("validation artifact has invalid world_sizes")
    if (
        not isinstance(message_bytes, list)
        or not message_bytes
        or not all(isinstance(value, int) and value > 0 for value in message_bytes)
    ):
        raise ValueError("validation artifact has invalid message_bytes")
    model = measurement.get("analytical_model")
    if not isinstance(model, dict) or model.get("source_function") != (
        "gpt2_nano_parallel.communication.ring_collective"
    ):
        raise ValueError("validation artifact has invalid analytical model")
    fit = measurement.get("fitted_constants")
    if not isinstance(fit, dict):
        raise ValueError("validation artifact has invalid fitted constants")
    for key in (
        "bandwidth_bytes_per_second",
        "bandwidth_gib_per_second",
        "latency_seconds",
        "latency_microseconds",
    ):
        if not _finite_number(fit.get(key)) or float(fit[key]) < 0:
            raise ValueError(f"validation artifact has invalid {key}")
    if float(fit["bandwidth_bytes_per_second"]) <= 0:
        raise ValueError("validation artifact has nonpositive bandwidth")
    residuals = measurement.get("residuals")
    if not isinstance(residuals, dict):
        raise ValueError("validation artifact has invalid residuals")
    for name in ("all_messages", "fit_range", "small_messages"):
        summary = residuals.get(name)
        if not isinstance(summary, dict) or not isinstance(summary.get("count"), int):
            raise ValueError(f"validation artifact has invalid {name} residuals")
        if summary["count"] > 0:
            for key in (
                "mean_absolute_relative_error",
                "max_absolute_relative_error",
                "root_mean_squared_error_seconds",
                "mean_signed_error_seconds",
            ):
                if not _finite_number(summary.get(key)):
                    raise ValueError(
                        f"validation artifact has invalid {name} {key}"
                    )
    measurements = measurement.get("measurements")
    expected_count = len(world_sizes) * len(message_bytes)
    if not isinstance(measurements, list) or len(measurements) != expected_count:
        raise ValueError("validation artifact has invalid measurements")
    seen: set[tuple[int, int]] = set()
    for row in measurements:
        if not isinstance(row, dict):
            raise ValueError("validation measurement must be an object")
        world_size = row.get("world_size")
        message_size = row.get("message_bytes")
        if (
            not isinstance(world_size, int)
            or world_size not in world_sizes
            or not isinstance(message_size, int)
            or message_size not in message_bytes
        ):
            raise ValueError("validation measurement has invalid dimensions")
        key = (world_size, message_size)
        if key in seen:
            raise ValueError("validation measurement has duplicate dimensions")
        seen.add(key)
        for field in (
            "measured_seconds",
            "model_seconds",
            "absolute_error_seconds",
            "relative_error",
            "ring_bytes_per_rank",
        ):
            if not _finite_number(row.get(field)) or float(row[field]) < 0:
                raise ValueError(f"validation measurement has invalid {field}")
    return measurement


def _memory_svg(path: Path) -> None:
    values = [zero_memory(44_000_000, 8, stage).total / 2**20 for stage in range(4)]
    width, height = 720, 420
    plot_left, plot_bottom, plot_height = 90, 350, 280
    scale = plot_height / max(values)
    bars = []
    labels = ["DP", "ZeRO-1", "ZeRO-2", "ZeRO-3"]
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        x = 125 + index * 145
        bar_height = value * scale
        y = plot_bottom - bar_height
        bars.append(
            f'<rect x="{x}" y="{y:.2f}" width="82" height="{bar_height:.2f}" '
            'fill="#2563eb"/>'
        )
        bars.append(
            f'<text x="{x + 41}" y="{y - 9:.2f}" text-anchor="middle" '
            f'font-size="14">{value:.1f} MiB</text>'
        )
        bars.append(
            f'<text x="{x + 41}" y="375" text-anchor="middle" '
            f'font-size="14">{label}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="360" y="30" text-anchor="middle" font-size="20" font-weight="bold">Analytical model-state memory per rank</text>
<text x="360" y="54" text-anchor="middle" font-size="13">44M parameters, world size 8, BF16 parameters and gradients, FP32 Adam states</text>
<line x1="{plot_left}" y1="{plot_bottom}" x2="690" y2="{plot_bottom}" stroke="#111827"/>
<line x1="{plot_left}" y1="70" x2="{plot_left}" y2="{plot_bottom}" stroke="#111827"/>
{''.join(bars)}
<text x="20" y="210" transform="rotate(-90 20 210)" text-anchor="middle" font-size="14">MiB per rank</text>
<text x="360" y="405" text-anchor="middle" font-size="12">Analytical, not measured hardware memory</text>
</svg>
"""
    path.write_text(svg)


def _pipeline_svg(path: Path, events: list[dict[str, object]]) -> None:
    stages = 4
    slots = max(int(event["slot"]) for event in events) + 1
    cell_width, cell_height = 34, 34
    width, height = 110 + slots * cell_width, 220
    cells = []
    lookup = {(int(e["slot"]), int(e["stage"])): e for e in events}
    for stage in range(stages):
        cells.append(
            f'<text x="75" y="{75 + stage * cell_height}" text-anchor="end" '
            f'font-size="13">stage {stage}</text>'
        )
        for slot in range(slots):
            x, y = 85 + slot * cell_width, 55 + stage * cell_height
            event = lookup.get((slot, stage))
            fill = "#e5e7eb"
            text = ""
            if event:
                fill = "#60a5fa" if event["phase"] == "F" else "#f59e0b"
                text = f'{event["phase"]}{event["microbatch"]}'
            cells.append(
                f'<rect x="{x}" y="{y}" width="30" height="26" fill="{fill}"/>'
                f'<text x="{x + 15}" y="{y + 18}" text-anchor="middle" '
                f'font-size="10">{text}</text>'
            )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="27" text-anchor="middle" font-size="19" font-weight="bold">Simulated GPipe schedule</text>
{''.join(cells)}
<text x="{width / 2}" y="207" text-anchor="middle" font-size="12">F is forward, B is backward, gray is idle; deterministic simulation</text>
</svg>
"""
    path.write_text(svg)


def main() -> None:
    artifacts = ROOT / "artifacts"
    figures = artifacts / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    report_path = artifacts / "publication_report.json"
    report = _report(str(ROOT / "configs" / "publication.toml"))
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    memory_path = figures / "zero_memory.svg"
    pipeline_path = figures / "pipeline_schedule.svg"
    _memory_svg(memory_path)
    _pipeline_svg(pipeline_path, report["pipeline"]["events"])

    measured_path = artifacts / "cpu_allreduce_measured.json"
    measured_digest: str | None = None
    measured_status = "pending"
    measured_files: list[str] = []
    measured_note = (
        "No validated local CPU measurement artifact is present. "
        "Run scripts/run_cpu_demo.py."
    )
    if measured_path.exists():
        try:
            measured_bytes = measured_path.read_bytes()
            _validate_cpu_measurement(measured_bytes)
            measured_digest = hashlib.sha256(measured_bytes).hexdigest()
            measured_status = "measured"
            measured_files = [str(measured_path.relative_to(ROOT))]
            measured_note = (
                "Validated local CPU Gloo process measurement; "
                "not a GPU scaling result."
            )
        except (OSError, ValueError) as error:
            measured_note = f"CPU measurement validation failed: {error}"

    validation_path = artifacts / "gloo_allreduce_model_validation.json"
    validation_digest: str | None = None
    validation_status = "pending"
    validation_files: list[str] = []
    validation_note = (
        "No validated local Gloo all-reduce model validation artifact is present. "
        "Run gpt2-parallel-allreduce."
    )
    if validation_path.exists():
        try:
            validation_bytes = validation_path.read_bytes()
            _validate_allreduce_validation(validation_bytes)
            validation_digest = hashlib.sha256(validation_bytes).hexdigest()
            validation_status = "measured"
            validation_files = [str(validation_path.relative_to(ROOT))]
            validation_note = (
                "Validated local CPU Gloo all-reduce sweep against the "
                "analytical ring byte model; not a GPU scaling result."
            )
        except (OSError, ValueError) as error:
            validation_note = (
                f"All-reduce model validation failed with {error}"
            )

    files = {
        str(path.relative_to(ROOT)): {"sha256": _sha256(path)}
        for path in (report_path, memory_path, pipeline_path)
    }
    if measured_digest is not None:
        files[str(measured_path.relative_to(ROOT))] = {
            "sha256": measured_digest
        }
    if validation_digest is not None:
        files[str(validation_path.relative_to(ROOT))] = {
            "sha256": validation_digest
        }

    manifest = {
        "schema_version": 1,
        "generated_by": "scripts/generate_artifacts.py",
        "status_definitions": {
            "analytical": "Derived from explicit formulas.",
            "simulated": "Produced by deterministic software simulation.",
            "measured": "Observed from a real local execution.",
            "pending": "Not run on the hardware needed for the claim.",
        },
        "claims": [
            {
                "id": "memory-and-communication",
                "status": "analytical",
                "files": [
                    str(report_path.relative_to(ROOT)),
                    str(memory_path.relative_to(ROOT)),
                ],
                "note": (
                    "Formula outputs for the report and memory figure. "
                    "Measured Gloo timing is tracked in a separate claim."
                ),
            },
            {
                "id": "pipeline-and-routing",
                "status": "simulated",
                "files": [
                    str(report_path.relative_to(ROOT)),
                    str(pipeline_path.relative_to(ROOT)),
                ],
                "note": "Deterministic event and token-routing simulations.",
            },
            {
                "id": "local-cpu-allreduce",
                "status": measured_status,
                "files": measured_files,
                "note": measured_note,
            },
            {
                "id": "local-gloo-allreduce-model-validation",
                "status": validation_status,
                "files": validation_files,
                "note": validation_note,
            },
            {
                "id": "multi-gpu-scaling",
                "status": "pending",
                "files": [],
                "note": "No multi-GPU run was performed.",
            },
        ],
        "files": files,
    }
    _atomic_write(
        artifacts / "manifest.json",
        json.dumps(manifest, indent=2) + "\n",
    )


if __name__ == "__main__":
    main()
