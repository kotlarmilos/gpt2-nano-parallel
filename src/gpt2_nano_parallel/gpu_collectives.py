"""Measure NCCL collectives and analyze held-out topology-calibrated models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Mapping, Sequence

from .topology_planner import (
    COLLECTIVES,
    compare_holdout,
    constant_choice_baseline,
    error_summary,
    evaluate_planner,
    fit_collective_models,
    fit_shared_collective_model,
)

ELEMENT_BYTES = 4


@dataclass(frozen=True)
class NcclStudyConfig:
    world_sizes: tuple[int, ...]
    operations: tuple[str, ...]
    calibration_message_bytes: tuple[int, ...]
    validation_message_bytes: tuple[int, ...]
    warmup_iterations: int
    iterations: int
    timeout_seconds: float

    @property
    def message_bytes(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                set(self.calibration_message_bytes)
                | set(self.validation_message_bytes)
            )
        )


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must not be negative")
    return value


def _positive_float(value: object, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive number")
    return float(value)


def _int_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    parsed = tuple(_positive_int(item, name) for item in value)
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{name} must not contain duplicates")
    return parsed


def _operation_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("operations must be a nonempty list")
    parsed: list[str] = []
    for operation in value:
        if not isinstance(operation, str) or operation not in COLLECTIVES:
            raise ValueError(f"unsupported operation: {operation}")
        parsed.append(operation)
    if len(set(parsed)) != len(parsed):
        raise ValueError("operations must not contain duplicates")
    return tuple(parsed)


def load_nccl_config(path: str | Path) -> NcclStudyConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    section = raw.get("nccl_study")
    if not isinstance(section, dict):
        raise ValueError("nccl_study must be a table")
    config = NcclStudyConfig(
        world_sizes=_int_tuple(section.get("world_sizes"), "world_sizes"),
        operations=_operation_tuple(section.get("operations")),
        calibration_message_bytes=_int_tuple(
            section.get("calibration_message_bytes"),
            "calibration_message_bytes",
        ),
        validation_message_bytes=_int_tuple(
            section.get("validation_message_bytes"),
            "validation_message_bytes",
        ),
        warmup_iterations=_non_negative_int(
            section.get("warmup_iterations"), "warmup_iterations"
        ),
        iterations=_positive_int(section.get("iterations"), "iterations"),
        timeout_seconds=_positive_float(
            section.get("timeout_seconds"), "timeout_seconds"
        ),
    )
    if any(world_size <= 1 for world_size in config.world_sizes):
        raise ValueError("world_sizes must be greater than one")
    overlap = set(config.calibration_message_bytes) & set(
        config.validation_message_bytes
    )
    if overlap:
        raise ValueError(
            f"calibration and validation message sizes overlap: {sorted(overlap)}"
        )
    for message_bytes in config.message_bytes:
        if message_bytes % ELEMENT_BYTES:
            raise ValueError("message sizes must be multiples of four bytes")
        for world_size in config.world_sizes:
            if message_bytes % (world_size * ELEMENT_BYTES):
                raise ValueError(
                    "message sizes must divide evenly across every world size"
                )
    return config


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict[str, object]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "git_revision": revision,
        "git_dirty": bool(status),
        "git_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "git_state_captured": "study_start",
    }


def _topology() -> str:
    result = subprocess.run(
        ["nvidia-smi", "topo", "-m"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _atomic_json(path: str | Path, value: Mapping[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.new")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class _CollectiveState:
    run: Callable[[], None]
    reset: Callable[[], None]
    verify: Callable[[], None]


def _collective_state(
    operation: str,
    message_bytes: int,
    rank: int,
    local_rank: int,
    world_size: int,
    torch: Any,
    dist: Any,
) -> _CollectiveState:
    device = torch.device("cuda", local_rank)
    elements = message_bytes // ELEMENT_BYTES
    if operation == "all_reduce":
        tensor = torch.empty(elements, dtype=torch.float32, device=device)
        expected = float(world_size * (world_size + 1) / 2)

        def reset() -> None:
            tensor.fill_(rank + 1)

        def run() -> None:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        def verify() -> None:
            if not bool(torch.all(tensor == expected).item()):
                raise RuntimeError("all_reduce checksum failed")

        return _CollectiveState(run, reset, verify)

    if operation == "reduce_scatter":
        output_elements = elements // world_size
        source = torch.empty(elements, dtype=torch.float32, device=device)
        output = torch.empty(output_elements, dtype=torch.float32, device=device)
        expected = float(world_size * (world_size + 1) / 2)

        def reset() -> None:
            source.fill_(rank + 1)

        def run() -> None:
            dist.reduce_scatter_tensor(output, source, op=dist.ReduceOp.SUM)

        def verify() -> None:
            if not bool(torch.all(output == expected).item()):
                raise RuntimeError("reduce_scatter checksum failed")

        return _CollectiveState(run, reset, verify)

    if operation == "all_gather":
        input_elements = elements // world_size
        source = torch.full(
            (input_elements,),
            rank + 1,
            dtype=torch.float32,
            device=device,
        )
        output = torch.empty(elements, dtype=torch.float32, device=device)

        def reset() -> None:
            return None

        def run() -> None:
            dist.all_gather_into_tensor(output, source)

        def verify() -> None:
            reshaped = output.view(world_size, input_elements)
            expected = torch.arange(
                1,
                world_size + 1,
                dtype=torch.float32,
                device=device,
            )
            if not torch.equal(
                reshaped,
                expected[:, None].expand(world_size, input_elements),
            ):
                raise RuntimeError("all_gather checksum failed")

        return _CollectiveState(run, reset, verify)

    if operation == "all_to_all":
        chunk_elements = elements // world_size
        chunks = [
            torch.full(
                (chunk_elements,),
                rank * world_size + destination,
                dtype=torch.float32,
                device=device,
            )
            for destination in range(world_size)
        ]
        source = torch.cat(chunks)
        output = torch.empty_like(source)

        def reset() -> None:
            return None

        def run() -> None:
            dist.all_to_all_single(output, source)

        def verify() -> None:
            reshaped = output.view(world_size, chunk_elements)
            expected = torch.arange(
                world_size,
                dtype=torch.float32,
                device=device,
            ) * world_size + rank
            if not torch.equal(
                reshaped,
                expected[:, None].expand(world_size, chunk_elements),
            ):
                raise RuntimeError("all_to_all checksum failed")

        return _CollectiveState(run, reset, verify)

    if operation == "neighbor_exchange":
        source = torch.full(
            (elements,),
            rank + 1,
            dtype=torch.float32,
            device=device,
        )
        output = torch.empty_like(source)
        destination = (rank + 1) % world_size
        previous = (rank - 1) % world_size

        def reset() -> None:
            return None

        def run() -> None:
            requests = dist.batch_isend_irecv(
                [
                    dist.P2POp(dist.isend, source, destination),
                    dist.P2POp(dist.irecv, output, previous),
                ]
            )
            for request in requests:
                request.wait()

        def verify() -> None:
            if not bool(torch.all(output == float(previous + 1)).item()):
                raise RuntimeError("neighbor_exchange checksum failed")

        return _CollectiveState(run, reset, verify)

    raise ValueError(f"unsupported operation: {operation}")


def _measure_collective(
    operation: str,
    message_bytes: int,
    rank: int,
    local_rank: int,
    world_size: int,
    config: NcclStudyConfig,
    torch: Any,
    dist: Any,
) -> dict[str, object] | None:
    state = _collective_state(
        operation,
        message_bytes,
        rank,
        local_rank,
        world_size,
        torch,
        dist,
    )
    for _ in range(config.warmup_iterations):
        state.reset()
        dist.barrier()
        state.run()
    torch.cuda.synchronize()
    if config.warmup_iterations:
        state.verify()

    samples: list[float] = []
    for _ in range(config.iterations):
        state.reset()
        dist.barrier()
        torch.cuda.synchronize()
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        started.record()
        state.run()
        finished.record()
        finished.synchronize()
        samples.append(float(started.elapsed_time(finished)) / 1000.0)
    state.verify()

    local = torch.tensor(samples, dtype=torch.float64, device=torch.device("cuda", rank))
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    if rank != 0:
        return None
    rank_samples = [tensor.cpu().tolist() for tensor in gathered]
    maximum_samples = [
        max(rank_samples[worker][index] for worker in range(world_size))
        for index in range(config.iterations)
    ]
    return {
        "operation": operation,
        "world_size": world_size,
        "message_bytes": message_bytes,
        "split": (
            "calibration"
            if message_bytes in config.calibration_message_bytes
            else "validation"
        ),
        "warmup_iterations": config.warmup_iterations,
        "iterations": config.iterations,
        "measured_seconds": mean(maximum_samples),
        "median_seconds": median(maximum_samples),
        "min_seconds": min(maximum_samples),
        "max_seconds": max(maximum_samples),
        "samples_seconds": maximum_samples,
        "rank_mean_seconds": [mean(worker) for worker in rank_samples],
        "checksum": "verified",
    }


def measure(config_path: str | Path, output: str | Path) -> dict[str, object] | None:
    """Run one torchrun world size and write a raw measured artifact on rank zero."""

    source_state = _git_state()
    config_hash = _sha256(config_path)
    config = load_nccl_config(config_path)
    try:
        import torch
        import torch.distributed as dist
    except ImportError as error:
        raise RuntimeError("PyTorch with distributed CUDA support is required") from error

    if not dist.is_nccl_available():
        raise RuntimeError("PyTorch was built without NCCL support")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    group_rank = int(os.environ.get("GROUP_RANK", "0"))
    if (
        local_world_size != world_size
        or group_rank != 0
        or rank != local_rank
    ):
        raise RuntimeError("NCCL publication measurements require one local node")
    if world_size not in config.world_sizes:
        raise ValueError(
            f"torchrun world size {world_size} is not declared in the config"
        )
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError("LOCAL_RANK exceeds visible CUDA devices")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        timeout=timedelta(seconds=config.timeout_seconds),
    )
    wall_started = time.perf_counter()
    try:
        rows: list[dict[str, object]] = []
        for operation in config.operations:
            for message_bytes in config.message_bytes:
                row = _measure_collective(
                    operation,
                    message_bytes,
                    rank,
                    local_rank,
                    world_size,
                    config,
                    torch,
                    dist,
                )
                if row is not None:
                    rows.append(row)
        dist.barrier()
        if rank != 0:
            return None
        result: dict[str, object] = {
            "schema_version": "nccl-collective-raw-v1",
            "status": "measured",
            "evidence": "measured",
            "scope": (
                "single-node NCCL collective microbenchmarks; not end-to-end "
                "distributed model training"
            ),
            "created_at": datetime.now(UTC).isoformat(),
            "command": (
                f"torchrun --standalone --nproc_per_node={world_size} "
                "-m gpt2_nano_parallel.gpu_collectives measure "
                f"{Path(config_path).as_posix()} --output {Path(output).as_posix()}"
            ),
            "config_path": Path(config_path).as_posix(),
            "config_sha256": config_hash,
            "source": source_state,
            "environment": {
                "backend": "nccl",
                "device": "cuda",
                "world_size": world_size,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "nccl": ".".join(str(value) for value in torch.cuda.nccl.version()),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "hostname": platform.node(),
                "gpu_names": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
                "topology": _topology(),
            },
            "config": asdict(config),
            "wall_seconds": time.perf_counter() - wall_started,
            "measurements": rows,
        }
        _atomic_json(output, result)
        return result
    finally:
        dist.destroy_process_group()


def _load_raw(
    path: str | Path,
    config: NcclStudyConfig,
    config_sha256: str,
) -> dict[str, object]:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")
    if raw.get("schema_version") != "nccl-collective-raw-v1":
        raise ValueError(f"{path} has an unsupported schema")
    if raw.get("status") != "measured":
        raise ValueError(f"{path} is not measured")
    if raw.get("evidence") != "measured":
        raise ValueError(f"{path} has invalid evidence metadata")
    if raw.get("config_sha256") != config_sha256:
        raise ValueError(f"{path} does not match the analysis config")
    embedded_config = json.loads(json.dumps(asdict(config)))
    if raw.get("config") != embedded_config:
        raise ValueError(f"{path} does not embed the analysis config")
    source = raw.get("source")
    if (
        not isinstance(source, dict)
        or source.get("git_dirty") is not False
        or source.get("git_state_captured") != "study_start"
        or not isinstance(source.get("git_revision"), str)
        or not source["git_revision"]
    ):
        raise ValueError(f"{path} has invalid source provenance")
    environment = raw.get("environment")
    if (
        not isinstance(environment, dict)
        or environment.get("backend") != "nccl"
        or environment.get("device") != "cuda"
    ):
        raise ValueError(f"{path} has invalid environment metadata")
    _environment_identity(environment)
    world_size = environment.get("world_size")
    if world_size not in config.world_sizes:
        raise ValueError(f"{path} has an unexpected world size")
    measurements = raw.get("measurements")
    expected = len(config.operations) * len(config.message_bytes)
    if not isinstance(measurements, list) or len(measurements) != expected:
        raise ValueError(f"{path} has incomplete measurements")
    seen: set[tuple[str, int]] = set()
    for row in measurements:
        if not isinstance(row, dict):
            raise ValueError(f"{path} contains an invalid measurement")
        operation = row.get("operation")
        message_bytes = row.get("message_bytes")
        if operation not in config.operations or message_bytes not in config.message_bytes:
            raise ValueError(f"{path} contains unexpected dimensions")
        if row.get("world_size") != world_size:
            raise ValueError(f"{path} contains a mismatched row world size")
        expected_split = (
            "calibration"
            if message_bytes in config.calibration_message_bytes
            else "validation"
        )
        if row.get("split") != expected_split:
            raise ValueError(f"{path} contains an invalid split label")
        key = (str(operation), int(message_bytes))
        if key in seen:
            raise ValueError(f"{path} contains duplicate measurements")
        seen.add(key)
        if row.get("checksum") != "verified":
            raise ValueError(f"{path} contains an unverified checksum")
        measured = row.get("measured_seconds")
        if (
            not isinstance(measured, (int, float))
            or isinstance(measured, bool)
            or not math.isfinite(measured)
            or measured <= 0
        ):
            raise ValueError(f"{path} contains an invalid timing")
        if (
            row.get("warmup_iterations") != config.warmup_iterations
            or row.get("iterations") != config.iterations
        ):
            raise ValueError(f"{path} contains mismatched iteration counts")
        samples = row.get("samples_seconds")
        rank_means = row.get("rank_mean_seconds")
        if (
            not isinstance(samples, list)
            or len(samples) != config.iterations
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value > 0
                for value in samples
            )
            or not isinstance(rank_means, list)
            or len(rank_means) != world_size
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value > 0
                for value in rank_means
            )
        ):
            raise ValueError(f"{path} contains invalid timing samples")
        summaries = {
            "measured_seconds": mean(samples),
            "median_seconds": median(samples),
            "min_seconds": min(samples),
            "max_seconds": max(samples),
        }
        if any(
            not isinstance(row.get(name), (int, float))
            or isinstance(row.get(name), bool)
            or not math.isclose(
                float(row[name]),
                expected_value,
                rel_tol=1e-12,
                abs_tol=0.0,
            )
            for name, expected_value in summaries.items()
        ):
            raise ValueError(f"{path} contains inconsistent timing summaries")
    return raw


def _environment_identity(environment: Mapping[str, object]) -> tuple[object, ...]:
    string_fields = (
        "torch",
        "cuda",
        "nccl",
        "python",
        "platform",
        "hostname",
        "topology",
    )
    values = []
    for field in string_fields:
        value = environment.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"NCCL environment has invalid {field}")
        values.append(value)
    gpu_names = environment.get("gpu_names")
    world_size = environment.get("world_size")
    if (
        not isinstance(gpu_names, list)
        or not gpu_names
        or not all(isinstance(name, str) and name for name in gpu_names)
        or not isinstance(world_size, int)
        or isinstance(world_size, bool)
        or world_size > len(gpu_names)
    ):
        raise ValueError("NCCL environment has invalid GPU inventory")
    return (*values, tuple(gpu_names))


def _validate_shared_environment(
    raw_artifacts: Sequence[Mapping[str, object]],
) -> None:
    identities = {
        _environment_identity(artifact["environment"])  # type: ignore[arg-type]
        for artifact in raw_artifacts
    }
    if len(identities) != 1:
        raise ValueError("raw artifacts were measured on different GPU hosts")


def _derive_study_analysis(
    config: NcclStudyConfig,
    raw_artifacts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    measurements = [
        dict(row)
        for artifact in raw_artifacts
        for row in artifact["measurements"]  # type: ignore[union-attr]
    ]
    calibration = [
        row
        for row in measurements
        if row["message_bytes"] in config.calibration_message_bytes
    ]
    holdout = [
        row
        for row in measurements
        if row["message_bytes"] in config.validation_message_bytes
    ]
    expected_calibration = (
        len(config.world_sizes)
        * len(config.operations)
        * len(config.calibration_message_bytes)
    )
    expected_holdout = (
        len(config.world_sizes)
        * len(config.operations)
        * len(config.validation_message_bytes)
    )
    if len(calibration) != expected_calibration or len(holdout) != expected_holdout:
        raise ValueError("calibration and validation rows are incomplete")

    collective_models = fit_collective_models(calibration)
    shared_model = fit_shared_collective_model(calibration)
    comparisons = compare_holdout(holdout, collective_models, shared_model)
    return {
        "collective_models": collective_models,
        "shared_ring_model": shared_model,
        "held_out": {
            "collective_specific_error": error_summary(
                comparisons, "collective_model_relative_error"
            ),
            "shared_ring_error": error_summary(
                comparisons, "shared_model_relative_error"
            ),
            "comparisons": comparisons,
        },
        "planner": evaluate_planner(holdout, collective_models),
    }


def analyze(
    config_path: str | Path,
    raw_paths: Sequence[str | Path],
    output: str | Path,
) -> dict[str, object]:
    """Fit timing models and evaluate held-out predictions and compositions."""

    source_state = _git_state()
    config_hash = _sha256(config_path)
    config = load_nccl_config(config_path)
    raw_artifacts = [_load_raw(path, config, config_hash) for path in raw_paths]
    world_sizes = sorted(
        int(artifact["environment"]["world_size"])  # type: ignore[index]
        for artifact in raw_artifacts
    )
    if world_sizes != sorted(config.world_sizes):
        raise ValueError("raw artifacts do not cover every configured world size once")
    revisions = {
        str(artifact["source"]["git_revision"])  # type: ignore[index]
        for artifact in raw_artifacts
    }
    if len(revisions) != 1:
        raise ValueError("raw artifacts were produced from different source revisions")
    if any(
        bool(artifact["source"]["git_dirty"])  # type: ignore[index]
        for artifact in raw_artifacts
    ):
        raise ValueError("raw artifacts must be measured from clean source trees")
    _validate_shared_environment(raw_artifacts)
    if source_state["git_dirty"] or source_state["git_revision"] not in revisions:
        raise ValueError("analysis must use the clean measurement source revision")
    derived = _derive_study_analysis(config, raw_artifacts)
    planner = derived["planner"]
    baseline = constant_choice_baseline(planner["scenarios"])  # type: ignore[index]
    baseline_label = str(baseline["choice"]).replace("_", " ")
    if baseline["choice"] == "context_parallel":
        baseline_label += " all-gather proxy"
    result: dict[str, object] = {
        "schema_version": "nccl-topology-study-v1",
        "status": "measured",
        "evidence": "measured",
        "scope": (
            "single-node NCCL collective timing prediction with held-out message "
            "sizes and a communication-composition planner"
        ),
        "created_at": datetime.now(UTC).isoformat(),
        "command": (
            "python -m gpt2_nano_parallel.gpu_collectives analyze "
            f"{Path(config_path).as_posix()} "
            + " ".join(Path(path).as_posix() for path in raw_paths)
            + f" --output {Path(output).as_posix()}"
        ),
        "config_path": Path(config_path).as_posix(),
        "config_sha256": config_hash,
        "source": source_state,
        "measurement_source_revision": next(iter(revisions)),
        "raw_artifacts": [
            {
                "path": Path(path).name,
                "sha256": _sha256(path),
                "world_size": artifact["environment"]["world_size"],  # type: ignore[index]
                "environment": artifact["environment"],
            }
            for path, artifact in zip(raw_paths, raw_artifacts, strict=True)
        ],
        "splits": {
            "calibration_message_bytes": list(config.calibration_message_bytes),
            "validation_message_bytes": list(config.validation_message_bytes),
            "overlap": False,
        },
        **derived,
        "caveats": [
            "The measurements are communication microbenchmarks, not end-to-end model training.",
            "The planner ranks declared collective compositions only and excludes compute, overlap, optimizer behavior, kernel fusion, and convergence.",
            (
                f"The planner scores {planner['correct_scenarios']}/"
                f"{planner['total_scenarios']}; always choosing "
                f"{baseline_label} scores {baseline['correct_scenarios']}/"
                f"{baseline['total_scenarios']}, so the planner does not beat "
                "that constant baseline."
            ),
            "Pipeline uses a simultaneous neighbor exchange proxy rather than a complete pipeline schedule.",
            "Context parallelism uses an all-gather proxy rather than a fused ring-attention kernel.",
            "The fitted constants describe only the recorded single-node GPU topology.",
        ],
    }
    _atomic_json(output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpt2-parallel-nccl")
    subparsers = parser.add_subparsers(dest="command", required=True)
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("config")
    measure_parser.add_argument("--output", required=True)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("config")
    analyze_parser.add_argument("raw_artifacts", nargs="+")
    analyze_parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "measure":
        result = measure(args.config, args.output)
        if result is not None:
            print(
                json.dumps(
                    {
                        "artifact": args.output,
                        "measurements": len(result["measurements"]),
                        "world_size": result["environment"]["world_size"],
                    },
                    indent=2,
                )
            )
    else:
        result = analyze(args.config, args.raw_artifacts, args.output)
        print(
            json.dumps(
                {
                    "artifact": args.output,
                    "operation_specific_mare": result["held_out"][
                        "collective_specific_error"
                    ]["mean_absolute_relative_error"],
                    "shared_model_mare": result["held_out"]["shared_ring_error"][
                        "mean_absolute_relative_error"
                    ],
                    "planner_top1_accuracy": result["planner"]["top1_accuracy"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
