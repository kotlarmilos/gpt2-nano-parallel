"""Validate the analytical ring all-reduce model against local Gloo measurements."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from queue import Empty
from statistics import median
from typing import Any, Mapping, Sequence

from .communication import ring_collective
from .config import load_config

DEFAULT_WORLD_SIZES = (2, 4, 8)
DEFAULT_MESSAGE_BYTES = (4, 256, 4096, 65536, 1048576, 8388608, 33554432)
DEFAULT_FIT_MIN_MESSAGE_BYTES = 8388608
ELEMENT_BYTES = 4


@dataclass(frozen=True)
class AllReduceValidationConfig:
    world_sizes: tuple[int, ...] = DEFAULT_WORLD_SIZES
    message_bytes: tuple[int, ...] = DEFAULT_MESSAGE_BYTES
    warmup_iterations: int = 3
    iterations: int = 12
    timeout_seconds: float = 180.0
    fit_min_message_bytes: int = DEFAULT_FIT_MIN_MESSAGE_BYTES


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)


def _float_field(record: Mapping[str, object], key: str) -> float:
    value = record.get(key)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{key} must be a finite number")
    return float(value)


def _int_field(record: Mapping[str, object], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _positive_int_tuple(values: object, key: str) -> tuple[int, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{key} must be a nonempty list")
    parsed: list[int] = []
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{key} must contain positive integers")
        parsed.append(value)
    return tuple(parsed)


def _positive_int(value: object, key: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _non_negative_int(value: object, key: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{key} must not be negative")
    return value


def _positive_float(value: object, key: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{key} must be a positive number")
    return float(value)


def _validate_config(config: AllReduceValidationConfig) -> AllReduceValidationConfig:
    if any(size <= 1 for size in config.world_sizes):
        raise ValueError("world_sizes must be greater than one")
    if len(set(config.world_sizes)) != len(config.world_sizes):
        raise ValueError("world_sizes must not contain duplicates")
    if len(set(config.message_bytes)) != len(config.message_bytes):
        raise ValueError("message_bytes must not contain duplicates")
    if any(size <= 0 or size % ELEMENT_BYTES != 0 for size in config.message_bytes):
        raise ValueError("message_bytes must be positive multiples of four")
    if config.warmup_iterations < 0:
        raise ValueError("warmup_iterations must not be negative")
    if config.iterations <= 0:
        raise ValueError("iterations must be positive")
    if config.timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if config.fit_min_message_bytes <= 0:
        raise ValueError("fit_min_message_bytes must be positive")
    return config


def load_validation_config(path: str | Path) -> AllReduceValidationConfig:
    """Load the all-reduce validation section from a repository config."""

    raw = load_config(path)
    section = raw.get("allreduce_validation", {})
    if not isinstance(section, dict):
        raise ValueError("allreduce_validation must be a table")
    config = AllReduceValidationConfig(
        world_sizes=_positive_int_tuple(
            section.get("world_sizes", list(DEFAULT_WORLD_SIZES)), "world_sizes"
        ),
        message_bytes=_positive_int_tuple(
            section.get("message_bytes", list(DEFAULT_MESSAGE_BYTES)),
            "message_bytes",
        ),
        warmup_iterations=_non_negative_int(
            section.get("warmup_iterations", 3), "warmup_iterations"
        ),
        iterations=_positive_int(section.get("iterations", 12), "iterations"),
        timeout_seconds=_positive_float(
            section.get("timeout_seconds", 180.0), "timeout_seconds"
        ),
        fit_min_message_bytes=_positive_int(
            section.get(
                "fit_min_message_bytes", DEFAULT_FIT_MIN_MESSAGE_BYTES
            ),
            "fit_min_message_bytes",
        ),
    )
    return _validate_config(config)


def ring_latency_steps(world_size: int) -> int:
    """Return the latency steps used by the ring all-reduce time model."""

    if world_size <= 1:
        raise ValueError("world_size must be greater than one")
    return 2 * (world_size - 1)


def analytical_ring_allreduce_seconds(
    message_bytes: int,
    world_size: int,
    bandwidth_bytes_per_second: float,
    latency_seconds: float,
) -> float:
    """Return ring all-reduce time from the analytical byte model."""

    if message_bytes <= 0:
        raise ValueError("message_bytes must be positive")
    if bandwidth_bytes_per_second <= 0:
        raise ValueError("bandwidth_bytes_per_second must be positive")
    if latency_seconds < 0:
        raise ValueError("latency_seconds must not be negative")
    bytes_per_rank = ring_collective(
        message_bytes, world_size, "all_reduce"
    ).bytes_per_rank
    return (
        bytes_per_rank / bandwidth_bytes_per_second
        + ring_latency_steps(world_size) * latency_seconds
    )


def fit_ring_constants(
    measurements: Sequence[Mapping[str, object]],
    *,
    min_message_bytes: int = DEFAULT_FIT_MIN_MESSAGE_BYTES,
) -> dict[str, object]:
    """Fit effective bandwidth and per-step latency constants."""

    if min_message_bytes <= 0:
        raise ValueError("min_message_bytes must be positive")
    points = [
        record
        for record in measurements
        if _int_field(record, "message_bytes") >= min_message_bytes
    ]
    if len(points) < 2:
        raise ValueError("at least two measurements are required for fitting")

    features: list[tuple[float, float, float]] = []
    for record in points:
        message_bytes = _int_field(record, "message_bytes")
        world_size = _int_field(record, "world_size")
        measured_seconds = _float_field(record, "measured_seconds")
        if measured_seconds <= 0:
            raise ValueError("measured_seconds must be positive")
        bytes_per_rank = ring_collective(
            message_bytes, world_size, "all_reduce"
        ).bytes_per_rank
        features.append(
            (
                float(bytes_per_rank),
                float(ring_latency_steps(world_size)),
                measured_seconds,
            )
        )

    def squared_error(beta: float, alpha: float) -> float:
        return sum((x * beta + z * alpha - y) ** 2 for x, z, y in features)

    sxx = sum(x * x for x, _, _ in features)
    sxz = sum(x * z for x, z, _ in features)
    szz = sum(z * z for _, z, _ in features)
    sxy = sum(x * y for x, _, y in features)
    szy = sum(z * y for _, z, y in features)
    determinant = sxx * szz - sxz * sxz
    candidates: list[tuple[float, float, float, str]] = []
    if abs(determinant) > 1e-30:
        beta = (sxy * szz - szy * sxz) / determinant
        alpha = (sxx * szy - sxz * sxy) / determinant
        if beta > 0 and alpha >= 0:
            candidates.append((squared_error(beta, alpha), beta, alpha, "least_squares"))
    if sxx > 0:
        beta = sxy / sxx
        if beta > 0:
            candidates.append((squared_error(beta, 0.0), beta, 0.0, "zero_latency"))
    if not candidates:
        raise ValueError("could not fit positive analytical constants")

    error, beta, alpha, method = min(candidates, key=lambda item: item[0])
    if beta <= 0:
        raise ValueError("fit produced a nonpositive bandwidth")
    bandwidth = 1.0 / beta
    return {
        "bandwidth_bytes_per_second": bandwidth,
        "bandwidth_gib_per_second": bandwidth / 2**30,
        "latency_seconds": alpha,
        "latency_microseconds": alpha * 1_000_000,
        "fit_min_message_bytes": min_message_bytes,
        "fit_points": len(points),
        "method": method,
        "sum_squared_error": error,
    }


def compare_measurements(
    measurements: Sequence[Mapping[str, object]],
    bandwidth_bytes_per_second: float,
    latency_seconds: float,
) -> list[dict[str, object]]:
    """Compare measured Gloo time with the analytical ring model."""

    rows: list[dict[str, object]] = []
    for record in measurements:
        message_bytes = _int_field(record, "message_bytes")
        world_size = _int_field(record, "world_size")
        measured_seconds = _float_field(record, "measured_seconds")
        model_seconds = analytical_ring_allreduce_seconds(
            message_bytes,
            world_size,
            bandwidth_bytes_per_second,
            latency_seconds,
        )
        signed_error = measured_seconds - model_seconds
        rows.append(
            {
                **dict(record),
                "ring_bytes_per_rank": ring_collective(
                    message_bytes, world_size, "all_reduce"
                ).bytes_per_rank,
                "latency_steps": ring_latency_steps(world_size),
                "model_seconds": model_seconds,
                "signed_error_seconds": signed_error,
                "absolute_error_seconds": abs(signed_error),
                "relative_error": (
                    abs(signed_error) / measured_seconds
                    if measured_seconds > 0
                    else math.inf
                ),
            }
        )
    return sorted(rows, key=lambda row: (row["world_size"], row["message_bytes"]))


def residual_summary(
    comparisons: Sequence[Mapping[str, object]],
    *,
    min_message_bytes: int | None = None,
    max_message_bytes: int | None = None,
) -> dict[str, object]:
    """Summarize residual errors for a selected message range."""

    selected: list[Mapping[str, object]] = []
    for record in comparisons:
        message_bytes = _int_field(record, "message_bytes")
        if min_message_bytes is not None and message_bytes < min_message_bytes:
            continue
        if max_message_bytes is not None and message_bytes > max_message_bytes:
            continue
        selected.append(record)
    if not selected:
        return {
            "count": 0,
            "mean_absolute_relative_error": None,
            "max_absolute_relative_error": None,
            "root_mean_squared_error_seconds": None,
            "mean_signed_error_seconds": None,
        }
    relative_errors = [
        _float_field(record, "relative_error") for record in selected
    ]
    signed_errors = [
        _float_field(record, "signed_error_seconds") for record in selected
    ]
    return {
        "count": len(selected),
        "mean_absolute_relative_error": _mean(relative_errors),
        "max_absolute_relative_error": max(relative_errors),
        "root_mean_squared_error_seconds": math.sqrt(
            _mean([error * error for error in signed_errors])
        ),
        "mean_signed_error_seconds": _mean(signed_errors),
    }


def analyze_measurements(
    measurements: Sequence[Mapping[str, object]],
    *,
    fit_min_message_bytes: int = DEFAULT_FIT_MIN_MESSAGE_BYTES,
) -> dict[str, object]:
    """Fit constants and attach model predictions to measurements."""

    fit = fit_ring_constants(
        measurements, min_message_bytes=fit_min_message_bytes
    )
    comparisons = compare_measurements(
        measurements,
        float(fit["bandwidth_bytes_per_second"]),
        float(fit["latency_seconds"]),
    )
    return {
        "fitted_constants": fit,
        "residuals": {
            "all_messages": residual_summary(comparisons),
            "fit_range": residual_summary(
                comparisons, min_message_bytes=fit_min_message_bytes
            ),
            "small_messages": residual_summary(
                comparisons, max_message_bytes=fit_min_message_bytes - 1
            ),
        },
        "measurements": comparisons,
    }


def aggregate_rank_measurements(
    rank_results: Sequence[Mapping[str, object]],
    *,
    world_size: int,
    message_bytes: Sequence[int],
    iterations: int,
    warmup_iterations: int,
) -> list[dict[str, object]]:
    """Aggregate per-rank Gloo timings into measured collective timings."""

    if len(rank_results) != world_size:
        raise ValueError("rank_results length must match world_size")
    by_rank: dict[int, Mapping[str, object]] = {}
    for result in rank_results:
        rank = _int_field(result, "rank")
        if rank not in range(world_size) or rank in by_rank:
            raise ValueError("rank_results contain invalid or duplicate ranks")
        by_rank[rank] = result

    rows: list[dict[str, object]] = []
    for size in message_bytes:
        per_rank_samples: list[list[float]] = []
        for rank in range(world_size):
            measurements = by_rank[rank].get("measurements")
            if not isinstance(measurements, list):
                raise ValueError("rank result measurements must be a list")
            matching = [
                item
                for item in measurements
                if isinstance(item, dict)
                and _int_field(item, "message_bytes") == size
            ]
            if len(matching) != 1:
                raise ValueError("rank result must contain each message size once")
            samples = matching[0].get("elapsed_seconds")
            if not isinstance(samples, list) or len(samples) != iterations:
                raise ValueError("elapsed_seconds must contain every iteration")
            parsed_samples = []
            for sample in samples:
                if (
                    not isinstance(sample, (int, float))
                    or isinstance(sample, bool)
                    or not math.isfinite(sample)
                    or sample < 0
                ):
                    raise ValueError("elapsed_seconds contains an invalid sample")
                parsed_samples.append(float(sample))
            per_rank_samples.append(parsed_samples)

        max_rank_samples = [
            max(per_rank_samples[rank][index] for rank in range(world_size))
            for index in range(iterations)
        ]
        rows.append(
            {
                "world_size": world_size,
                "message_bytes": size,
                "iterations": iterations,
                "warmup_iterations": warmup_iterations,
                "measured_seconds": _mean(max_rank_samples),
                "median_seconds": median(max_rank_samples),
                "min_seconds": min(max_rank_samples),
                "max_seconds": max(max_rank_samples),
                "rank_mean_seconds": [
                    _mean(samples) for samples in per_rank_samples
                ],
                "samples_seconds": max_rank_samples,
            }
        )
    return rows


def _worker(
    rank: int,
    world_size: int,
    port: int,
    message_bytes: tuple[int, ...],
    warmup_iterations: int,
    iterations: int,
    timeout_seconds: float,
    queue: Any,
) -> None:
    dist = None
    try:
        import torch
        import torch.distributed as dist

        torch.set_num_threads(1)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=timeout_seconds),
        )
        rank_measurements: list[dict[str, object]] = []
        expected = float(world_size * (world_size + 1) / 2)
        for size in message_bytes:
            tensor = torch.empty(size // ELEMENT_BYTES, dtype=torch.float32)
            for _ in range(warmup_iterations):
                tensor.fill_(rank + 1)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            dist.barrier()
            elapsed_samples: list[float] = []
            for _ in range(iterations):
                tensor.fill_(rank + 1)
                started = time.perf_counter()
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                elapsed_samples.append(time.perf_counter() - started)
                if float(tensor[0].item()) != expected:
                    raise RuntimeError("all_reduce returned an invalid checksum")
            dist.barrier()
            rank_measurements.append(
                {
                    "message_bytes": size,
                    "elapsed_seconds": elapsed_samples,
                    "checksum": expected,
                }
            )
        queue.put(
            {
                "kind": "result",
                "rank": rank,
                "measurements": rank_measurements,
            }
        )
    except BaseException as error:
        queue.put(
            {
                "kind": "error",
                "rank": rank,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        raise
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


def _run_world_size(
    world_size: int, config: AllReduceValidationConfig
) -> list[dict[str, object]]:
    try:
        import torch.multiprocessing as mp
    except ImportError as error:
        raise RuntimeError("PyTorch is required for Gloo measurement") from error

    context = mp.get_context("spawn")
    queue = context.Queue()
    port = _free_port()
    processes = [
        context.Process(
            target=_worker,
            args=(
                rank,
                world_size,
                port,
                config.message_bytes,
                config.warmup_iterations,
                config.iterations,
                config.timeout_seconds,
                queue,
            ),
        )
        for rank in range(world_size)
    ]
    started_processes: list[Any] = []
    rank_results: dict[int, Mapping[str, object]] = {}
    deadline = time.perf_counter() + config.timeout_seconds
    try:
        for process in processes:
            process.start()
            started_processes.append(process)

        while len(rank_results) < world_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(
                    f"Gloo sweep timed out after {config.timeout_seconds:.1f} seconds"
                )
            try:
                message = queue.get(timeout=min(0.1, remaining))
            except Empty:
                failed = [
                    process
                    for process in started_processes
                    if process.exitcode not in {None, 0}
                ]
                if failed:
                    exits = ", ".join(
                        f"pid {process.pid} code {process.exitcode}"
                        for process in failed
                    )
                    raise RuntimeError(f"Gloo worker failed: {exits}")
                continue
            if not isinstance(message, dict):
                raise RuntimeError("Gloo worker returned an invalid message")
            if message.get("kind") == "error":
                raise RuntimeError(
                    f"Gloo worker {message.get('rank')} failed: "
                    f"{message.get('error')}"
                )
            if message.get("kind") != "result":
                raise RuntimeError("Gloo worker returned an invalid message")
            rank = message.get("rank")
            if not isinstance(rank, int) or rank not in range(world_size):
                raise RuntimeError("Gloo worker returned an invalid rank")
            if rank in rank_results:
                raise RuntimeError("Gloo worker returned a duplicate rank")
            rank_results[rank] = message

        for process in started_processes:
            process.join(timeout=max(0.0, deadline - time.perf_counter()))
            if process.is_alive():
                raise TimeoutError(
                    f"Gloo worker {process.pid} did not exit before the deadline"
                )
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Gloo worker {process.pid} exited with code {process.exitcode}"
                )
        return aggregate_rank_measurements(
            [rank_results[rank] for rank in range(world_size)],
            world_size=world_size,
            message_bytes=config.message_bytes,
            iterations=config.iterations,
            warmup_iterations=config.warmup_iterations,
        )
    finally:
        for process in started_processes:
            if process.is_alive():
                process.terminate()
        for process in started_processes:
            process.join(timeout=5)
        for process in started_processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        queue.close()
        queue.join_thread()


def run_gloo_allreduce_sweep(
    config: AllReduceValidationConfig | None = None,
) -> dict[str, object]:
    """Run the real local Gloo sweep and compare it with the model."""

    config = _validate_config(config or AllReduceValidationConfig())
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for Gloo measurement") from error

    wall_started = time.perf_counter()
    measurements: list[dict[str, object]] = []
    for world_size in config.world_sizes:
        measurements.extend(_run_world_size(world_size, config))

    analysis = analyze_measurements(
        measurements, fit_min_message_bytes=config.fit_min_message_bytes
    )
    return {
        "schema_version": 1,
        "generated_by": "gpt2_nano_parallel.allreduce_validation.run_gloo_allreduce_sweep",
        "evidence": "measured",
        "scope": (
            "local CPU processes using Gloo on 127.0.0.1; "
            "validates the communication cost model only; "
            "not a GPU scaling result"
        ),
        "backend": "gloo",
        "device": "cpu",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "wall_seconds": time.perf_counter() - wall_started,
        "config": asdict(config),
        "analytical_model": {
            "source_function": "gpt2_nano_parallel.communication.ring_collective",
            "operation": "all_reduce",
            "latency_steps": "2 * (world_size - 1)",
            "time_formula": (
                "ring_collective(message_bytes, world_size, 'all_reduce')"
                ".bytes_per_rank / bandwidth_bytes_per_second "
                "+ 2 * (world_size - 1) * latency_seconds"
            ),
        },
        **analysis,
    }


def write_artifact(path: str | Path, result: Mapping[str, object]) -> None:
    """Write a JSON artifact atomically inside the repository."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.new")
    try:
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def run_from_config(path: str | Path, output: str | Path) -> dict[str, object]:
    """Run the configured sweep and write the measured artifact."""

    result = run_gloo_allreduce_sweep(load_validation_config(path))
    write_artifact(output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpt2-parallel-allreduce")
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/allreduce_gloo.toml",
        help="TOML config path",
    )
    parser.add_argument(
        "--output",
        default="artifacts/gloo_allreduce_model_validation.json",
        help="JSON artifact path",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_from_config(args.config, args.output)
    summary = {
        "artifact": args.output,
        "measurements": len(result["measurements"]),
        "bandwidth_gib_per_second": result["fitted_constants"][
            "bandwidth_gib_per_second"
        ],
        "latency_microseconds": result["fitted_constants"][
            "latency_microseconds"
        ],
        "fit_range_mean_absolute_relative_error": result["residuals"][
            "fit_range"
        ]["mean_absolute_relative_error"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
