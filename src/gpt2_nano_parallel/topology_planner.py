"""Fit measured collective models and rank communication-only strategy plans."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Mapping, Sequence

from .communication import ring_collective

COLLECTIVES = (
    "all_reduce",
    "reduce_scatter",
    "all_gather",
    "all_to_all",
    "neighbor_exchange",
)


def _finite_float(value: object, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def collective_bytes_per_rank(
    operation: str, message_bytes: int, world_size: int
) -> float:
    """Return the modeled transmitted bytes per rank for one collective."""

    _positive_int(message_bytes, "message_bytes")
    if world_size <= 1:
        raise ValueError("world_size must be greater than one")
    if operation == "neighbor_exchange":
        return float(message_bytes)
    if operation not in COLLECTIVES:
        raise ValueError(f"unsupported operation: {operation}")
    return ring_collective(message_bytes, world_size, operation).bytes_per_rank


def collective_latency_steps(operation: str, world_size: int) -> int:
    """Return the simple ring or pairwise latency-step count."""

    if world_size <= 1:
        raise ValueError("world_size must be greater than one")
    if operation == "all_reduce":
        return 2 * (world_size - 1)
    if operation in {"reduce_scatter", "all_gather", "all_to_all"}:
        return world_size - 1
    if operation == "neighbor_exchange":
        return 1
    raise ValueError(f"unsupported operation: {operation}")


def collective_seconds(
    operation: str,
    message_bytes: int,
    world_size: int,
    bandwidth_bytes_per_second: float,
    latency_seconds: float,
) -> float:
    """Predict one collective with an effective bandwidth and latency model."""

    bandwidth = _finite_float(
        bandwidth_bytes_per_second, "bandwidth_bytes_per_second"
    )
    latency = _finite_float(latency_seconds, "latency_seconds")
    if bandwidth <= 0:
        raise ValueError("bandwidth_bytes_per_second must be positive")
    if latency < 0:
        raise ValueError("latency_seconds must not be negative")
    return (
        collective_bytes_per_rank(operation, message_bytes, world_size) / bandwidth
        + collective_latency_steps(operation, world_size) * latency
    )


def _measurement_field(record: Mapping[str, object], key: str) -> float:
    value = _finite_float(record.get(key), key)
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _measurement_dimensions(
    record: Mapping[str, object],
) -> tuple[str, int, int]:
    operation = record.get("operation")
    if not isinstance(operation, str) or operation not in COLLECTIVES:
        raise ValueError("measurement has an unsupported operation")
    world_size = _positive_int(record.get("world_size"), "world_size")
    if world_size <= 1:
        raise ValueError("world_size must be greater than one")
    message_bytes = _positive_int(record.get("message_bytes"), "message_bytes")
    return operation, world_size, message_bytes


def _fit_features(
    features: Sequence[tuple[float, float, float]],
) -> dict[str, float | str | int]:
    if len(features) < 2:
        raise ValueError("at least two measurements are required for fitting")
    if len({x for x, _, _ in features}) < 2:
        raise ValueError("fitting requires at least two message sizes")

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
        raise ValueError("could not fit positive collective constants")
    error, beta, alpha, method = min(candidates, key=lambda value: value[0])
    bandwidth = 1.0 / beta
    return {
        "bandwidth_bytes_per_second": bandwidth,
        "bandwidth_gib_per_second": bandwidth / 2**30,
        "latency_seconds": alpha,
        "latency_microseconds": alpha * 1_000_000,
        "fit_points": len(features),
        "method": method,
        "sum_squared_error": error,
    }


def fit_collective_models(
    measurements: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Fit one effective model for each operation and measured topology."""

    grouped: dict[tuple[str, int], list[tuple[float, float, float]]] = {}
    for record in measurements:
        operation, world_size, message_bytes = _measurement_dimensions(record)
        measured = _measurement_field(record, "measured_seconds")
        grouped.setdefault((operation, world_size), []).append(
            (
                collective_bytes_per_rank(operation, message_bytes, world_size),
                float(collective_latency_steps(operation, world_size)),
                measured,
            )
        )
    models: dict[str, dict[str, object]] = {}
    for (operation, world_size), features in sorted(grouped.items()):
        key = model_key(operation, world_size)
        models[key] = {
            "operation": operation,
            "world_size": world_size,
            **_fit_features(features),
        }
    return models


def fit_shared_ring_model(
    measurements: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Fit one operation-agnostic model to the calibration measurements."""

    features: list[tuple[float, float, float]] = []
    for record in measurements:
        operation, world_size, message_bytes = _measurement_dimensions(record)
        features.append(
            (
                collective_bytes_per_rank(operation, message_bytes, world_size),
                float(collective_latency_steps(operation, world_size)),
                _measurement_field(record, "measured_seconds"),
            )
        )
    return {
        "scope": "one operation-agnostic model across every collective and world size",
        **_fit_features(features),
    }


def model_key(operation: str, world_size: int) -> str:
    if operation not in COLLECTIVES:
        raise ValueError(f"unsupported operation: {operation}")
    if world_size <= 1:
        raise ValueError("world_size must be greater than one")
    return f"{operation}:world_size={world_size}"


def compare_holdout(
    measurements: Sequence[Mapping[str, object]],
    collective_models: Mapping[str, Mapping[str, object]],
    shared_model: Mapping[str, object],
) -> list[dict[str, object]]:
    """Compare primitive-specific and shared models on held-out measurements."""

    rows: list[dict[str, object]] = []
    shared_bandwidth = _measurement_field(
        shared_model, "bandwidth_bytes_per_second"
    )
    shared_latency = _finite_float(shared_model.get("latency_seconds"), "latency_seconds")
    for record in measurements:
        operation, world_size, message_bytes = _measurement_dimensions(record)
        measured = _measurement_field(record, "measured_seconds")
        key = model_key(operation, world_size)
        model = collective_models.get(key)
        if model is None:
            raise ValueError(f"missing collective model for {key}")
        predicted = collective_seconds(
            operation,
            message_bytes,
            world_size,
            _measurement_field(model, "bandwidth_bytes_per_second"),
            _finite_float(model.get("latency_seconds"), "latency_seconds"),
        )
        shared_predicted = collective_seconds(
            operation,
            message_bytes,
            world_size,
            shared_bandwidth,
            shared_latency,
        )
        rows.append(
            {
                **dict(record),
                "collective_model_seconds": predicted,
                "collective_model_relative_error": abs(predicted - measured) / measured,
                "shared_model_seconds": shared_predicted,
                "shared_model_relative_error": abs(shared_predicted - measured)
                / measured,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            int(row["world_size"]),
            str(row["operation"]),
            int(row["message_bytes"]),
        ),
    )


def error_summary(
    comparisons: Sequence[Mapping[str, object]], key: str
) -> dict[str, float | int]:
    if not comparisons:
        raise ValueError("comparisons must not be empty")
    errors = [_finite_float(record.get(key), key) for record in comparisons]
    return {
        "count": len(errors),
        "mean_absolute_relative_error": mean(errors),
        "max_absolute_relative_error": max(errors),
        "root_mean_squared_relative_error": math.sqrt(
            mean([error * error for error in errors])
        ),
    }


@dataclass(frozen=True)
class CollectiveCall:
    operation: str
    message_bytes: int
    count: int = 1

    def __post_init__(self) -> None:
        if self.operation not in COLLECTIVES:
            raise ValueError(f"unsupported operation: {self.operation}")
        _positive_int(self.message_bytes, "message_bytes")
        _positive_int(self.count, "count")


@dataclass(frozen=True)
class StrategyCandidate:
    name: str
    calls: tuple[CollectiveCall, ...]
    memory_bytes_per_rank: int = 0
    evidence: str = "communication-only model"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("candidate name must not be empty")
        if not self.calls:
            raise ValueError("candidate calls must not be empty")
        if self.memory_bytes_per_rank < 0:
            raise ValueError("memory_bytes_per_rank must not be negative")


def _model_for(
    models: Mapping[str, Mapping[str, object]], operation: str, world_size: int
) -> Mapping[str, object]:
    key = model_key(operation, world_size)
    model = models.get(key)
    if model is None:
        raise ValueError(f"missing collective model for {key}")
    return model


def predict_candidate_seconds(
    candidate: StrategyCandidate,
    world_size: int,
    models: Mapping[str, Mapping[str, object]],
) -> float:
    total = 0.0
    for call in candidate.calls:
        model = _model_for(models, call.operation, world_size)
        total += call.count * collective_seconds(
            call.operation,
            call.message_bytes,
            world_size,
            _measurement_field(model, "bandwidth_bytes_per_second"),
            _finite_float(model.get("latency_seconds"), "latency_seconds"),
        )
    return total


def measured_candidate_seconds(
    candidate: StrategyCandidate,
    world_size: int,
    measurements: Sequence[Mapping[str, object]],
) -> float:
    lookup: dict[tuple[str, int, int], float] = {}
    for record in measurements:
        operation, measured_world_size, message_bytes = _measurement_dimensions(record)
        key = (operation, measured_world_size, message_bytes)
        if key in lookup:
            raise ValueError(f"duplicate measurement for {key}")
        lookup[key] = _measurement_field(record, "measured_seconds")
    total = 0.0
    for call in candidate.calls:
        key = (call.operation, world_size, call.message_bytes)
        if key not in lookup:
            raise ValueError(f"missing held-out measurement for {key}")
        total += call.count * lookup[key]
    return total


def rank_candidates(
    candidates: Sequence[StrategyCandidate],
    world_size: int,
    models: Mapping[str, Mapping[str, object]],
    *,
    memory_budget_bytes: int | None = None,
) -> list[dict[str, object]]:
    """Rank feasible candidates by predicted communication time only."""

    if not candidates:
        raise ValueError("candidates must not be empty")
    if memory_budget_bytes is not None and memory_budget_bytes <= 0:
        raise ValueError("memory_budget_bytes must be positive")
    rows = []
    for candidate in candidates:
        feasible = (
            memory_budget_bytes is None
            or candidate.memory_bytes_per_rank <= memory_budget_bytes
        )
        rows.append(
            {
                "name": candidate.name,
                "feasible": feasible,
                "memory_bytes_per_rank": candidate.memory_bytes_per_rank,
                "predicted_communication_seconds": (
                    predict_candidate_seconds(candidate, world_size, models)
                    if feasible
                    else None
                ),
                "calls": [asdict(call) for call in candidate.calls],
            }
        )
    feasible_rows = [row for row in rows if row["feasible"]]
    if not feasible_rows:
        raise ValueError("no candidate fits the memory budget")
    feasible_rows.sort(key=lambda row: float(row["predicted_communication_seconds"]))
    infeasible_rows = [row for row in rows if not row["feasible"]]
    return feasible_rows + infeasible_rows


def default_strategy_candidates(message_bytes: int) -> tuple[StrategyCandidate, ...]:
    """Return five primitive signatures for one matched communication payload."""

    _positive_int(message_bytes, "message_bytes")
    return (
        StrategyCandidate(
            "data_parallel",
            (CollectiveCall("all_reduce", message_bytes),),
        ),
        StrategyCandidate(
            "zero_stage_2",
            (
                CollectiveCall("reduce_scatter", message_bytes),
                CollectiveCall("all_gather", message_bytes),
            ),
        ),
        StrategyCandidate(
            "tensor_parallel",
            (CollectiveCall("all_reduce", message_bytes, count=2),),
        ),
        StrategyCandidate(
            "context_parallel",
            (CollectiveCall("all_gather", message_bytes),),
        ),
        StrategyCandidate(
            "pipeline_parallel",
            (CollectiveCall("neighbor_exchange", message_bytes),),
        ),
        StrategyCandidate(
            "expert_parallel",
            (CollectiveCall("all_to_all", message_bytes, count=2),),
        ),
    )


def evaluate_planner(
    holdout_measurements: Sequence[Mapping[str, object]],
    models: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Evaluate communication-only top-1 choices against held-out primitives."""

    dimensions = sorted(
        {
            (_positive_int(record.get("world_size"), "world_size"),
             _positive_int(record.get("message_bytes"), "message_bytes"))
            for record in holdout_measurements
        }
    )
    scenarios: list[dict[str, object]] = []
    correct = 0
    for world_size, message_bytes in dimensions:
        candidates = default_strategy_candidates(message_bytes)
        predicted = rank_candidates(candidates, world_size, models)
        measured = sorted(
            (
                {
                    "name": candidate.name,
                    "measured_composed_seconds": measured_candidate_seconds(
                        candidate,
                        world_size,
                        holdout_measurements,
                    ),
                }
                for candidate in candidates
            ),
            key=lambda row: float(row["measured_composed_seconds"]),
        )
        predicted_choice = str(predicted[0]["name"])
        measured_choice = str(measured[0]["name"])
        is_correct = predicted_choice == measured_choice
        correct += int(is_correct)
        scenarios.append(
            {
                "world_size": world_size,
                "message_bytes": message_bytes,
                "predicted_choice": predicted_choice,
                "measured_choice": measured_choice,
                "correct": is_correct,
                "predicted_ranking": predicted,
                "measured_composed_ranking": measured,
            }
        )
    if not scenarios:
        raise ValueError("holdout_measurements must not be empty")
    return {
        "scope": (
            "communication-only primitive compositions; excludes model compute, "
            "overlap, optimizer behavior, and end-to-end training throughput"
        ),
        "top1_accuracy": correct / len(scenarios),
        "correct_scenarios": correct,
        "total_scenarios": len(scenarios),
        "scenarios": scenarios,
    }
