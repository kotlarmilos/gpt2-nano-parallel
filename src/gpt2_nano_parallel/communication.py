"""Analytical communication-volume calculators."""

from __future__ import annotations

from dataclasses import dataclass


def _world_size(world_size: int) -> None:
    if world_size <= 0:
        raise ValueError("world_size must be positive")


@dataclass(frozen=True)
class Communication:
    operation: str
    bytes_per_rank: float
    aggregate_bytes: float
    evidence: str = "analytical"


@dataclass(frozen=True)
class PerRankCommunication:
    """Communication volumes when ranks transmit different byte counts."""

    operation: str
    bytes_per_rank: tuple[int, ...]
    aggregate_bytes: int
    evidence: str = "analytical"


def ring_collective(payload_bytes: int, world_size: int, operation: str) -> Communication:
    """Return transmitted bytes for ring all-reduce, all-gather, or reduce-scatter."""

    _world_size(world_size)
    if payload_bytes < 0:
        raise ValueError("payload_bytes must not be negative")
    factors = {
        "all_reduce": 2 * (world_size - 1) / world_size,
        "all_gather": (world_size - 1) / world_size,
        "reduce_scatter": (world_size - 1) / world_size,
        "all_to_all": (world_size - 1) / world_size,
    }
    if operation not in factors:
        raise ValueError(f"unsupported operation: {operation}")
    per_rank = payload_bytes * factors[operation]
    return Communication(operation, per_rank, per_rank * world_size)


def data_parallel_step(
    parameters: int, world_size: int, *, gradient_bytes: int = 2
) -> Communication:
    """Model a ring gradient all-reduce for one data-parallel step."""

    return ring_collective(parameters * gradient_bytes, world_size, "all_reduce")


def zero_step(
    parameters: int,
    world_size: int,
    stage: int,
    *,
    parameter_bytes: int = 2,
    gradient_bytes: int = 2,
) -> Communication:
    """Approximate steady-state ZeRO communication per rank and training step."""

    if stage not in {1, 2, 3}:
        raise ValueError("ZeRO communication stage must be 1, 2, or 3")
    _world_size(world_size)
    if parameters <= 0:
        raise ValueError("parameters must be positive")
    if stage == 1:
        return data_parallel_step(parameters, world_size, gradient_bytes=gradient_bytes)
    factor = (world_size - 1) / world_size
    if stage == 2:
        per_rank = parameters * (gradient_bytes + parameter_bytes) * factor
    else:
        per_rank = parameters * (gradient_bytes + 2 * parameter_bytes) * factor
    return Communication(f"zero_{stage}_step", per_rank, per_rank * world_size)


def tensor_linear(
    batch_tokens: int,
    in_features: int,
    out_features: int,
    world_size: int,
    mode: str,
    *,
    element_bytes: int = 2,
) -> Communication:
    """Communication for one Megatron-style row or column parallel linear layer."""

    _world_size(world_size)
    if min(batch_tokens, in_features, out_features, element_bytes) <= 0:
        raise ValueError("tensor dimensions and element_bytes must be positive")
    if mode == "row":
        payload = batch_tokens * out_features * element_bytes
        return ring_collective(payload, world_size, "all_reduce")
    if mode == "column":
        payload = batch_tokens * in_features * element_bytes
        return ring_collective(payload, world_size, "all_reduce")
    raise ValueError("mode must be row or column")


def ring_attention(
    sequence_length: int,
    hidden_size: int,
    world_size: int,
    *,
    element_bytes: int = 2,
) -> PerRankCommunication:
    """Count K and V bytes sent by each rank over a complete ring rotation.

    Rank r sends its current block clockwise for world_size - 1 steps. With
    balanced contiguous token ranges, that rank sends every block except the
    one initially held by rank r + 1.
    """

    _world_size(world_size)
    if sequence_length <= 0 or hidden_size <= 0 or element_bytes <= 0:
        raise ValueError("dimensions and element_bytes must be positive")
    if world_size > sequence_length:
        raise ValueError("world_size must not exceed sequence_length")
    base, remainder = divmod(sequence_length, world_size)
    tokens_by_rank = tuple(
        base + (rank < remainder) for rank in range(world_size)
    )
    bytes_by_block = tuple(
        2 * tokens * hidden_size * element_bytes for tokens in tokens_by_rank
    )
    total_block_bytes = sum(bytes_by_block)
    bytes_per_rank = tuple(
        total_block_bytes - bytes_by_block[(rank + 1) % world_size]
        for rank in range(world_size)
    )
    return PerRankCommunication(
        "ring_attention_kv",
        bytes_per_rank,
        sum(bytes_per_rank),
    )
