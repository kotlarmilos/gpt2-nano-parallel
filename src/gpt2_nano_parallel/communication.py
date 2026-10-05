"""Communication-volume model used by the measured NCCL study."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Communication:
    operation: str
    bytes_per_rank: float
    aggregate_bytes: float


def ring_collective(
    payload_bytes: int,
    world_size: int,
    operation: str,
) -> Communication:
    """Return modeled ring bytes for the four collective primitives."""

    if payload_bytes <= 0:
        raise ValueError("payload_bytes must be positive")
    if world_size <= 1:
        raise ValueError("world_size must be greater than one")
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
