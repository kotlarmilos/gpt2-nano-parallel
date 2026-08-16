"""Deterministic partition planners used by the examples and tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Range:
    rank: int
    start: int
    stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start


def balanced_ranges(length: int, world_size: int) -> list[Range]:
    if length < 0:
        raise ValueError("length must not be negative")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    base, remainder = divmod(length, world_size)
    ranges: list[Range] = []
    start = 0
    for rank in range(world_size):
        size = base + (rank < remainder)
        ranges.append(Range(rank, start, start + size))
        start += size
    return ranges


def context_partitions(sequence_length: int, world_size: int) -> list[Range]:
    """Partition token positions without gaps or overlap."""

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if world_size > sequence_length:
        raise ValueError("world_size must not exceed sequence_length")
    return balanced_ranges(sequence_length, world_size)


def column_partition(
    matrix: Sequence[Sequence[T]], world_size: int
) -> list[list[list[T]]]:
    """Split a rectangular matrix along output columns."""

    if not matrix or not matrix[0]:
        raise ValueError("matrix must not be empty")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix must be rectangular")
    ranges = balanced_ranges(width, world_size)
    if any(part.size == 0 for part in ranges):
        raise ValueError("world_size must not exceed the column count")
    return [[list(row[part.start : part.stop]) for row in matrix] for part in ranges]


def reconstruct_columns(partitions: Sequence[Sequence[Sequence[T]]]) -> list[list[T]]:
    if not partitions or not partitions[0]:
        raise ValueError("partitions must not be empty")
    rows = len(partitions[0])
    if any(len(part) != rows for part in partitions):
        raise ValueError("partitions must have equal row counts")
    return [
        [value for part in partitions for value in part[row_index]]
        for row_index in range(rows)
    ]


def row_partition(
    matrix: Sequence[Sequence[T]], world_size: int
) -> list[list[list[T]]]:
    """Split a rectangular matrix along input rows."""

    if not matrix or not matrix[0]:
        raise ValueError("matrix must not be empty")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix must be rectangular")
    ranges = balanced_ranges(len(matrix), world_size)
    if any(part.size == 0 for part in ranges):
        raise ValueError("world_size must not exceed the row count")
    return [[list(row) for row in matrix[part.start : part.stop]] for part in ranges]


def reconstruct_rows(partitions: Sequence[Sequence[Sequence[T]]]) -> list[list[T]]:
    if not partitions:
        raise ValueError("partitions must not be empty")
    return [[value for value in row] for part in partitions for row in part]
