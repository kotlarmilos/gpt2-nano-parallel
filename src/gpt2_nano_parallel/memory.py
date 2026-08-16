"""Analytical model-state memory accounting for data parallelism and ZeRO."""

from __future__ import annotations

from dataclasses import asdict, dataclass


def _positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class MemoryBreakdown:
    """Per-rank steady-state model-state memory, excluding activations."""

    stage: int
    world_size: int
    parameters: float
    gradients: float
    optimizer: float
    master_parameters: float

    @property
    def total(self) -> float:
        return (
            self.parameters
            + self.gradients
            + self.optimizer
            + self.master_parameters
        )

    def as_dict(self) -> dict[str, int | float | str]:
        result: dict[str, int | float | str] = asdict(self)
        result["total"] = self.total
        result["evidence"] = "analytical"
        return result


def zero_memory(
    parameters: int,
    world_size: int,
    stage: int,
    *,
    parameter_bytes: int = 2,
    gradient_bytes: int = 2,
    optimizer_bytes: int = 8,
    master_parameter_bytes: int = 4,
) -> MemoryBreakdown:
    """Calculate per-rank bytes for ordinary DP or ZeRO stages 1 through 3.

    Stage 0 is replicated data parallelism. The default assumes BF16 model
    parameters and gradients, two FP32 Adam moments, and FP32 master weights.
    Divisibility is not required because the result is an idealized balance.
    """

    _positive("parameters", parameters)
    _positive("world_size", world_size)
    if stage not in {0, 1, 2, 3}:
        raise ValueError("stage must be one of 0, 1, 2, or 3")
    for name, value in (
        ("parameter_bytes", parameter_bytes),
        ("gradient_bytes", gradient_bytes),
        ("optimizer_bytes", optimizer_bytes),
        ("master_parameter_bytes", master_parameter_bytes),
    ):
        if value < 0:
            raise ValueError(f"{name} must not be negative")

    parameter_divisor = world_size if stage >= 3 else 1
    gradient_divisor = world_size if stage >= 2 else 1
    state_divisor = world_size if stage >= 1 else 1
    return MemoryBreakdown(
        stage=stage,
        world_size=world_size,
        parameters=parameters * parameter_bytes / parameter_divisor,
        gradients=parameters * gradient_bytes / gradient_divisor,
        optimizer=parameters * optimizer_bytes / state_divisor,
        master_parameters=parameters * master_parameter_bytes / state_divisor,
    )
