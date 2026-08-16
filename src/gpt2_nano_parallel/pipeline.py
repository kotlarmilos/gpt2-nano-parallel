"""Pipeline-parallel GPipe schedule simulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PipelineEvent:
    slot: int
    stage: int
    microbatch: int
    phase: str


def gpipe_schedule(stages: int, microbatches: int) -> list[PipelineEvent]:
    """Build deterministic forward-then-backward GPipe events."""

    if stages <= 0 or microbatches <= 0:
        raise ValueError("stages and microbatches must be positive")
    events: list[PipelineEvent] = []
    forward_slots = stages + microbatches - 1
    for slot in range(forward_slots):
        for stage in range(stages):
            microbatch = slot - stage
            if 0 <= microbatch < microbatches:
                events.append(PipelineEvent(slot, stage, microbatch, "F"))
    backward_start = forward_slots
    for offset in range(forward_slots):
        slot = backward_start + offset
        for stage in range(stages):
            local_index = offset - (stages - 1 - stage)
            if 0 <= local_index < microbatches:
                microbatch = microbatches - 1 - local_index
                events.append(PipelineEvent(slot, stage, microbatch, "B"))
    return events


def bubble_fraction(stages: int, microbatches: int) -> float:
    """Return the standard GPipe bubble fraction per directional sweep."""

    if stages <= 0 or microbatches <= 0:
        raise ValueError("stages and microbatches must be positive")
    return (stages - 1) / (microbatches + stages - 1)


def schedule_report(stages: int, microbatches: int) -> dict[str, object]:
    events = gpipe_schedule(stages, microbatches)
    return {
        "evidence": "simulated",
        "scheduler": "GPipe forward-then-backward",
        "stages": stages,
        "microbatches": microbatches,
        "bubble_fraction": bubble_fraction(stages, microbatches),
        "slots": max(event.slot for event in events) + 1,
        "events": [asdict(event) for event in events],
    }
