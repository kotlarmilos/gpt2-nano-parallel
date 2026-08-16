"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .communication import (
    data_parallel_step,
    ring_attention,
    tensor_linear,
    zero_step,
)
from .config import load_config
from .cpu_demo import run_demo
from .expert import routing_report
from .memory import zero_memory
from .partitions import balanced_ranges, context_partitions
from .pipeline import schedule_report


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2))


def _tensor_report(
    batch_tokens: int, in_features: int, out_features: int, world_size: int
) -> dict[str, object]:
    if world_size > out_features:
        raise ValueError("world_size must not exceed the MLP expansion width")
    return {
        "evidence": "analytical",
        "column_projection": {
            "in_features": in_features,
            "out_features": out_features,
        },
        "row_projection": {
            "in_features": out_features,
            "out_features": in_features,
        },
        "column_output_ranges": [
            asdict(part) for part in balanced_ranges(out_features, world_size)
        ],
        "row_input_ranges": [
            asdict(part) for part in balanced_ranges(out_features, world_size)
        ],
        "column_backward_communication": asdict(
            tensor_linear(
                batch_tokens,
                in_features,
                out_features,
                world_size,
                "column",
            )
        ),
        "row_forward_communication": asdict(
            tensor_linear(
                batch_tokens,
                out_features,
                in_features,
                world_size,
                "row",
            )
        ),
    }


def _report(path: str) -> dict[str, object]:
    config = load_config(path)
    model = config["model"]
    parallel = config["parallel"]
    parameters = int(model["parameters"])
    world_size = int(parallel["world_size"])
    sequence = int(model["context_length"])
    hidden = int(model["hidden_size"])
    return {
        "configuration": Path(path).name,
        "memory": [
            zero_memory(parameters, world_size, stage).as_dict()
            for stage in range(4)
        ],
        "data_parallel_communication": asdict(
            data_parallel_step(parameters, world_size)
        ),
        "zero_communication": [
            asdict(zero_step(parameters, world_size, stage))
            for stage in range(1, 4)
        ],
        "context_partitions": [
            asdict(part) for part in context_partitions(sequence, world_size)
        ],
        "ring_attention": asdict(ring_attention(sequence, hidden, world_size)),
        "tensor_parallel": _tensor_report(
            int(parallel["tokens"]),
            hidden,
            int(model["mlp_size"]),
            world_size,
        ),
        "pipeline": schedule_report(
            int(parallel["pipeline_stages"]),
            int(parallel["microbatches"]),
        ),
        "experts": routing_report(
            int(parallel["tokens"]),
            int(parallel["experts"]),
            world_size,
            hidden_size=hidden,
            top_k=int(parallel["top_k"]),
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpt2-parallel")
    subparsers = parser.add_subparsers(dest="command", required=True)
    report = subparsers.add_parser("report", help="run configured calculators")
    report.add_argument("config")
    memory = subparsers.add_parser("memory", help="calculate ZeRO memory")
    memory.add_argument("--parameters", type=int, default=44_000_000)
    memory.add_argument("--world-size", type=int, required=True)
    memory.add_argument("--stage", type=int, choices=range(4), required=True)
    pipeline = subparsers.add_parser("pipeline", help="simulate GPipe")
    pipeline.add_argument("--stages", type=int, required=True)
    pipeline.add_argument("--microbatches", type=int, required=True)
    experts = subparsers.add_parser("experts", help="simulate expert routing")
    experts.add_argument("--tokens", type=int, required=True)
    experts.add_argument("--experts", type=int, required=True)
    experts.add_argument("--world-size", type=int, required=True)
    experts.add_argument("--hidden-size", type=int, default=512)
    experts.add_argument("--top-k", type=int, default=1)
    tensor = subparsers.add_parser("tensor", help="plan a tensor-parallel MLP")
    tensor.add_argument("--batch-tokens", type=int, required=True)
    tensor.add_argument("--in-features", type=int, default=512)
    tensor.add_argument("--out-features", type=int, default=2048)
    tensor.add_argument("--world-size", type=int, required=True)
    context = subparsers.add_parser("context", help="plan context partitions")
    context.add_argument("--sequence-length", type=int, default=1024)
    context.add_argument("--hidden-size", type=int, default=512)
    context.add_argument("--world-size", type=int, required=True)
    demo = subparsers.add_parser("cpu-demo", help="measure a local Gloo all-reduce")
    demo.add_argument("--world-size", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "report":
        _print(_report(args.config))
    elif args.command == "memory":
        _print(zero_memory(args.parameters, args.world_size, args.stage).as_dict())
    elif args.command == "pipeline":
        _print(schedule_report(args.stages, args.microbatches))
    elif args.command == "experts":
        _print(
            routing_report(
                args.tokens,
                args.experts,
                args.world_size,
                hidden_size=args.hidden_size,
                top_k=args.top_k,
            )
        )
    elif args.command == "tensor":
        _print(
            _tensor_report(
                args.batch_tokens,
                args.in_features,
                args.out_features,
                args.world_size,
            )
        )
    elif args.command == "context":
        _print(
            {
                "evidence": "analytical",
                "partitions": [
                    asdict(part)
                    for part in context_partitions(
                        args.sequence_length, args.world_size
                    )
                ],
                "ring_attention": asdict(
                    ring_attention(
                        args.sequence_length,
                        args.hidden_size,
                        args.world_size,
                    )
                ),
            }
        )
    elif args.command == "cpu-demo":
        _print(run_demo(args.world_size))


if __name__ == "__main__":
    main()
