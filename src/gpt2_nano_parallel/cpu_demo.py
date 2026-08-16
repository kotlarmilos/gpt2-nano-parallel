"""Optional measured torch.distributed CPU all-reduce demonstration."""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from queue import Empty
from typing import Any


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _worker(rank: int, world_size: int, port: int, queue: Any) -> None:
    import torch
    import torch.distributed as dist

    try:
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world_size,
        )
        value = torch.tensor([float(rank + 1)])
        dist.barrier()
        started = time.perf_counter()
        dist.all_reduce(value)
        elapsed = time.perf_counter() - started
        queue.put(
            {
                "kind": "result",
                "rank": rank,
                "sum": value.item(),
                "elapsed_seconds": elapsed,
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
        if dist.is_initialized():
            dist.destroy_process_group()


def run_demo(
    world_size: int = 2, *, timeout_seconds: float = 60.0
) -> dict[str, object]:
    """Run a real local Gloo collective and return measured timing."""

    if world_size <= 1:
        raise ValueError("world_size must be greater than one")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    try:
        import torch
        import torch.multiprocessing as mp
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required; install the cpu-demo optional dependency"
        ) from error
    context = mp.get_context("spawn")
    queue = context.Queue()
    port = _free_port()
    processes = [
        context.Process(target=_worker, args=(rank, world_size, port, queue))
        for rank in range(world_size)
    ]
    started_processes: list[Any] = []
    results: dict[int, dict[str, object]] = {}
    wall_started = time.perf_counter()
    deadline = wall_started + timeout_seconds
    try:
        for process in processes:
            process.start()
            started_processes.append(process)

        while len(results) < world_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(
                    f"Gloo demo timed out after {timeout_seconds:.1f} seconds"
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
                if all(
                    process.exitcode == 0 for process in started_processes
                ) and len(started_processes) == world_size:
                    raise RuntimeError(
                        "Gloo workers exited without returning every result"
                    )
                continue

            if message.get("kind") == "error":
                raise RuntimeError(
                    f"Gloo worker {message.get('rank')} failed: "
                    f"{message.get('error')}"
                )
            if message.get("kind") != "result":
                raise RuntimeError("Gloo worker returned an invalid message")
            rank = message.get("rank")
            if not isinstance(rank, int) or rank not in range(world_size):
                raise RuntimeError(f"Gloo worker returned invalid rank {rank!r}")
            if rank in results:
                raise RuntimeError(f"Gloo worker returned duplicate rank {rank}")
            results[rank] = {
                "rank": rank,
                "sum": message["sum"],
                "elapsed_seconds": message["elapsed_seconds"],
            }

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

        return {
            "schema_version": 1,
            "generated_by": "gpt2_nano_parallel.cpu_demo.run_demo",
            "evidence": "measured",
            "scope": "local CPU processes using Gloo; not a GPU scaling result",
            "backend": "gloo",
            "device": "cpu",
            "torch_version": torch.__version__,
            "world_size": world_size,
            "expected_sum": world_size * (world_size + 1) / 2,
            "wall_seconds": time.perf_counter() - wall_started,
            "ranks": [results[rank] for rank in range(world_size)],
        }
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


def write_demo(path: Path, world_size: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.new")
    try:
        temporary.write_text(json.dumps(run_demo(world_size), indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
