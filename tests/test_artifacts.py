import hashlib
import importlib.util
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

from gpt2_nano_parallel.gpu_collectives import (
    _derive_study_analysis,
    load_nccl_config,
)
from gpt2_nano_parallel.topology_planner import collective_seconds


def _load_generator() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "generate_artifacts.py"
    spec = importlib.util.spec_from_file_location("generate_artifacts", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load artifact generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArtifactTests(unittest.TestCase):
    @staticmethod
    def _measurement() -> dict[str, object]:
        return {
            "schema_version": 1,
            "generated_by": "gpt2_nano_parallel.cpu_demo.run_demo",
            "evidence": "measured",
            "scope": "local CPU processes using Gloo; not a GPU scaling result",
            "backend": "gloo",
            "device": "cpu",
            "torch_version": "2.13.0",
            "world_size": 2,
            "expected_sum": 3.0,
            "wall_seconds": 1.0,
            "ranks": [
                {"rank": 0, "sum": 3.0, "elapsed_seconds": 0.1},
                {"rank": 1, "sum": 3.0, "elapsed_seconds": 0.1},
            ],
        }

    @staticmethod
    def _allreduce_validation() -> dict[str, object]:
        return {
            "schema_version": 1,
            "generated_by": (
                "gpt2_nano_parallel.allreduce_validation.run_gloo_allreduce_sweep"
            ),
            "evidence": "measured",
            "scope": (
                "local CPU processes using Gloo on 127.0.0.1; "
                "validates the communication cost model only; "
                "not a GPU scaling result"
            ),
            "backend": "gloo",
            "device": "cpu",
            "torch_version": "2.13.0",
            "python_version": "3.11.0",
            "platform": "test",
            "wall_seconds": 1.0,
            "config": {
                "world_sizes": [2],
                "message_bytes": [4],
                "warmup_iterations": 1,
                "iterations": 2,
                "timeout_seconds": 30.0,
                "fit_min_message_bytes": 4,
            },
            "analytical_model": {
                "source_function": "gpt2_nano_parallel.communication.ring_collective",
                "operation": "all_reduce",
                "latency_steps": "2 * (world_size - 1)",
                "time_formula": "test",
            },
            "fitted_constants": {
                "bandwidth_bytes_per_second": 1000.0,
                "bandwidth_gib_per_second": 1000.0 / 2**30,
                "latency_seconds": 0.001,
                "latency_microseconds": 1000.0,
                "fit_min_message_bytes": 4,
                "fit_points": 1,
                "method": "least_squares",
                "sum_squared_error": 0.0,
            },
            "residuals": {
                "all_messages": {
                    "count": 1,
                    "mean_absolute_relative_error": 0.0,
                    "max_absolute_relative_error": 0.0,
                    "root_mean_squared_error_seconds": 0.0,
                    "mean_signed_error_seconds": 0.0,
                },
                "fit_range": {
                    "count": 1,
                    "mean_absolute_relative_error": 0.0,
                    "max_absolute_relative_error": 0.0,
                    "root_mean_squared_error_seconds": 0.0,
                    "mean_signed_error_seconds": 0.0,
                },
                "small_messages": {
                    "count": 0,
                    "mean_absolute_relative_error": None,
                    "max_absolute_relative_error": None,
                    "root_mean_squared_error_seconds": None,
                    "mean_signed_error_seconds": None,
                },
            },
            "measurements": [
                {
                    "world_size": 2,
                    "message_bytes": 4,
                    "iterations": 2,
                    "warmup_iterations": 1,
                    "measured_seconds": 0.01,
                    "median_seconds": 0.01,
                    "min_seconds": 0.01,
                    "max_seconds": 0.01,
                    "rank_mean_seconds": [0.01, 0.01],
                    "samples_seconds": [0.01, 0.01],
                    "ring_bytes_per_rank": 4.0,
                    "latency_steps": 2,
                    "model_seconds": 0.01,
                    "signed_error_seconds": 0.0,
                    "absolute_error_seconds": 0.0,
                    "relative_error": 0.0,
                }
            ],
        }

    @staticmethod
    def _write_nccl_study(directory: Path) -> dict[str, object]:
        root = Path(__file__).parents[1]
        config_path = root / "configs" / "nccl_publication.toml"
        config = load_nccl_config(config_path)
        config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
        source = {
            "git_revision": "abc123",
            "git_dirty": False,
            "git_state_captured": "study_start",
        }
        raw_payloads = []
        raw_entries = []
        common_environment = {
            "torch": "2.13.0",
            "cuda": "13.0",
            "nccl": "2.28.0",
            "python": "3.11.0",
            "platform": "test-platform",
            "hostname": "test-host",
            "gpu_names": ["Test GPU"] * max(config.world_sizes),
            "topology": "GPU0 X GPU1",
        }
        for world_size in config.world_sizes:
            environment = {
                "backend": "nccl",
                "device": "cuda",
                "world_size": world_size,
                **common_environment,
            }
            measurements = []
            for operation in config.operations:
                for message_bytes in config.message_bytes:
                    measured = collective_seconds(
                        operation,
                        message_bytes,
                        world_size,
                        8_000_000_000.0,
                        0.00001,
                    )
                    measurements.append(
                        {
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
                            "measured_seconds": measured,
                            "median_seconds": measured,
                            "min_seconds": measured,
                            "max_seconds": measured,
                            "samples_seconds": [measured] * config.iterations,
                            "rank_mean_seconds": [measured] * world_size,
                            "checksum": "verified",
                        }
                    )
            raw = {
                "schema_version": "nccl-collective-raw-v1",
                "status": "measured",
                "evidence": "measured",
                "config_sha256": config_hash,
                "config": asdict(config),
                "source": source,
                "environment": environment,
                "measurements": measurements,
            }
            raw_path = directory / f"raw-world-size-{world_size}.json"
            raw_path.write_text(json.dumps(raw), encoding="utf-8")
            raw_payloads.append(raw)
            raw_entries.append(
                {
                    "path": raw_path.name,
                    "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                    "world_size": world_size,
                    "environment": environment,
                }
            )
        return {
            "schema_version": "nccl-topology-study-v1",
            "status": "measured",
            "evidence": "measured",
            "scope": (
                "single-node NCCL collective calibration with held-out message sizes "
                "and a communication-only strategy planner"
            ),
            "config_path": "configs/nccl_publication.toml",
            "config_sha256": config_hash,
            "source": source,
            "measurement_source_revision": "abc123",
            "splits": {
                "calibration_message_bytes": list(
                    config.calibration_message_bytes
                ),
                "validation_message_bytes": list(config.validation_message_bytes),
                "overlap": False,
            },
            "raw_artifacts": raw_entries,
            **_derive_study_analysis(config, raw_payloads),
        }

    def test_validates_cpu_measurement_provenance_and_content(self) -> None:
        generator = _load_generator()
        raw = json.dumps(self._measurement()).encode()
        self.assertEqual(generator._validate_cpu_measurement(raw)["world_size"], 2)

    def test_rejects_invalid_cpu_measurement(self) -> None:
        generator = _load_generator()
        measurement = self._measurement()
        measurement["generated_by"] = "unknown"
        with self.assertRaises(ValueError):
            generator._validate_cpu_measurement(json.dumps(measurement).encode())

        measurement = self._measurement()
        measurement["ranks"][1]["sum"] = 2.0
        with self.assertRaises(ValueError):
            generator._validate_cpu_measurement(json.dumps(measurement).encode())

    def test_validates_allreduce_model_validation_artifact(self) -> None:
        generator = _load_generator()
        raw = json.dumps(self._allreduce_validation()).encode()
        validated = generator._validate_allreduce_validation(raw)
        self.assertEqual(validated["backend"], "gloo")

    def test_rejects_invalid_allreduce_model_validation_artifact(self) -> None:
        generator = _load_generator()
        measurement = self._allreduce_validation()
        measurement["measurements"][0]["relative_error"] = -1.0
        with self.assertRaises(ValueError):
            generator._validate_allreduce_validation(json.dumps(measurement).encode())

    def test_validates_nccl_study(self) -> None:
        generator = _load_generator()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study = self._write_nccl_study(root)
            validated = generator._validate_nccl_study(
                json.dumps(study).encode(),
                root,
            )
            self.assertEqual(validated["planner"]["top1_accuracy"], 1.0)

    def test_rejects_nccl_calibration_leakage(self) -> None:
        generator = _load_generator()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study = self._write_nccl_study(root)
            study["splits"]["validation_message_bytes"] = [4096]
            with self.assertRaises(ValueError):
                generator._validate_nccl_study(json.dumps(study).encode(), root)

    def test_rejects_nccl_results_not_derived_from_raw_evidence(self) -> None:
        generator = _load_generator()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study = self._write_nccl_study(root)
            study["held_out"]["comparisons"][0]["measured_seconds"] = 999.0
            with self.assertRaisesRegex(ValueError, "held_out"):
                generator._validate_nccl_study(json.dumps(study).encode(), root)


if __name__ == "__main__":
    unittest.main()
