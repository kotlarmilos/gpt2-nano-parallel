import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from gpt2_nano_parallel.gpu_collectives import _load_raw, load_nccl_config


class NcclConfigTests(unittest.TestCase):
    def test_loads_publication_config(self) -> None:
        root = Path(__file__).parents[1]
        config = load_nccl_config(root / "configs" / "nccl_publication.toml")
        self.assertEqual(config.world_sizes, (2, 4))
        self.assertEqual(len(config.operations), 5)
        self.assertFalse(
            set(config.calibration_message_bytes)
            & set(config.validation_message_bytes)
        )

    def test_rejects_overlapping_splits(self) -> None:
        content = """
[nccl_study]
world_sizes = [2]
operations = ["all_reduce"]
calibration_message_bytes = [4096]
validation_message_bytes = [4096]
warmup_iterations = 0
iterations = 1
timeout_seconds = 30.0
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(content)
            with self.assertRaises(ValueError):
                load_nccl_config(path)

    def test_rejects_message_size_not_divisible_by_world_size(self) -> None:
        content = """
[nccl_study]
world_sizes = [4]
operations = ["all_reduce"]
calibration_message_bytes = [12]
validation_message_bytes = [16]
warmup_iterations = 0
iterations = 1
timeout_seconds = 30.0
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(content)
            with self.assertRaises(ValueError):
                load_nccl_config(path)

    def test_rejects_raw_split_relabeling(self) -> None:
        content = """
[nccl_study]
world_sizes = [2]
operations = ["all_reduce"]
calibration_message_bytes = [4096]
validation_message_bytes = [16384]
warmup_iterations = 0
iterations = 1
timeout_seconds = 30.0
"""
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.toml"
            raw_path = Path(directory) / "raw.json"
            config_path.write_text(content)
            config = load_nccl_config(config_path)
            config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
            raw_path.write_text(
                json.dumps(
                    {
                        "schema_version": "nccl-collective-raw-v1",
                        "status": "measured",
                        "evidence": "measured",
                        "config_sha256": config_hash,
                        "config": asdict(config),
                        "source": {
                            "git_revision": "abc123",
                            "git_dirty": False,
                            "git_state_captured": "study_start",
                        },
                        "environment": {
                            "backend": "nccl",
                            "device": "cuda",
                            "world_size": 2,
                            "torch": "2.13.0",
                            "cuda": "13.0",
                            "nccl": "2.28.0",
                            "python": "3.11.0",
                            "platform": "test-platform",
                            "hostname": "test-host",
                            "gpu_names": ["Test GPU", "Test GPU"],
                            "topology": "GPU0 X GPU1",
                        },
                        "measurements": [
                            {
                                "operation": "all_reduce",
                                "world_size": 2,
                                "message_bytes": 4096,
                                "split": "validation",
                                "measured_seconds": 0.1,
                                "median_seconds": 0.1,
                                "min_seconds": 0.1,
                                "max_seconds": 0.1,
                                "samples_seconds": [0.1],
                                "rank_mean_seconds": [0.1, 0.1],
                                "warmup_iterations": 0,
                                "iterations": 1,
                                "checksum": "verified",
                            },
                            {
                                "operation": "all_reduce",
                                "world_size": 2,
                                "message_bytes": 16384,
                                "split": "calibration",
                                "measured_seconds": 0.2,
                                "median_seconds": 0.2,
                                "min_seconds": 0.2,
                                "max_seconds": 0.2,
                                "samples_seconds": [0.2],
                                "rank_mean_seconds": [0.2, 0.2],
                                "warmup_iterations": 0,
                                "iterations": 1,
                                "checksum": "verified",
                            },
                        ],
                    }
                )
            )
            with self.assertRaises(ValueError):
                _load_raw(raw_path, config, config_hash)


if __name__ == "__main__":
    unittest.main()
