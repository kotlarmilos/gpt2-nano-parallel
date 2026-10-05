import copy
import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import ModuleType

from gpt2_nano_parallel.gpu_collectives import load_nccl_config
from gpt2_nano_parallel.topology_planner import (
    collective_bytes_per_rank,
    collective_latency_steps,
    collective_seconds,
    compare_holdout,
    constant_choice_baseline,
    default_communication_compositions,
    evaluate_planner,
    fit_collective_models,
    fit_shared_collective_model,
    rank_compositions,
)

ROOT = Path(__file__).parents[1]


def _load_verifier() -> ModuleType:
    path = ROOT / "scripts" / "verify_release.py"
    spec = importlib.util.spec_from_file_location("verify_release", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load release verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseTests(unittest.TestCase):
    def test_collective_math_covers_five_primitives(self) -> None:
        cases = (
            ("all_reduce", 150.0, 6, 1.56),
            ("reduce_scatter", 75.0, 3, 0.78),
            ("all_gather", 75.0, 3, 0.78),
            ("all_to_all", 75.0, 3, 0.78),
            ("neighbor_exchange", 100.0, 1, 1.01),
        )
        for operation, expected_bytes, expected_steps, expected_seconds in cases:
            with self.subTest(operation=operation):
                self.assertEqual(
                    collective_bytes_per_rank(operation, 100, 4),
                    expected_bytes,
                )
                self.assertEqual(
                    collective_latency_steps(operation, 4),
                    expected_steps,
                )
                self.assertAlmostEqual(
                    collective_seconds(operation, 100, 4, 100.0, 0.01),
                    expected_seconds,
                )

    def test_publication_config_has_disjoint_complete_splits(self) -> None:
        config = load_nccl_config(ROOT / "configs" / "nccl_publication.toml")
        self.assertEqual(config.world_sizes, (2, 4))
        self.assertEqual(len(config.operations), 5)
        self.assertTrue(
            set(config.calibration_message_bytes).isdisjoint(
                config.validation_message_bytes
            )
        )
        self.assertEqual(len(config.message_bytes), 8)

    def test_fits_use_only_calibration_rows(self) -> None:
        calibration_sizes = (4_096, 1_048_576)
        holdout_sizes = (65_536,)
        constants = {
            "all_reduce": (8_000_000_000.0, 0.00001),
            "reduce_scatter": (7_000_000_000.0, 0.00002),
            "all_gather": (3_000_000_000.0, 0.00004),
            "all_to_all": (4_000_000_000.0, 0.00003),
            "neighbor_exchange": (5_000_000_000.0, 0.00005),
        }

        def rows(message_sizes: tuple[int, ...]) -> list[dict[str, object]]:
            return [
                {
                    "operation": operation,
                    "world_size": 4,
                    "message_bytes": message_bytes,
                    "measured_seconds": collective_seconds(
                        operation,
                        message_bytes,
                        4,
                        bandwidth,
                        latency,
                    ),
                }
                for operation, (bandwidth, latency) in constants.items()
                for message_bytes in message_sizes
            ]

        calibration = rows(calibration_sizes)
        holdout = rows(holdout_sizes)
        models = fit_collective_models(calibration)
        shared = fit_shared_collective_model(calibration)
        comparisons = compare_holdout(holdout, models, shared)

        self.assertEqual(len(models), 5)
        self.assertEqual(len(comparisons), 5)
        for row in comparisons:
            self.assertAlmostEqual(row["collective_model_relative_error"], 0.0)
        self.assertTrue(
            any(row["shared_model_relative_error"] > 0.01 for row in comparisons)
        )

    def test_planner_and_fixed_baseline_match_evidence(self) -> None:
        study = json.loads(
            (ROOT / "artifacts" / "nccl-publication" / "results.json").read_text()
        )
        holdout = []
        for world_size in (2, 4):
            raw = json.loads(
                (
                    ROOT
                    / "artifacts"
                    / "nccl-publication"
                    / f"raw-world-size-{world_size}.json"
                ).read_text()
            )
            holdout.extend(
                row for row in raw["measurements"] if row["split"] == "validation"
            )

        planner = evaluate_planner(holdout, study["collective_models"])
        baseline = constant_choice_baseline(planner["scenarios"])
        compositions = {
            item.name: [(call.operation, call.count) for call in item.calls]
            for item in default_communication_compositions(16_384)
        }

        self.assertEqual(
            compositions["zero_stage_2"],
            [("reduce_scatter", 1), ("all_gather", 1)],
        )
        self.assertEqual(compositions["expert_parallel"], [("all_to_all", 2)])
        self.assertEqual(planner["correct_scenarios"], 7)
        self.assertEqual(
            baseline,
            {
                "choice": "context_parallel",
                "correct_scenarios": 8,
                "total_scenarios": 8,
                "top1_accuracy": 1.0,
            },
        )

    def test_verifier_checks_hashes_provenance_and_derived_values(self) -> None:
        verifier = _load_verifier()
        study = verifier.validate_result()
        manifest = verifier.validate_manifest(study)

        self.assertEqual(
            manifest["base_release_revision"],
            "83d4fbe9b991739b6f13764dd64d2ffac4823a6e",
        )
        self.assertEqual(study["planner"]["correct_scenarios"], 7)
        self.assertEqual(len(study["held_out"]["comparisons"]), 40)

    def test_verifier_rejects_tampering(self) -> None:
        verifier = _load_verifier()
        study = json.loads(
            (ROOT / "artifacts" / "nccl-publication" / "results.json").read_text()
        )
        mutations = (
            (("source", "git_revision"), "wrong-revision"),
            (("splits", "validation_message_bytes"), [4_096]),
            (("held_out", "comparisons", 0, "measured_seconds"), 999.0),
        )
        for path, value in mutations:
            with self.subTest(path=path):
                changed = copy.deepcopy(study)
                target = changed
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaises(ValueError):
                    verifier.validate_result(changed)

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            collective_seconds("all_reduce", 100, 4, 0.0, 0.01)
        with self.assertRaises(ValueError):
            collective_bytes_per_rank("all_reduce", 100, 1)
        with self.assertRaises(ValueError):
            rank_compositions(
                (default_communication_compositions(4_096)[0],),
                4,
                {},
            )

    def test_cpu_release_smoke(self) -> None:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT / "src")
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "verify_release.py")],
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        output = json.loads(completed.stdout)
        self.assertEqual(output["status"], "verified")
        self.assertEqual(output["planner"], "7/8")
        self.assertEqual(output["constant_baseline"], "8/8 context_parallel")


if __name__ == "__main__":
    unittest.main()
