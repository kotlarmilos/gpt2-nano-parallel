#!/usr/bin/env python3
"""Verify the measured NCCL release artifacts and their derived result."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpt2_nano_parallel.gpu_collectives import (
    _derive_study_analysis,
    _load_raw,
    _validate_shared_environment,
    load_nccl_config,
)
from gpt2_nano_parallel.topology_planner import constant_choice_baseline

CONFIG_PATH = "configs/nccl_publication.toml"
RESULT_PATH = "artifacts/nccl-publication/results.json"
MANIFEST_PATH = "artifacts/manifest.json"
EXPECTED_SCOPE = (
    "single-node NCCL collective timing prediction with held-out message "
    "sizes and a communication-composition planner"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _values_match(actual: object, expected: object) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(
            float(actual),
            float(expected),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _values_match(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _values_match(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def validate_result(
    study: Mapping[str, object] | None = None,
    root: Path = ROOT,
) -> dict[str, object]:
    if study is None:
        loaded = json.loads((root / RESULT_PATH).read_text())
        if not isinstance(loaded, dict):
            raise ValueError("results artifact must be a JSON object")
        study = loaded
    result = dict(study)

    expected_metadata = {
        "schema_version": "nccl-topology-study-v1",
        "status": "measured",
        "evidence": "measured",
        "scope": EXPECTED_SCOPE,
        "config_path": CONFIG_PATH,
    }
    for key, expected in expected_metadata.items():
        if result.get(key) != expected:
            raise ValueError(f"results artifact has invalid {key}")

    config_path = root / CONFIG_PATH
    config_hash = _sha256(config_path)
    if result.get("config_sha256") != config_hash:
        raise ValueError("results artifact has an invalid config hash")
    config = load_nccl_config(config_path)

    source = result.get("source")
    if (
        not isinstance(source, dict)
        or source.get("git_dirty") is not False
        or source.get("git_state_captured") != "study_start"
        or not isinstance(source.get("git_revision"), str)
        or not source["git_revision"]
    ):
        raise ValueError("results artifact has invalid source provenance")

    raw_metadata = result.get("raw_artifacts")
    if not isinstance(raw_metadata, list) or len(raw_metadata) != len(
        config.world_sizes
    ):
        raise ValueError("results artifact has incomplete raw provenance")

    raw_directory = root / "artifacts" / "nccl-publication"
    raw_artifacts: list[dict[str, object]] = []
    seen_world_sizes: set[int] = set()
    revisions: set[str] = set()
    for metadata in raw_metadata:
        if not isinstance(metadata, dict):
            raise ValueError("results artifact has invalid raw provenance")
        name = metadata.get("path")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("raw artifact path must be a file name")
        path = raw_directory / name
        if metadata.get("sha256") != _sha256(path):
            raise ValueError("raw artifact hash does not match")
        raw = _load_raw(path, config, config_hash)
        environment = raw["environment"]
        world_size = environment["world_size"]
        if (
            not isinstance(world_size, int)
            or world_size in seen_world_sizes
            or metadata.get("world_size") != world_size
            or metadata.get("environment") != environment
            or name != f"raw-world-size-{world_size}.json"
        ):
            raise ValueError("raw artifact provenance is inconsistent")
        seen_world_sizes.add(world_size)
        revisions.add(str(raw["source"]["git_revision"]))
        raw_artifacts.append(raw)

    if seen_world_sizes != set(config.world_sizes):
        raise ValueError("raw artifacts do not cover configured world sizes")
    _validate_shared_environment(raw_artifacts)
    if (
        len(revisions) != 1
        or result.get("measurement_source_revision") != next(iter(revisions))
        or source.get("git_revision") != next(iter(revisions))
    ):
        raise ValueError("measurement source revisions are inconsistent")

    expected_splits = {
        "calibration_message_bytes": list(config.calibration_message_bytes),
        "validation_message_bytes": list(config.validation_message_bytes),
        "overlap": False,
    }
    if result.get("splits") != expected_splits:
        raise ValueError("results artifact has invalid calibration splits")

    derived = _derive_study_analysis(config, raw_artifacts)
    for key, expected in derived.items():
        if not _values_match(result.get(key), expected):
            raise ValueError(f"results artifact has invalid derived {key}")

    planner = result["planner"]
    baseline = constant_choice_baseline(planner["scenarios"])
    if (
        planner["correct_scenarios"] != 7
        or planner["total_scenarios"] != 8
        or baseline
        != {
            "choice": "context_parallel",
            "correct_scenarios": 8,
            "total_scenarios": 8,
            "top1_accuracy": 1.0,
        }
    ):
        raise ValueError("planner baseline comparison does not match the release")
    return result


def validate_manifest(
    study: Mapping[str, object],
    root: Path = ROOT,
) -> dict[str, object]:
    loaded = json.loads((root / MANIFEST_PATH).read_text())
    if not isinstance(loaded, dict):
        raise ValueError("manifest must be a JSON object")
    manifest = loaded
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest has an invalid schema version")
    if (
        manifest.get("base_release_revision")
        != "83d4fbe9b991739b6f13764dd64d2ffac4823a6e"
    ):
        raise ValueError("manifest has an invalid base release revision")
    files = manifest.get("files")
    expected_files = {
        CONFIG_PATH,
        "artifacts/nccl-publication/raw-world-size-2.json",
        "artifacts/nccl-publication/raw-world-size-4.json",
        RESULT_PATH,
    }
    if not isinstance(files, dict) or set(files) != expected_files:
        raise ValueError("manifest file set is invalid")
    for relative_path, metadata in files.items():
        if (
            not isinstance(metadata, dict)
            or metadata.get("sha256") != _sha256(root / relative_path)
        ):
            raise ValueError(f"manifest hash does not match {relative_path}")

    baseline = constant_choice_baseline(study["planner"]["scenarios"])
    claim = manifest.get("claim")
    expected_claim = {
        "measurement_rows": 80,
        "gpu_model": "NVIDIA GeForce RTX 4090",
        "gpu_count": 4,
        "collective_primitives": 5,
        "collective_specific_mean_absolute_relative_error": study["held_out"][
            "collective_specific_error"
        ]["mean_absolute_relative_error"],
        "shared_model_mean_absolute_relative_error": study["held_out"][
            "shared_ring_error"
        ]["mean_absolute_relative_error"],
        "collective_specific_max_absolute_relative_error": study["held_out"][
            "collective_specific_error"
        ]["max_absolute_relative_error"],
        "planner_correct_scenarios": study["planner"]["correct_scenarios"],
        "planner_total_scenarios": study["planner"]["total_scenarios"],
        "constant_baseline_choice": baseline["choice"],
        "constant_baseline_correct_scenarios": baseline["correct_scenarios"],
    }
    if not _values_match(claim, expected_claim):
        raise ValueError("manifest claim does not match measured evidence")

    provenance = manifest.get("provenance")
    expected_provenance = {
        "measurement_source_revision": study["measurement_source_revision"],
        "config_path": CONFIG_PATH,
        "config_sha256": study["config_sha256"],
        "raw_world_sizes": [2, 4],
    }
    if provenance != expected_provenance:
        raise ValueError("manifest provenance does not match measured evidence")
    return manifest


def main() -> None:
    study = validate_result()
    validate_manifest(study)
    print(
        json.dumps(
            {
                "measurement_rows": 80,
                "collective_specific_mare": study["held_out"][
                    "collective_specific_error"
                ]["mean_absolute_relative_error"],
                "shared_model_mare": study["held_out"]["shared_ring_error"][
                    "mean_absolute_relative_error"
                ],
                "planner": "7/8",
                "constant_baseline": "8/8 context_parallel",
                "status": "verified",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
