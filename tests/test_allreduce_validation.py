import math
import unittest
from pathlib import Path

from gpt2_nano_parallel.allreduce_validation import (
    aggregate_rank_measurements,
    analytical_ring_allreduce_seconds,
    compare_measurements,
    fit_ring_constants,
    load_validation_config,
    residual_summary,
    ring_latency_steps,
)


class AllReduceValidationTests(unittest.TestCase):
    def test_analytical_model_reuses_ring_bytes(self) -> None:
        seconds = analytical_ring_allreduce_seconds(
            100,
            4,
            bandwidth_bytes_per_second=100.0,
            latency_seconds=0.01,
        )
        self.assertAlmostEqual(seconds, 1.56)
        self.assertEqual(ring_latency_steps(4), 6)

    def test_fit_recovers_synthetic_constants(self) -> None:
        bandwidth = 2_000_000_000.0
        latency = 0.0002
        measurements = []
        for world_size in (2, 4, 8):
            for message_bytes in (1_048_576, 8_388_608):
                measurements.append(
                    {
                        "world_size": world_size,
                        "message_bytes": message_bytes,
                        "measured_seconds": analytical_ring_allreduce_seconds(
                            message_bytes,
                            world_size,
                            bandwidth,
                            latency,
                        ),
                    }
                )
        fit = fit_ring_constants(measurements, min_message_bytes=1_048_576)
        self.assertTrue(
            math.isclose(
                float(fit["bandwidth_bytes_per_second"]),
                bandwidth,
                rel_tol=1e-9,
            )
        )
        self.assertTrue(
            math.isclose(float(fit["latency_seconds"]), latency, rel_tol=1e-9)
        )

    def test_compare_and_residual_summary(self) -> None:
        measurements = [
            {"world_size": 2, "message_bytes": 4, "measured_seconds": 0.1},
            {
                "world_size": 2,
                "message_bytes": 1_000_000,
                "measured_seconds": 1.0,
            },
        ]
        comparisons = compare_measurements(
            measurements,
            bandwidth_bytes_per_second=1_000_000.0,
            latency_seconds=0.0,
        )
        self.assertAlmostEqual(comparisons[0]["relative_error"], 0.99996)
        summary = residual_summary(comparisons, min_message_bytes=1_000_000)
        self.assertEqual(summary["count"], 1)
        self.assertEqual(summary["mean_signed_error_seconds"], 0.0)

    def test_aggregate_rank_measurements_uses_max_rank_samples(self) -> None:
        rank_results = [
            {
                "rank": 0,
                "measurements": [
                    {"message_bytes": 4, "elapsed_seconds": [0.1, 0.2]}
                ],
            },
            {
                "rank": 1,
                "measurements": [
                    {"message_bytes": 4, "elapsed_seconds": [0.3, 0.1]}
                ],
            },
        ]
        aggregate = aggregate_rank_measurements(
            rank_results,
            world_size=2,
            message_bytes=(4,),
            iterations=2,
            warmup_iterations=1,
        )
        self.assertEqual(aggregate[0]["samples_seconds"], [0.3, 0.2])
        self.assertEqual(aggregate[0]["measured_seconds"], 0.25)

    def test_loads_repository_config(self) -> None:
        root = Path(__file__).parents[1]
        config = load_validation_config(root / "configs" / "allreduce_gloo.toml")
        self.assertEqual(config.world_sizes, (2, 4, 8))
        self.assertEqual(config.message_bytes[-1], 33_554_432)


if __name__ == "__main__":
    unittest.main()
