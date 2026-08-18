import unittest

from gpt2_nano_parallel.topology_planner import (
    CollectiveCall,
    StrategyCandidate,
    collective_seconds,
    compare_holdout,
    evaluate_planner,
    fit_collective_models,
    fit_shared_ring_model,
    rank_candidates,
)


class TopologyPlannerTests(unittest.TestCase):
    @staticmethod
    def _measurements(message_sizes: tuple[int, ...]) -> list[dict[str, object]]:
        bandwidth = 8_000_000_000.0
        latency = 0.00001
        rows = []
        for operation in (
            "all_reduce",
            "reduce_scatter",
            "all_gather",
            "all_to_all",
            "neighbor_exchange",
        ):
            for message_bytes in message_sizes:
                rows.append(
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
                )
        return rows

    def test_collective_models_recover_synthetic_constants(self) -> None:
        calibration = self._measurements((4096, 65536, 1048576))
        holdout = self._measurements((262144,))
        models = fit_collective_models(calibration)
        shared = fit_shared_ring_model(calibration)
        comparisons = compare_holdout(holdout, models, shared)
        self.assertEqual(len(comparisons), 5)
        for row in comparisons:
            self.assertAlmostEqual(row["collective_model_relative_error"], 0.0)
            self.assertAlmostEqual(row["shared_model_relative_error"], 0.0)

    def test_planner_choice_matches_synthetic_measurements(self) -> None:
        calibration = self._measurements((4096, 65536, 1048576))
        holdout = self._measurements((262144,))
        result = evaluate_planner(holdout, fit_collective_models(calibration))
        self.assertEqual(result["top1_accuracy"], 1.0)
        self.assertEqual(result["total_scenarios"], 1)

    def test_rank_candidates_enforces_memory_budget(self) -> None:
        models = fit_collective_models(self._measurements((4096, 65536, 1048576)))
        candidates = (
            StrategyCandidate(
                "too_large",
                (CollectiveCall("all_reduce", 4096),),
                memory_bytes_per_rank=200,
            ),
            StrategyCandidate(
                "fits",
                (CollectiveCall("all_gather", 4096),),
                memory_bytes_per_rank=100,
            ),
        )
        rows = rank_candidates(candidates, 4, models, memory_budget_bytes=100)
        self.assertEqual(rows[0]["name"], "fits")
        self.assertFalse(rows[1]["feasible"])

    def test_rejects_missing_collective_model(self) -> None:
        candidate = StrategyCandidate(
            "missing",
            (CollectiveCall("all_reduce", 4096),),
        )
        with self.assertRaises(ValueError):
            rank_candidates((candidate,), 4, {})


if __name__ == "__main__":
    unittest.main()
