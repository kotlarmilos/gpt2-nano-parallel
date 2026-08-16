import unittest
from collections import Counter

from gpt2_nano_parallel.pipeline import bubble_fraction, gpipe_schedule


class PipelineTests(unittest.TestCase):
    def test_every_task_runs_forward_and_backward_once(self) -> None:
        events = gpipe_schedule(3, 5)
        counts = Counter(
            (event.stage, event.microbatch, event.phase) for event in events
        )
        self.assertEqual(len(events), 3 * 5 * 2)
        self.assertTrue(all(count == 1 for count in counts.values()))
        for stage in range(3):
            for microbatch in range(5):
                forward = next(
                    event.slot
                    for event in events
                    if (event.stage, event.microbatch, event.phase)
                    == (stage, microbatch, "F")
                )
                backward = next(
                    event.slot
                    for event in events
                    if (event.stage, event.microbatch, event.phase)
                    == (stage, microbatch, "B")
                )
                self.assertLess(forward, backward)

    def test_dependencies_follow_pipeline_order(self) -> None:
        events = gpipe_schedule(4, 3)
        slots = {
            (event.stage, event.microbatch, event.phase): event.slot
            for event in events
        }
        for microbatch in range(3):
            for stage in range(1, 4):
                self.assertLess(
                    slots[(stage - 1, microbatch, "F")],
                    slots[(stage, microbatch, "F")],
                )
                self.assertLess(
                    slots[(stage, microbatch, "B")],
                    slots[(stage - 1, microbatch, "B")],
                )

    def test_bubble_formula(self) -> None:
        self.assertEqual(bubble_fraction(1, 8), 0)
        self.assertAlmostEqual(bubble_fraction(4, 8), 3 / 11)

    def test_invalid_world_shape(self) -> None:
        with self.assertRaises(ValueError):
            gpipe_schedule(0, 2)


if __name__ == "__main__":
    unittest.main()
