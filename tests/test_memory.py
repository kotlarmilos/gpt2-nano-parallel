import unittest

from gpt2_nano_parallel.memory import zero_memory


class MemoryTests(unittest.TestCase):
    def test_zero_reduces_expected_states(self) -> None:
        dp = zero_memory(100, 4, 0)
        stage1 = zero_memory(100, 4, 1)
        stage2 = zero_memory(100, 4, 2)
        stage3 = zero_memory(100, 4, 3)
        self.assertEqual(dp.total, 1600)
        self.assertEqual(stage1.total, 700)
        self.assertEqual(stage2.total, 550)
        self.assertEqual(stage3.total, 400)
        self.assertEqual(stage1.parameters, dp.parameters)
        self.assertEqual(stage2.gradients, dp.gradients / 4)
        self.assertEqual(stage3.parameters, dp.parameters / 4)

    def test_invalid_world_size(self) -> None:
        with self.assertRaises(ValueError):
            zero_memory(100, 0, 1)

    def test_invalid_stage(self) -> None:
        with self.assertRaises(ValueError):
            zero_memory(100, 2, 4)


if __name__ == "__main__":
    unittest.main()
