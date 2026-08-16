import unittest

from gpt2_nano_parallel.communication import (
    data_parallel_step,
    ring_attention,
    ring_collective,
    tensor_linear,
    zero_step,
)


class CommunicationTests(unittest.TestCase):
    def test_ring_collective_formulas(self) -> None:
        self.assertEqual(ring_collective(100, 4, "all_reduce").bytes_per_rank, 150)
        self.assertEqual(ring_collective(100, 4, "all_gather").bytes_per_rank, 75)
        self.assertEqual(
            ring_collective(100, 4, "reduce_scatter").bytes_per_rank, 75
        )

    def test_data_parallel_payload(self) -> None:
        result = data_parallel_step(100, 2, gradient_bytes=2)
        self.assertEqual(result.bytes_per_rank, 200)

    def test_zero_stage_three_counts_two_parameter_gathers(self) -> None:
        result = zero_step(100, 2, 3, parameter_bytes=2, gradient_bytes=2)
        self.assertEqual(result.bytes_per_rank, 300)

    def test_tensor_linear_payloads(self) -> None:
        row = tensor_linear(8, 4, 6, 2, "row", element_bytes=2)
        column = tensor_linear(8, 4, 6, 2, "column", element_bytes=2)
        self.assertEqual(row.bytes_per_rank, 96)
        self.assertEqual(column.bytes_per_rank, 64)

    def test_ring_attention_formula(self) -> None:
        result = ring_attention(16, 4, 2, element_bytes=2)
        self.assertEqual(result.bytes_per_rank, (128, 128))
        self.assertEqual(result.aggregate_bytes, 256)

    def test_ring_attention_reports_uneven_rank_volumes(self) -> None:
        result = ring_attention(10, 2, 3, element_bytes=2)
        self.assertEqual(result.bytes_per_rank, (56, 56, 48))
        self.assertEqual(result.aggregate_bytes, 160)

    def test_invalid_world_size(self) -> None:
        with self.assertRaises(ValueError):
            ring_collective(10, 0, "all_reduce")
        with self.assertRaises(ValueError):
            ring_attention(2, 4, 3)


if __name__ == "__main__":
    unittest.main()
