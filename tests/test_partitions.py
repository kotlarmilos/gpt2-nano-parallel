import unittest

from gpt2_nano_parallel.partitions import (
    column_partition,
    context_partitions,
    reconstruct_columns,
    reconstruct_rows,
    row_partition,
)


class PartitionTests(unittest.TestCase):
    def test_column_partition_reconstructs(self) -> None:
        matrix = [[0, 1, 2, 3], [4, 5, 6, 7]]
        self.assertEqual(reconstruct_columns(column_partition(matrix, 2)), matrix)

    def test_row_partition_reconstructs(self) -> None:
        matrix = [[0, 1], [2, 3], [4, 5], [6, 7]]
        self.assertEqual(reconstruct_rows(row_partition(matrix, 2)), matrix)

    def test_context_partition_has_exact_coverage(self) -> None:
        partitions = context_partitions(11, 3)
        positions = [
            position
            for partition in partitions
            for position in range(partition.start, partition.stop)
        ]
        self.assertEqual(positions, list(range(11)))
        self.assertLessEqual(
            max(part.size for part in partitions)
            - min(part.size for part in partitions),
            1,
        )

    def test_invalid_tensor_world_size(self) -> None:
        with self.assertRaises(ValueError):
            column_partition([[1, 2]], 3)

    def test_invalid_context_world_size(self) -> None:
        with self.assertRaises(ValueError):
            context_partitions(2, 3)


if __name__ == "__main__":
    unittest.main()
