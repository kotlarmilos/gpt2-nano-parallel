import unittest

from gpt2_nano_parallel.expert import route_tokens, routing_report


class ExpertTests(unittest.TestCase):
    def test_each_token_gets_top_k_distinct_experts(self) -> None:
        routes = route_tokens(12, 4, 2, top_k=2)
        for token in range(12):
            assigned = [route.expert for route in routes if route.token == token]
            self.assertEqual(len(assigned), 2)
            self.assertEqual(len(set(assigned)), 2)

    def test_accounting_matches_remote_routes(self) -> None:
        report = routing_report(8, 4, 2, hidden_size=512, top_k=1)
        expected = report["remote_routes"] * 512 * 2
        self.assertEqual(report["dispatch_bytes"], expected)
        self.assertEqual(report["total_all_to_all_bytes"], expected * 2)

    def test_invalid_expert_world_size(self) -> None:
        with self.assertRaises(ValueError):
            route_tokens(8, 3, 2)


if __name__ == "__main__":
    unittest.main()
