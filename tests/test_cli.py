import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


class CliTests(unittest.TestCase):
    def test_smoke_report(self) -> None:
        root = Path(__file__).parents[1]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(root / "src")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "gpt2_nano_parallel.cli",
                "report",
                str(root / "configs" / "smoke.toml"),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        report = json.loads(result.stdout)
        self.assertEqual(report["memory"][3]["evidence"], "analytical")
        self.assertEqual(report["pipeline"]["evidence"], "simulated")
        tensor = report["tensor_parallel"]
        self.assertEqual(
            tensor["row_projection"],
            {"in_features": 2048, "out_features": 512},
        )
        self.assertEqual(tensor["row_input_ranges"][-1]["stop"], 2048)
        self.assertEqual(
            tensor["row_forward_communication"]["bytes_per_rank"],
            16 * 512 * 2,
        )


if __name__ == "__main__":
    unittest.main()
