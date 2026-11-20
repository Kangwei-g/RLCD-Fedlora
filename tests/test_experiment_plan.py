"""Check the public experiment command without loading data or models."""

import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ExperimentPlanTests(unittest.TestCase):
    def test_all_default_experiments_run_for_200_rounds(self):
        for backbone in ("roberta", "opt"):
            with self.subTest(backbone=backbone):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "run_experiments.py"),
                     "--backbone", backbone, "--data-file", "unused.pkl", "--dry-run"],
                    cwd=ROOT, capture_output=True, text=True, check=True,
                )
                jobs = json.loads(result.stdout)["jobs"]
                self.assertTrue(jobs)
                for job in jobs:
                    with self.subTest(backbone=backbone, setting=job["setting"],
                                      method=job["method"], seed=job["seed"]):
                        self.assertEqual(job["rounds"], 200)
                        command = job["command"]
                        self.assertEqual(command[command.index("--rounds") + 1], "200")


if __name__ == "__main__":
    unittest.main()
