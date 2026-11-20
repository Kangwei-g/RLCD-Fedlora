"""Check the public experiment command without loading data or models."""

import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASELINES = {"d01", "flexlora_cd", "spdcfl_cd", "fedfmsl_cd", "rl_ffa_lora",
             "rl_fedavg_lora", "rl_rolora", "rl_flora", "rl_aslora"}
ABLATIONS = {"fixed_g12", "random_g", "random_layers", "random_factors"}
PARADIGMS = {"local_only", "centralized"}
DECODER = {"d01", "flexlora_cd", "rl_rolora", "rl_fedavg_lora"}


class ExperimentPlanTests(unittest.TestCase):
    def plan(self, backbone):
        result = subprocess.run(
            [sys.executable, str(ROOT / "run_experiments.py"),
             "--backbone", backbone, "--data-file", "unused.pkl", "--dry-run"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        return json.loads(result.stdout)["jobs"]

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

    def test_plan_matches_the_five_paper_experiments(self):
        expected = {(setting, method) for setting in ("5_5", "10_8")
                    for method in BASELINES | ABLATIONS}
        expected |= {("5_5", method) for method in PARADIGMS}
        for backbone, combinations in (("roberta", expected),
                                       ("opt", {("5_5", method) for method in DECODER})):
            jobs = self.plan(backbone)
            self.assertEqual({(job["setting"], job["method"]) for job in jobs}, combinations)
            self.assertEqual(len(jobs), len(combinations) * 3)
            self.assertEqual(len({job["directory"] for job in jobs}), len(jobs))
            for setting, method in combinations:
                self.assertEqual(len([j for j in jobs if (j["setting"], j["method"]) == (setting, method)]), 3)

    def test_retained_entrypoints_generate_valid_commands(self):
        for backbone in ("roberta", "opt"):
            seen = set()
            for job in self.plan(backbone):
                identity = job["setting"], job["method"]
                if identity in seen:
                    continue
                seen.add(identity)
                result = subprocess.run(job["command"] + ["--dry-run"], cwd=ROOT,
                                        capture_output=True, text=True, check=True)
                config = json.loads(result.stdout)
                command = config["engine_argv"]
                self.assertEqual(command[command.index("--comm_round") + 1], "200")


if __name__ == "__main__":
    unittest.main()
