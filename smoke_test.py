#!/usr/bin/env python3
"""Exercise every released method with tiny random models and synthetic data."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from run_experiments import METHODS, OPT_METHODS, ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--backbones", nargs="+", choices=["roberta", "opt"], default=["roberta", "opt"])
    args = p.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    outcomes = []
    for backbone in args.backbones:
        methods = METHODS if backbone == "roberta" else {m: METHODS[m] for m in sorted(OPT_METHODS)}
        for method, internal in methods.items():
            script = "jl_smoke.py" if backbone == "roberta" else "jl_opt_smoke.py"
            log = root / (backbone + "_" + method + ".log")
            command = [sys.executable, str(ROOT / "experiments" / backbone / script),
                       "--method", internal, "--setting", "5_5",
                       "--output-root", str(root / backbone)]
            print("Checking", backbone, method, flush=True)
            with log.open("w") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            outcomes.append(dict(backbone=backbone, method=method, returncode=result.returncode))
            if result.returncode:
                print(log.read_text()[-6000:])
    # Unequal client sizes and partial participation must also pass the complete path.
    command = [sys.executable, str(ROOT / "experiments/roberta/jl_smoke.py"),
               "--method", "ddqn_s01", "--setting", "10_8", "--output-root", str(root / "roberta")]
    with (root / "roberta_partial.log").open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    outcomes.append(dict(backbone="roberta", method="d01_partial", returncode=result.returncode))
    (root / "outcomes.json").write_text(json.dumps(outcomes, indent=2) + "\n")
    if any(r["returncode"] for r in outcomes):
        raise SystemExit("Some smoke tests failed; see " + str(root))
    print("All synthetic smoke tests passed. Logs:", root)


if __name__ == "__main__":
    main()
