#!/usr/bin/env python3
"""Run one experiment for each requested backbone/setting/method/seed."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
METHODS = {
    "d01": "ddqn_s01", "flexlora_cd": "flexlora_cd",
    "spdcfl_cd": "spdcfl_cd", "fedfmsl_cd": "fedfmsl_cd",
    "rl_ffa_lora": "rl_ffa_lora", "rl_fedavg_lora": "rl_fedavg_lora",
    "rl_rolora": "rl_rolora", "rl_flora": "rl_flora", "rl_aslora": "rl_aslora",
    "fixed_g12": "fixed_g12", "random_g": "random_g",
    "random_layers": "d01_random_layers", "random_factors": "d01_random_factors",
    "centralized": "centralized", "local_only": "local_only", "fedavg_full": "fedavg_full",
}
OPT_METHODS = {"d01", "flexlora_cd", "rl_rolora", "rl_fedavg_lora"}


def load_config(backbone):
    return json.loads((ROOT / "configs" / (backbone + "_experiments.json")).read_text())


def make_plan(args):
    config = load_config(args.backbone)
    methods = args.methods or config["methods"]
    seeds = args.seeds if args.seeds is not None else config["seeds"]
    settings = args.settings or config["settings"]
    if (len(set(seeds)) != len(seeds) or len(set(methods)) != len(methods)
            or len(set(settings)) != len(settings) or any(s < 0 for s in seeds)):
        raise ValueError("Use unique methods/settings and unique nonnegative seeds")
    if args.backbone == "opt" and (set(methods) - OPT_METHODS or settings != ["5_5"]):
        raise ValueError("The OPT suite covers four methods under setting 5_5")
    rounds = args.rounds if args.rounds is not None else config["rounds"]
    if min(rounds, args.workers) < 1:
        raise ValueError("Rounds and workers must be positive")
    if args.precision == "bf16" and args.device == "cpu":
        raise ValueError("Use --precision fp32 for CPU runs")
    data = Path(args.data_file).expanduser().resolve()
    model = args.model_path or str(ROOT / "models" / args.backbone)
    if Path(model).expanduser().is_dir():
        model = str(Path(model).expanduser().resolve())
    output = Path(args.output_root).expanduser().resolve() / args.backbone
    jobs = []
    for setting in settings:
        for method in methods:
            for seed in seeds:
                internal = METHODS[method]
                destination = output / setting / internal / ("seed%d_standard" % seed)
                command = [sys.executable, str(ROOT / "experiments" / args.backbone / "jl_run.py"),
                           "--method", internal, "--setting", setting, "--seed", str(seed),
                           "--rounds", str(rounds), "--data-file", str(data),
                           "--model-path", model, "--output-root", str(output),
                           "--run-name", "standard", "--devices", args.device,
                           "--workers", str(args.workers), "--per-gpu-parallel", str(args.workers),
                           "--precision", args.precision, "--attention", args.attention,
                           "--dropout", "disabled"]
                if args.backbone == "opt":
                    command += ["--batch-size", "64", "--micro-batch-size", "64"]
                jobs.append(dict(backbone=args.backbone, setting=setting, method=method,
                                 seed=seed, rounds=rounds, command=command,
                                 directory=str(destination), summary=str(destination / "jl_summary.json")))
    return jobs


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", choices=["roberta", "opt"], required=True)
    p.add_argument("--data-file", required=True, help="Trusted cache produced by preprocessing")
    p.add_argument("--model-path", help="Downloaded model directory; defaults to models/<backbone>")
    p.add_argument("--settings", choices=["5_5", "10_8"], nargs="+")
    p.add_argument("--methods", choices=sorted(METHODS), nargs="+")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--rounds", type=int,
                   help="Communication rounds per run (default: 200 for both backbones)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=1, help="Concurrent clients per experiment")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--attention", choices=["eager", "sdpa"], default="sdpa")
    p.add_argument("--output-root", default="runs")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true", help="Skip completed jobs in the identical saved plan")
    return p


def main():
    args = parser().parse_args()
    jobs = make_plan(args)
    plan = dict(schema_version=1, seeds=sorted({j["seed"] for j in jobs}), jobs=jobs)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if not Path(args.data_file).expanduser().is_file():
        raise SystemExit("Dataset file not found")
    root = Path(args.output_root).expanduser().resolve() / args.backbone
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "experiment_plan.json"
    if manifest.exists():
        if not args.resume or json.loads(manifest.read_text()) != plan:
            raise SystemExit("A plan already exists; use --resume with identical options or a new output directory")
    else:
        with manifest.open("x") as handle:
            json.dump(plan, handle, indent=2)
    for index, job in enumerate(jobs, 1):
        directory = Path(job["directory"])
        if directory.exists():
            config_path = directory / "jl_config.json"
            completed = config_path.is_file() and json.loads(config_path.read_text()).get("status") == "completed"
            if args.resume and completed and Path(job["summary"]).is_file():
                print("Skip completed:", directory, flush=True)
                continue
            raise SystemExit("Incomplete or existing output retained: " + str(directory))
        print("[%d/%d] %s %s seed=%d" % (index, len(jobs), job["setting"], job["method"], job["seed"]), flush=True)
        subprocess.run(job["command"], check=True)


if __name__ == "__main__":
    main()
