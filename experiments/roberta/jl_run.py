#!/usr/bin/env python3
"""Single-seed entry point for the paper experiments."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
METHODS = {
    "ddqn_s01": dict(label="RLCD-FedLoRA", factor="adaptive_score", layers="rl", controller_variant="ddqn_s01"),
    "flexlora_cd": dict(label="FlexLoRA + CD (G=12, validation-controlled common rank)", factor="adaptive_score", layers="all", rank_policy="server_validation_feedback"),
    "spdcfl_cd": dict(label="SPD-CFL + CD (layer-budget proxy)", factor="adaptive_score", layers="spdcfl"),
    "fedfmsl_cd": dict(label="FedFMSL + CD (SAL layer-selection proxy)", factor="adaptive_score", layers="fedfmsl"),
    "rl_ffa_lora": dict(label="RL + FFA-LoRA", factor="fixed_baba", layers="rl", aggregation="ffa"),
    "rl_fedavg_lora": dict(label="RL + FedAvg-LoRA", factor="fixed_baba", layers="rl", aggregation="fedavg"),
    "rl_rolora": dict(label="RL + RoLoRA", factor="fixed_baba", layers="rl"),
    "rl_flora": dict(label="RL + FLoRA (fixed-rank product-SVD controlled variant)", factor="fixed_baba", layers="rl", aggregation="flora"),
    "rl_aslora": dict(label="RL + AS-LoRA", factor="arc_aslora", layers="rl"),
    "centralized": dict(label="Centralized LoRA", factor="joint", layers="all", mode="centralized"),
    "local_only": dict(label="Local-only LoRA", factor="joint", layers="all", mode="local_only"),
    "fixed_g12": dict(label="RLCD-FedLoRA ablation (fixed G=12, r=10)", factor="adaptive_score", layers="all"),
    "random_g": dict(label="RLCD-FedLoRA ablation (uniform random G=1..12, r=10)",
                     factor="adaptive_score", layers="random_g", count_policy="uniform_integer",
                     count_warmup_rounds=2),
}
MAIN_METHODS = tuple(list(METHODS)[:9])
from jl_controller_variants import VARIANTS
from jl_component_ablation import METHOD_MODES, SEED_OFFSETS
for _name, _mode in METHOD_MODES.items():
    METHODS[_name] = dict(label='D01 ablation: ' + _mode.replace('_', ' '),
                         factor='adaptive_score', layers='rl', controller_variant='ddqn_s01',
                         component_ablation=_mode)


def parser(default_method=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=METHODS, default=default_method, required=default_method is None)
    p.add_argument("--setting", choices=["5_5", "10_8"], default="5_5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=200,
                   help="Communication rounds per run (default: 200)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=None, help="Local batch size (default: 64)")
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=None, help="Local learning rate (default: 5e-4)")
    p.add_argument("--devices", default="cuda:0")
    p.add_argument("--workers", type=int, default=1,
                   help="Default serial clients for reproducible seed streams; >1 is explicitly exploratory")
    p.add_argument("--per-gpu-parallel", type=int, default=1)
    p.add_argument("--dropout", choices=["model_default", "disabled"], default="model_default",
                   help="disabled sets model nn.Dropout probabilities to zero; gradients remain enabled")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32",
                   help="Local training autocast; validation, rewards and final testing remain FP32")
    p.add_argument("--attention", choices=["eager", "sdpa"], default="eager")
    p.add_argument("--model-path", default=os.environ.get("JL_MODEL_PATH", "FacebookAI/roberta-base"))
    local_data = ROOT / "data/mimic_icu10_server500_tokenized_512.pkl"
    parent_data = ROOT.parent / "data/mimic_icu10_server500_tokenized_512.pkl"
    p.add_argument("--data-file", default=os.environ.get("JL_DATA_FILE", str(local_data if local_data.is_file() else parent_data)))
    p.add_argument("--output-root", default=str(ROOT / "jl_runs"))
    p.add_argument("--run-name", default=None)
    p.add_argument("--test-every-round", action="store_true", help="Diagnostic predictions only; never selects a model")
    p.add_argument("--validation-target", type=float, default=None)
    p.add_argument("--flex-rank-min", type=int, default=6, help="FlexLoRA+CD only: common-rank floor; initial rank is 10")
    p.add_argument("--flex-rank-max", type=int, default=10, help="FlexLoRA+CD only: common-rank cap")
    p.add_argument("--flex-rank-step", type=int, default=1)
    p.add_argument("--flex-rank-patience", type=int, default=5, help="Consecutive nonimproving server validation loss rounds")
    p.add_argument("--flex-rank-decrease-patience", type=int, default=3, help="Consecutive worsening server validation loss rounds")
    p.add_argument("--flex-rank-min-delta", type=float, default=1e-4, help="Absolute server validation CE improvement tolerance")
    p.add_argument("--dry-run", action="store_true")
    return p


def engine_arguments(args, output):
    if args.method in ("local_only", "centralized") and args.setting != "5_5":
        raise ValueError("Local-only and centralized comparisons use the primary setting 5_5")
    spec = METHODS[args.method]
    m, k = (5, 5) if args.setting == "5_5" else (10, 8)
    values = {
        "jl_method": args.method, "jl_setting": args.setting,
        "jl_data_file": str(Path(args.data_file).resolve()), "jl_output": str(output),
        "jl_model_path": args.model_path, "jl_eval_batch_size": args.eval_batch_size,
        "jl_dropout": args.dropout,
        "jl_precision": args.precision, "jl_attention": args.attention,
        "data": "mimic_icu10_tok512", "client_partition": "icu5_large" if m == 5 else "icu10",
        "n_parties": m, "sample_fraction": k / m,
        "comm_round": args.rounds, "epochs": args.epochs, "init_seed": args.seed,
        "initial_rank": 10, "max_length": 512, "batch_size": args.batch_size if args.batch_size is not None else 64,
        "lr": args.lr if args.lr is not None else 5e-4, "lr_a_scale": 1, "lora_weight_decay": 0,
        "coordinate_schedule": "none" if spec.get("mode") else "baba",
        "factor_selection_mode": spec["factor"], "layer_select_method": spec["layers"],
        "layer_select_mode": "max", "factor_credit_mode": "equal", "factor_score_beta": 0.3,
        "factor_warmup_rounds": 0 if spec.get("mode") else 2, "factor_epsilon": 0.2,
        "layer_random_prob": 0.2, "layer_score_prob": 0.8,
        "if_rl": args.method not in ("centralized", "local_only"),
        "lora": True, "reset_memory": True,
        "gpu_devices": args.devices, "device": args.devices.split(",")[0],
        "max_workers": args.workers, "per_gpu_max_parallel": args.per_gpu_parallel,
        "logdir": str(output), "modeldir": str(output),
    }
    if spec.get('controller_variant'):
        controller = VARIANTS[spec['controller_variant']]
        values.update(jl_controller_variant=spec['controller_variant'],
                      factor_epsilon=controller['coordinate_exploration'],
                      layer_random_prob=controller['coordinate_exploration'],
                      layer_score_prob=1 - controller['coordinate_exploration'])
    if spec.get('component_ablation'):
        values['jl_component_ablation'] = spec['component_ablation']
    if args.method == 'flexlora_cd':
        values.update(min_rank=args.flex_rank_min, max_rank=args.flex_rank_max,
                      jl_rank_policy='server_validation_feedback',
                      jl_rank_step=args.flex_rank_step, jl_rank_patience=args.flex_rank_patience,
                      jl_rank_decrease_patience=args.flex_rank_decrease_patience,
                      jl_rank_min_delta=args.flex_rank_min_delta)
    else:
        values.update(min_rank=10, max_rank=10)
    if args.validation_target is not None:
        values["jl_validation_target"] = args.validation_target
    flags = ["--use_mk_aggregation", "--no_client_tqdm"]
    if args.test_every_round:
        flags.append("--jl_test_every_round")
    return [item for key, value in values.items() for item in ("--" + key, str(value))] + flags


def main(default_method=None):
    args = parser(default_method).parse_args()
    if args.batch_size is None:
        args.batch_size = 64
    if args.lr is None:
        args.lr = 5e-4
    if min(args.rounds, args.epochs, args.batch_size, args.eval_batch_size, args.workers, args.per_gpu_parallel) < 1:
        raise SystemExit("Rounds, epochs, batches and concurrency must be positive")
    if args.validation_target is not None and not 0 <= args.validation_target <= 1:
        raise SystemExit("Validation target must be within [0, 1]")
    if args.validation_target is not None and METHODS[args.method].get('mode'):
        raise SystemExit('Communication-target stopping is only supported for federated runs')
    if args.method == 'flexlora_cd':
        import math
        if (not 0 < args.flex_rank_min < args.flex_rank_max or not args.flex_rank_min <= 10 <= args.flex_rank_max
                or min(args.flex_rank_step, args.flex_rank_patience, args.flex_rank_decrease_patience) < 1
                or not math.isfinite(args.flex_rank_min_delta) or args.flex_rank_min_delta < 0):
            raise SystemExit('Flex rank policy requires distinct positive bounds containing 10, positive step/patience and finite nonnegative min-delta')
    name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if Path(name).name != name or name in (".", ".."):
        raise SystemExit("run-name must be a directory name, not a path")
    output = Path(args.output_root).resolve() / args.setting / args.method / ("seed%d_%s" % (args.seed, name))
    argv = engine_arguments(args, output)
    config = {"method": args.method, "definition": METHODS[args.method], "options": vars(args),
              "implementation_revision": "jl_sdpa_amp_v5_random_g_v1" if args.method == "random_g" else "jl_sdpa_amp_v5",
              "numerics": {"training": args.precision, "evaluation": "fp32",
                           "server_reward": "fp32", "factor_scoring": "fp32",
                           "attention": args.attention},
              "engine_argv": argv, "output": str(output), "phase": "training",
              "selection": "maximum equal-client validation Macro-F1; shared threshold 0.05..0.95 step 0.01",
              "ties": "smallest threshold; earliest checkpoint", "status": "planned"}
    if METHODS[args.method].get('controller_variant'):
        config['implementation_revision'] = 'jl_controller_study_v1_20261108'
        config['controller_study'] = VARIANTS[METHODS[args.method]['controller_variant']]
        config['client_shuffle'] = 'independent (seed, round, client) torch generators'
        if args.test_every_round or args.validation_target is not None:
            raise SystemExit('Controller study uses the full budget and final-only testing')
    if args.method in METHOD_MODES:
        mode = METHOD_MODES[args.method]
        config.update(implementation_revision='jl_d01_component_ablations_v1_20261113',
                      component_ablation=dict(mode=mode, reference='ddqn_s01',
                          warmup_rounds=2, starts_at_round=3,
                          random_layers='uniform subset without replacement of DDQN-selected size',
                          random_factors='independent A/B with probability 0.5 per selected layer',
                          seed_offset=SEED_OFFSETS[mode],
                          rng='private ablation stream; reference selector draws retained',
                          reporting='one result per configured seed'))
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    if not Path(args.data_file).is_file():
        raise SystemExit("Missing explicit dataset file: " + args.data_file)
    output.mkdir(parents=True, exist_ok=False)
    from jl_progress import configure_progress, emit_progress
    configure_progress(output / "jl_progress.log")
    emit_progress(f"[Setup] {args.method} | setting={args.setting} | seed={args.seed} | preparing data/model")
    emit_progress(f"[Setup] model dropout={args.dropout} | independent client backbones")
    emit_progress(f"[Setup] training={args.precision} | attention={args.attention} | evaluation/reward=fp32")
    os.environ["JL_MODEL_PATH"] = args.model_path
    os.environ["LORA_BASELINE_MODE"] = METHODS[args.method].get("aggregation", "")
    os.environ["LAYER_SELECT_METHOD"] = METHODS[args.method]["layers"]
    from jl_protocol import write_json, sha256
    from jl_data import inspect_data
    config["data_sha256"] = sha256(args.data_file)
    model_path = Path(args.model_path)
    config['model_files_sha256'] = {p.name: sha256(p) for p in model_path.glob('*')
                                  if p.is_file() and p.suffix in ('.json', '.bin', '.safetensors')}
    from jl_package import dependencies
    config["code_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in dependencies()}
    import importlib.metadata
    config['environment'] = {name: importlib.metadata.version(name) for name in
                             ['torch', 'transformers', 'datasets', 'numpy', 'scikit-learn']}
    config['environment']['python'] = sys.version
    write_json(output / "jl_config.json", config)
    write_json(output / "jl_data_manifest.json", inspect_data(args.data_file, args.setting))
    config["status"] = "running"
    write_json(output / "jl_config.json", config)
    sys.argv = [str(ROOT / "jl_engine.py")] + argv
    try:
        if METHODS[args.method].get("mode"):
            from jl_bounds import run
            run(METHODS[args.method]["mode"])
        else:
            try:
                runpy.run_path(str(ROOT / "jl_engine.py"), run_name="__main__")
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    raise
        if not (output / "jl_summary.json").is_file():
            raise RuntimeError("Training exited without its validation-selected final report")
        config["status"] = "completed"
    except BaseException as exc:
        config.update(status="failed", error=repr(exc))
        emit_progress(f"[Failed] {args.method} | {exc!r}")
        raise
    finally:
        write_json(output / "jl_config.json", config)
    emit_progress(f"[Completed] {args.method} | training, final testing and output finished: {output}")


if __name__ == "__main__":
    main()
