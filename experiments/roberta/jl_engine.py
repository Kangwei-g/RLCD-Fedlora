import argparse
import copy
import csv
import datetime
import json
import os
import pickle
from pathlib import Path
import random
import re
import subprocess
import threading
import concurrent.futures
import gc
import math
from contextlib import nullcontext
from itertools import chain
from typing import Optional, Tuple
import psutil
import time
import hickle as hkl
import pynvml
import torch
import numpy as np
from datasets import concatenate_datasets
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from torch import nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoModelForSequenceClassification, \
    GPT2Model, GPT2ForSequenceClassification
import logging
from lora.lora_finetune import add_lora, freeze_layer, unfrozen
from rl.Agent import DQNAgent
from jl_controller_study import ControllerStudy
from jl_component_ablation import ComponentAblation, METHOD_MODES
from jl_controller_variants import VARIANTS, client_shuffle_seed
from datetime import datetime
from shared_lora_runtime import (
    SHARED_BACKBONE_REGISTRY,
    create_lora_client_states,
    sync_lora_client_state_from_global,
    train_shared_lora_client,
)
from arc_aslora import (
    ASLoRAComponentSelector,
    OFFICIAL_EMA_BETA,
    OFFICIAL_FD_EPS,
    OFFICIAL_GAMMA,
    OFFICIAL_SCORE_EPS,
    OFFICIAL_T0,
    OFFICIAL_TMIN,
)
from outcome_alignment import (
    OutcomeAlignmentFactorSelector,
    compute_server_validation_alignment,
)
from cmfv import (
    CMFVSelector,
    compute_client_factor_coherence,
    compute_server_task_fit,
)
from gi_cmfv import (
    GICMFVSelector,
    compute_gi_cmfv_scores,
    compute_server_gi_cmfv_scores,
)
from qgs import QGSSelector, compute_federated_qgs_scores
from cfvg import (
    aggregate_uploaded_deltas,
    compute_counterfactual_federated_validation_gains,
)


# Importing necessary modules from the project

from jl_trainer import compute_class_weights, configure_dropout, fit, prepare_tensor_dataloader, test
from jl_protocol import EvaluationProtocol, no_local_evaluation, write_json, write_csv
from jl_rank import ValidationFeedbackRank, lora_layout, match_checkpoint_layout, resize_common_rank
from jl_progress import emit_progress, report_progress, FINAL_TEST_NOTE
from utils.gpu_memory_guard import (
    apply_gpu_memory_reserve,
    get_gpu_memory_guard,
    init_gpu_memory_guard,
)

# 还有trainer.py要改！
model_file = os.environ.get('JL_MODEL_PATH', 'FacebookAI/roberta-base')
import sys

# 多线程下避免 HF tokenizer 触发并行 fork 警告 / 潜在卡死
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def str2bool(v):
    """Parse CLI booleans correctly (argparse type=bool breaks on 'False' -> True)."""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off", ""}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def use_shared_backbone_lora(args):
    return bool(
        args.shared_backbone_lora
        and args.lora
        and getattr(args, "model", "roberta") == "roberta"
        and args.finetune is None
        and not args.if_sa
    )


# Function to parse command line arguments
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training. Default is 64.")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate for the optimizer. Default is 0.001.")
    parser.add_argument(
        "--lr_a_scale",
        type=float,
        default=4.0,
        help="LR multiplier for lora_a on pure-A coordinate rounds; compensates the small "
             "||B|| that otherwise makes A-phase updates nearly a no-op. Set 1.0 to disable.",
    )
    parser.add_argument(
        "--lora_weight_decay",
        type=float,
        default=0.0,
        help="Weight decay applied to LoRA factors. The old 0.01 lets Adam's decay term "
             "dominate the weak A-phase gradient and shrink lora_a toward 0.",
    )
    parser.add_argument(
        "--lr_head",
        type=float,
        default=None,
        help="Classifier head LR for full fine-tuning (default: 5x --lr when finetune=full).",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of local epochs to run. Default is 3.")
    parser.add_argument("--n_parties", type=int, default=10,
                        help="Number of workers in a distributed cluster. Default is 5.")
    parser.add_argument("--comm_round", type=int, default=5,
                        help="Maximum number of communication rounds. Default is 5.")
    parser.add_argument(
        "--early_stop_macro_f1",
        type=float,
        default=None,
        help="Stop after a round when client-mean test macro-F1 >= this threshold (None = disabled).",
    )
    parser.add_argument("--init_seed", type=int, default=0, help="Seed for random number generation. Default is 0.")
    parser.add_argument("--datadir", type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument("--logdir", type=str, required=False, default="./logs/", help="Log directory path")
    parser.add_argument("--modeldir", type=str, required=False, default="./models/", help="Model directory path")

    parser.add_argument("--sample_fraction", type=float, default=0.5,
                        help="Fraction of clients to be sampled in each round. Default is 1.0.")
    parser.add_argument("--device", type=str, default="cuda:1", help="The device to run the program")
    parser.add_argument("--optimizer", type=str, default="adam",
                        help="Optimizer to use for training. Default is 'adam'.")

    parser.add_argument("--initial_rank", type=int, default=10, help="initial rank for lora")
    parser.add_argument(
        "--initial_num_layers",
        type=int,
        default=12,
        help="Initial layer budget G1 for round 1 (sampled uniformly without replacement).",
    )

    parser.add_argument("--if_rl", type=str2bool, default=False, help="Whether to use RL or not. Default is False.")
    parser.add_argument("--cd", type=str, default=None, help="Whether to use lora or not. Default is False.")
    parser.add_argument("--lora", type=str2bool, default=True, help="Whether to use lora or not. Default is False.")
    parser.add_argument("--finetune", type=str, default=None ,help="Whether to use lora or not. Default is False.")
    parser.add_argument("--if_ffa", type=str2bool, default=False)
    parser.add_argument("--acc_threshold",type=int,default=0,help="threshold for rank")
    parser.add_argument("--max_rank", type=int, default=10, help="threshold for rank")
    parser.add_argument("--min_rank",type=int,default=10,help="threshold for rank")
    parser.add_argument("--model", type=str, default='roberta', help="model")
    parser.add_argument("--out_dim",type=int,default=2)
    parser.add_argument("--data", type=str, default="eICU", help="Whether to use lora or not. Default is False.")
    parser.add_argument(
        "--client_partition",
        type=str,
        default="icu10",
        choices=["icu10", "icu5_large"],
        help="ICU subset: icu10 (all 10 clients) or icu5_large (5 larger *_1500 clients).",
    )
    parser.add_argument("--max_length", type=int, default=128, help="Tokenizer max_length for raw-text datasets.")
    parser.add_argument("--if_sa", type=str2bool, default=False)
    parser.add_argument("--if_sa_initial", type=str2bool, default=False)
    parser.add_argument("--if_sa_anneal", type=str2bool, default=False)
    parser.add_argument('--change',type = int,default=100)
    parser.add_argument("--reset_memory", type=str2bool, default=True, help="Whether to reset RL memory.")
    parser.add_argument("--regenerate_layers", type=str2bool, default=True,
                        help="Whether to regenerate layers_output.json for new dataset.")
    parser.add_argument("--max_workers", type=int, default=0,
                        help="Thread pool size (0 = all clients this round).")
    parser.add_argument("--gpu_devices", type=str, default="cuda:0,cuda:1",
                        help="Comma-separated GPUs for client training, e.g. cuda:0,cuda:1")
    parser.add_argument(
        "--per_gpu_max_parallel",
        type=str,
        default="3",
        help="每张 GPU 上最多同时跑几个客户端。可写单个整数（各卡相同），"
             "或按 --gpu_devices 顺序逗号分隔，如 2,4 表示 cuda:0 最多 2 路、cuda:1 最多 4 路。"
             "设为 0 表示不限制；显存不足时可改为 1 或 2,4。",
    )
    parser.add_argument(
        "--shared_backbone_lora",
        type=str2bool,
        default=False,
        help="Use one frozen backbone per GPU while clients keep only LoRA states.",
    )
    parser.add_argument(
        "--gpu_memory_reserve",
        type=str,
        default="",
        help="每张 GPU 为本进程保留的显存上限（GB），防止他人挤占。"
             "启动即占卡；训练/验证时仅缩到保底占位，轮间占满；后台线程持续补占。"
             "格式: cuda:0:75 或 cuda:0:75,cuda:1:75。",
    )
    parser.add_argument(
        "--gpu_memory_training_headroom",
        type=float,
        default=72.0,
        help="训练/验证阶段留给计算的显存（GB）。保底占位 = reserve - headroom（如 75-72=3GB）。",
    )
    parser.add_argument(
        "--gpu_memory_persistent_reserve",
        action="store_true",
        help="Keep the reserved CUDA allocator pool owned by this process during "
             "compute so training reuses it without exposing it to other jobs.",
    )
    parser.add_argument(
        "--no_client_tqdm",
        action="store_true",
        help="关闭各客户端 fit/test 的 tqdm（默认开启；多线程时多行进度可能交错，不需要展示时再加本项）。",
    )
    parser.add_argument(
        "--client_tqdm",
        action="store_true",
        help="兼容旧命令行：各客户端 tqdm 已默认开启，无需再传。",
    )
    parser.add_argument(
        "--evaluate_client_test_every_round",
        action="store_true",
        help="Evaluate each selected client on its test split after local training. "
             "Metrics are reporting-only and never enter aggregation or RL/factor feedback.",
    )
    parser.add_argument(
        "--epsilon_layer",
        type=float,
        default=0.2,
        help="Epsilon for epsilon-greedy layer selection. Default 0.2.",
    )
    parser.add_argument(
        "--epsilon_matrix",
        type=float,
        default=0.2,
        help="Epsilon for matrix-related exploration (logged in run name). Default 0.2.",
    )
    parser.add_argument(
        "--layer_select_mode",
        type=str,
        default="max",
        choices=["min", "max"],
        help="Greedy layer pick: 'min' = smallest contribution first, 'max' = largest first.",
    )
    parser.add_argument(
        "--layer_select_method",
        type=str,
        default="rl",
        choices=["rl", "all", "fedfmsl", "spdcfl", "random_g", "none"],
        help="Where-to-fine-tune: rl (Double DQN), all (all transformer layers), "
             "fedfmsl, spdcfl, random_g (uniform count), or none (FedAvg-Full).",
    )
    parser.add_argument(
        "--fedfmsl_queue_size",
        type=int,
        default=5,
        help="FedFMSL capability queue length Q (default 5).",
    )
    parser.add_argument(
        "--fedfmsl_delta",
        type=float,
        default=0.005,
        help="FedFMSL bottleneck threshold delta on server macro-F1 increment.",
    )
    parser.add_argument(
        "--spdcfl_dropout_step",
        type=int,
        default=1,
        help="SPD-CFL: layers to drop when SGC stagnates (default 1).",
    )
    parser.add_argument(
        "--spdcfl_ema_decay",
        type=float,
        default=0.9,
        help="SPD-CFL: EMA decay theta for SGC (default 0.9).",
    )
    parser.add_argument(
        "--rl_checkpoint_dir",
        type=str,
        default="",
        help="Load offline pre-trained DQN (q_network.pth, target_network.pth, memory.pkl) from this directory.",
    )
    parser.add_argument(
        "--online_epsilon_start",
        type=float,
        default=0.2,
        help="Epsilon for DQN layer-count exploration when warm-starting from rl_checkpoint_dir.",
    )
    parser.add_argument(
        "--use_mk_aggregation",
        action="store_true",
        help="Use (M/K)*|D_k|/|D| aggregation for partial client participation (Eq.3).",
    )
    parser.add_argument(
        "--coordinate_schedule",
        type=str,
        default="bba",
        choices=list(COORDINATE_SCHEDULES.keys()),
        help="LoRA matrix alternation rhythm: bba (default), bbb, bbba, baba, bbbaaa, b10a1, b20a1, b10_bba, b10_b2a2, b20_b10a10, b10a10, b5a5, b20a20, or bbaa.",
    )
    parser.add_argument(
        "--factor_selection_mode",
        type=str,
        default="global_adaptive",
        choices=[
            "joint",
            "fixed_baba",
            "global_adaptive",
            "adaptive_score",
            "product_delta",
            "arc_aslora",
            "outcome_alignment",
            "cmfv",
            "gi_cmfv",
            "qgs",
            "cfvg",
        ],
        help="LoRA factor selector. joint trains all A/B factors in centralized/local-only controls. "
             "product_delta keeps ARC layer coordination, "
             "alternates one factor across selected layers during warm-up, and assigns "
             "validation reward by each layer's effective A@B product change; "
             "arc_aslora keeps the ARC resource module and replaces "
             "only its A/B selector with AS-LoRA gradient-curvature scoring; "
             "outcome_alignment uses server-validation gradient/update alignment; "
             "cmfv jointly values layer-factor coordinates before training; "
             "gi_cmfv preserves RL/CD layer decisions and applies gauge-invariant, "
             "client-counterfactual factor valuation; qgs uses deterministic "
             "quotient Gauss-Southwell selection on the participating-client "
             "mean validation gradient; cfvg executes separate multi-step "
             "A/B optimizer trajectories and selects by paired federated "
             "validation gain.",
    )
    parser.add_argument(
        "--factor_warmup_rounds",
        type=int,
        default=2,
        help="Global adaptive factor-selection warm-up rounds.",
    )
    parser.add_argument(
        "--layer_random_prob",
        type=float,
        default=0.20,
        help="Adaptive layer-selection random branch probability (fixed 20%%).",
    )
    parser.add_argument(
        "--layer_score_prob",
        type=float,
        default=0.80,
        help="Adaptive layer-selection score branch probability (fixed 80%%).",
    )
    parser.add_argument(
        "--factor_epsilon",
        type=float,
        default=0.20,
        help="Global adaptive factor-selection epsilon-greedy exploration probability.",
    )
    parser.add_argument(
        "--factor_max_staleness",
        type=int,
        default=5,
        help="Maximum allowed rounds since a factor was last selected before forcing it.",
    )
    parser.add_argument(
        "--factor_score_beta",
        type=float,
        default=0.20,
        help="EMA beta for adaptive factor contribution scores.",
    )
    parser.add_argument(
        "--factor_reward_normalization",
        type=str,
        default="per_layer",
        choices=["per_layer"],
        help="Reward normalization mode for global adaptive factor score update.",
    )
    parser.add_argument(
        "--factor_score_clip",
        type=float,
        default=None,
        help="Optional symmetric clip for factor utility before EMA update.",
    )
    parser.add_argument(
        "--factor_credit_mode",
        type=str,
        default="equal",
        choices=["equal", "rms"],
        help="Credit assignment for adaptive_score factor scores: equal split or RMS-weighted.",
    )
    parser.add_argument("--aslora_ema_beta", type=float, default=OFFICIAL_EMA_BETA)
    parser.add_argument("--aslora_fd_eps", type=float, default=OFFICIAL_FD_EPS)
    parser.add_argument("--aslora_norm_eps", type=float, default=OFFICIAL_SCORE_EPS)
    parser.add_argument("--aslora_temperature_0", type=float, default=OFFICIAL_T0)
    parser.add_argument("--aslora_temperature_min", type=float, default=OFFICIAL_TMIN)
    parser.add_argument("--aslora_temperature_gamma", type=float, default=OFFICIAL_GAMMA)
    parser.add_argument(
        "--aslora_projection_seed_offset",
        type=int,
        default=104729,
        help="Independent seed offset for cached Gaussian projection matrices.",
    )
    parser.add_argument("--alignment_min_observations", type=int, default=2)
    parser.add_argument("--alignment_ema_decay", type=float, default=0.9)
    parser.add_argument("--alignment_local_weight", type=float, default=0.7)
    parser.add_argument("--alignment_ucb_coefficient", type=float, default=0.25)
    parser.add_argument("--alignment_reward_scale_decay", type=float, default=0.9)
    parser.add_argument("--alignment_credit_clip", type=float, default=1.0)
    parser.add_argument("--alignment_validation_batch_size", type=int, default=32)
    parser.add_argument("--cmfv_validation_batch_size", type=int, default=32)
    parser.add_argument("--cmfv_curvature_fd_eps", type=float, default=1e-3)
    parser.add_argument("--cmfv_coherence_prior", type=float, default=0.5)
    parser.add_argument("--cmfv_coherence_ema_decay", type=float, default=0.9)
    parser.add_argument("--cmfv_count_discount", type=float, default=0.97)
    parser.add_argument("--cmfv_confidence_z", type=float, default=1.0)
    parser.add_argument("--cmfv_min_task_fit", type=float, default=1e-10)
    parser.add_argument("--cmfv_exploration_beta", type=float, default=0.05)
    parser.add_argument("--cmfv_posterior_temperature", type=float, default=0.5)
    parser.add_argument("--gi_cmfv_validation_batch_size", type=int, default=32)
    parser.add_argument("--gi_cmfv_rank_rtol", type=float, default=1e-5)
    parser.add_argument("--gi_cmfv_ema_decay", type=float, default=0.9)
    parser.add_argument("--gi_cmfv_count_discount", type=float, default=0.98)
    parser.add_argument("--gi_cmfv_server_weight", type=float, default=0.2)
    parser.add_argument(
        "--gi_cmfv_disagreement_penalty", type=float, default=0.2
    )
    parser.add_argument("--gi_cmfv_temperature_0", type=float, default=1.0)
    parser.add_argument("--gi_cmfv_temperature_min", type=float, default=0.1)
    parser.add_argument("--gi_cmfv_temperature_gamma", type=float, default=0.95)
    parser.add_argument("--qgs_validation_batch_size", type=int, default=32)
    parser.add_argument("--qgs_rank_rtol", type=float, default=1e-5)
    parser.add_argument("--cfvg_probe_train_batch_size", type=int, default=64)
    parser.add_argument("--cfvg_probe_validation_batch_size", type=int, default=32)
    parser.add_argument("--cfvg_probe_steps", type=int, default=4)
    parser.add_argument("--cfvg_probe_validation_batches", type=int, default=2)
    parser.add_argument("--cfvg_confidence_z", type=float, default=1.0)
    parser.add_argument('--jl_data_file', required=True)
    parser.add_argument('--jl_output', required=True)
    parser.add_argument('--jl_model_path', default=model_file)
    parser.add_argument('--jl_method', default='rlcd')
    parser.add_argument('--jl_controller_variant', choices=tuple(VARIANTS), default=None)
    parser.add_argument('--jl_component_ablation', choices=['none', 'random_layers', 'random_factors'], default='none')
    parser.add_argument('--jl_setting', default='5_5')
    parser.add_argument('--jl_eval_batch_size', type=int, default=64)
    parser.add_argument('--jl_dropout', choices=['model_default', 'disabled'], default='model_default')
    parser.add_argument('--jl_precision', choices=['fp32', 'bf16'], default='fp32')
    parser.add_argument('--jl_attention', choices=['eager', 'sdpa'], default='eager')
    parser.add_argument('--jl_test_every_round', action='store_true')
    parser.add_argument('--jl_validation_target', type=float, default=None)
    parser.add_argument('--jl_rank_policy', choices=['fixed', 'server_validation_feedback'], default='fixed')
    parser.add_argument('--jl_rank_step', type=int, default=1)
    parser.add_argument('--jl_rank_patience', type=int, default=5)
    parser.add_argument('--jl_rank_decrease_patience', type=int, default=3)
    parser.add_argument('--jl_rank_min_delta', type=float, default=1e-4)
    args = parser.parse_args()
    component_active = args.jl_method in METHOD_MODES
    if (args.jl_component_ablation != METHOD_MODES.get(args.jl_method, 'none')
            or (component_active and args.jl_controller_variant != 'ddqn_s01')):
        parser.error('Component ablations require their matching method and frozen D01 controller')
    if args.jl_controller_variant and ((not component_active and args.jl_method != args.jl_controller_variant)
            or args.layer_select_method != 'rl' or args.factor_selection_mode != 'adaptive_score'
            or args.factor_warmup_rounds != 2 or not args.if_rl or not args.lora):
        parser.error('Controller study requires matching method, adaptive factors and two warm-up rounds')
    if args.early_stop_macro_f1 is not None:
        parser.error('Test-based early stopping is disabled; use --jl_validation_target')
    if args.shared_backbone_lora or args.evaluate_client_test_every_round or args.if_sa or args.if_sa_initial or args.if_sa_anneal:
        parser.error('JL uses the audited full-model client runtime without local test evaluation/rank annealing')
    if args.rl_checkpoint_dir or not args.reset_memory:
        parser.error('JL pilot runs require a fresh controller; legacy checkpoints are incompatible')
    if args.jl_method == 'flexlora_cd':
        if (args.jl_rank_policy != 'server_validation_feedback' or not args.lora or not args.if_rl
                or args.layer_select_method != 'all' or args.factor_selection_mode != 'adaptive_score'
                or args.initial_rank != 10):
            parser.error('FlexLoRA+CD requires common-rank adjustment, initial r=10, G=12 and our adaptive factor policy')
        ValidationFeedbackRank(args.initial_rank, args.min_rank, args.max_rank, args.jl_rank_step,
                               args.jl_rank_patience, args.jl_rank_decrease_patience,
                               args.jl_rank_min_delta, args.factor_warmup_rounds)
    elif args.jl_rank_policy != 'fixed' or (args.lora and (args.initial_rank, args.min_rank, args.max_rank) != (10, 10, 10)):
        parser.error('All other JL LoRA methods retain fixed rank 10')
    if args.jl_method == 'fixed_g12' and (args.layer_select_method != 'all' or args.factor_selection_mode != 'adaptive_score'):
        parser.error('fixed_g12 retains our adaptive factor policy at G=12')
    if args.jl_method == 'random_g' and (args.layer_select_method != 'random_g'
            or args.factor_selection_mode != 'adaptive_score' or not args.lora or not args.if_rl
            or args.factor_warmup_rounds != 2):
        parser.error('random_g requires adaptive factors, fixed rank 10 and two all-layer warm-up rounds')
    if args.jl_method in ('centralized', 'local_only'):
        if (not args.lora or args.if_rl or args.layer_select_method != 'all'
                or args.factor_selection_mode != 'joint' or args.coordinate_schedule != 'none'):
            parser.error('Centralized/local-only controls jointly train all LoRA A/B factors without layer or factor selection')
    elif args.factor_selection_mode == 'joint':
        parser.error('The joint factor mode is only used by centralized/local-only controls')
    return args

# Function to set the seed for all random number generators to ensure reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

def infer_lora_rank_from_state_dict(state_dict):
    """Return the LoRA rank encoded by a checkpoint state dict."""
    for name, value in state_dict.items():
        if name.endswith("lora_a") and getattr(value, "ndim", 0) == 2:
            return int(value.shape[1])
        if name.endswith("lora_b") and getattr(value, "ndim", 0) == 2:
            return int(value.shape[0])
    return None

def _log_training_eta(logger, round_idx, total_rounds, round_elapsed_sec, training_elapsed_sec, method=""):
    report_progress(round_idx + 1, total_rounds, round_elapsed_sec, training_elapsed_sec,
                    label=method, note=FINAL_TEST_NOTE, logger=logger)


def _fed_progress_prefix(round_idx, total_rounds, client_id=None, tag=None):
    """供 tqdm / 打印使用的轮次与客户端标签。"""
    bits = [f"第{round_idx + 1}/{total_rounds}轮"]
    if client_id is not None:
        bits.append(f"客户端{client_id}")
    if tag:
        bits.append(tag)
    return "[" + "|".join(bits) + "] "


def _fed_final_progress_prefix(total_rounds):
    """全部通信轮结束后的最终验证。"""
    return f"[全部{total_rounds}轮联邦已完成|最终全局验证] "


def _write_client_round_metrics(log_dir, rows):
    """Atomically expose partial client metrics while a long run is still active."""
    if not rows:
        return None
    json_path = os.path.join(log_dir, "client_round_test_metrics.json")
    csv_path = os.path.join(log_dir, "client_round_test_metrics.csv")
    json_tmp = json_path + ".tmp"
    csv_tmp = csv_path + ".tmp"
    with open(json_tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    with open(csv_tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(json_tmp, json_path)
    os.replace(csv_tmp, csv_path)
    return csv_path


def _write_rows_csv_atomic(path, rows):
    if not rows:
        return None
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)
    return path


def _resolve_devices(args, n_clients):
    if args.gpu_devices:
        return [d.strip() for d in args.gpu_devices.split(",") if d.strip()]
    if torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return [args.device]


def _parse_per_gpu_limits(spec, devices):
    """Parse per_gpu_max_parallel: int/str '3' or '2,4' aligned with unique devices order."""
    if spec is None:
        return None
    text = str(spec).strip()
    if not text:
        return None
    parts = [p.strip() for p in text.split(",")]
    unique_devices = []
    for d in devices:
        if d not in unique_devices:
            unique_devices.append(d)
    if len(parts) == 1:
        n = int(parts[0])
        if n <= 0:
            return None
        return {d: max(1, n) for d in unique_devices}
    if len(parts) != len(unique_devices):
        raise ValueError(
            f"--per_gpu_max_parallel has {len(parts)} value(s) but "
            f"{len(unique_devices)} GPU(s) in --gpu_devices: {unique_devices}"
        )
    limits = {}
    for d, p in zip(unique_devices, parts):
        n = int(p)
        if n <= 0:
            return None
        limits[d] = max(1, n)
    return limits


def _gpu_guard_devices(args):
    devices = list(dict.fromkeys(_resolve_devices(args, args.n_parties)))
    device = str(args.device).strip()
    if device and device not in devices:
        devices.append(device)
    return devices


def _gpu_enter_compute(args, logger=None):
    guard = get_gpu_memory_guard()
    if guard is not None:
        guard.enter_compute(_gpu_guard_devices(args), logger=logger)


def _gpu_enter_idle(args, logger=None):
    guard = get_gpu_memory_guard()
    if guard is not None:
        guard.enter_idle(_gpu_guard_devices(args), logger=logger)


def _gpu_semaphores(devices, per_gpu_max_parallel):
    """每张 GPU 一个信号量；per_gpu_max_parallel<=0 表示不限制（返回 None）。"""
    limits = _parse_per_gpu_limits(per_gpu_max_parallel, devices)
    if limits is None:
        return None
    return {d: threading.Semaphore(limits[d]) for d in limits}


def _execute_client_training_on_device(
        net, args, trainset, valiset, testset, frozen_layer, not_com_layer,
        device_id, prefix, round_idx, total_rounds, net_id, no_client_tqdm, tqdm_pos,
        as_lora_active_layers=None, gi_cmfv_active_layers=None):
    arc_aslora_mode = str(getattr(args, "factor_selection_mode", "")) == "arc_aslora"
    gi_cmfv_mode = str(getattr(args, "factor_selection_mode", "")) == "gi_cmfv"
    arc_aslora_config = None
    if arc_aslora_mode:
        arc_aslora_config = {
            "active_layers": list(as_lora_active_layers or []),
            "eta": float(args.lr),
            "fd_eps": float(args.aslora_fd_eps),
            "norm_eps": float(args.aslora_norm_eps),
            "projection_seed": int(args.init_seed) + int(args.aslora_projection_seed_offset),
        }
    fit_result = fit(
        net,
        trainset,
        args.epochs,
        args.lr,
        args.optimizer,
        device=device_id,
        finetune=args.finetune,
        model=args.model,
        progress_prefix=prefix,
        disable_tqdm=no_client_tqdm,
        tqdm_position=tqdm_pos,
        max_length=args.max_length,
        batch_size=args.batch_size,
        lr_head=args.lr_head,
        lr_a_scale=getattr(args, "lr_a_scale", 1.0),
        lora_weight_decay=getattr(args, "lora_weight_decay", None),
        arc_aslora_config=arc_aslora_config,
        precision=getattr(args, "jl_precision", "fp32"),
        shuffle_seed=(client_shuffle_seed(args.init_seed, round_idx, net_id)
                      if args.jl_controller_variant else None),
    )
    if arc_aslora_mode:
        train_acc, train_loss, arc_aslora_scores, arc_aslora_diagnostics = fit_result
    else:
        train_acc, train_loss = fit_result
        arc_aslora_scores, arc_aslora_diagnostics = None, None
    if gi_cmfv_mode:
        gi_validation_loader = prepare_tensor_dataloader(
            valiset,
            model_name=args.model,
            max_length=args.max_length,
            progress_prefix=prefix,
            split_name="GI-CMFV client validation",
            batch_size=int(args.gi_cmfv_validation_batch_size),
            shuffle=False,
        )
        try:
            gi_validation_batch = next(iter(gi_validation_loader))
        except StopIteration as exc:
            raise RuntimeError(
                f"GI-CMFV client {net_id} validation split is empty"
            ) from exc
        arc_aslora_scores, arc_aslora_diagnostics = compute_gi_cmfv_scores(
            model=net,
            batch=gi_validation_batch,
            active_layers=list(gi_cmfv_active_layers or []),
            device=device_id,
            finetune=args.finetune,
            rank_rtol=float(args.gi_cmfv_rank_rtol),
        )
        print(
            f"{prefix}GI-CMFV counterfactual validation evidence complete: "
            f"layers={sorted(arc_aslora_scores)}, "
            f"loss={arc_aslora_diagnostics['validation_loss']:.6f}",
            flush=True,
        )
    # Only the aggregated model is evaluated; local training never reads test data.
    val_metrics = no_local_evaluation()
    test_metrics = dict(val_metrics)
    test_acc = test_loss = float('nan')

    layers_params = {}
    if if_lora:
        for name, param in net.named_parameters():
            if name.split('.')[-1] in ['lora_a', 'lora_b'] and name not in frozen_layer and name not in not_com_layer:
                t = param.detach().cpu().clone()
                if param.grad is not None:
                    t.grad = param.grad.detach().cpu().clone()
                layers_params[name] = t
    else:
        if hasattr(net, "backbone_model"):
            param_iter = chain(net.backbone_model.named_parameters(), net.named_parameters())
        else:
            param_iter = net.named_parameters()
        for name, param in param_iter:
            if param.requires_grad:
                t = param.detach().cpu().clone()
                if param.grad is not None:
                    t.grad = param.grad.detach().cpu().clone()
                layers_params[name] = t

    net.to("cpu")
    return (
        train_acc,
        train_loss,
        test_acc,
        test_loss,
        val_metrics,
        test_metrics,
        layers_params,
        arc_aslora_scores,
        arc_aslora_diagnostics,
    )


def train_single_client(net_id, net, args, trainset, valiset, testset, frozen_layer, not_com_layer,
                        device_id, log_lock, round_idx, total_rounds,
                        gpu_semaphores, no_client_tqdm, shared_source_model=None,
                        as_lora_active_layers=None, gi_cmfv_active_layers=None):
    prefix = _fed_progress_prefix(round_idx, total_rounds, client_id=net_id)
    tqdm_pos = min(int(net_id), 40)
    with log_lock:
        logger.info("%sTraining network %s on %s. n_training: %d" % (
            prefix, str(net_id), device_id, len(trainset)))
        logger.info(f'{prefix}------------第{net_id}个客户端训练-------------')

    shared_runtime = use_shared_backbone_lora(args) and shared_source_model is not None
    if if_lora and not shared_runtime:
        net = freeze_layer(net, frozen_layer)

    lock_ctx = gpu_semaphores[device_id] if gpu_semaphores is not None else nullcontext()
    with lock_ctx:
        emit_progress(f"[训练阶段] {args.jl_method} | {prefix.strip()} | device={device_id}", logger=logger)
        if shared_runtime:
            val_prefix = _fed_progress_prefix(round_idx, total_rounds, client_id=net_id, tag="阈值验证集")
            test_prefix = _fed_progress_prefix(round_idx, total_rounds, client_id=net_id, tag="测试集")
            train_acc, train_loss, test_acc, test_loss, val_metrics, test_metrics, layers_params = (
                train_shared_lora_client(
                    net,
                    shared_source_model,
                    args,
                    trainset,
                    valiset,
                    testset,
                    frozen_layer,
                    not_com_layer,
                    device_id,
                    logger,
                    prefix,
                    val_prefix,
                    test_prefix,
                    no_client_tqdm,
                    tqdm_pos,
                )
            )
            arc_aslora_scores, arc_aslora_diagnostics = None, None
        else:
            (
                train_acc,
                train_loss,
                test_acc,
                test_loss,
                val_metrics,
                test_metrics,
                layers_params,
                arc_aslora_scores,
                arc_aslora_diagnostics,
            ) = (
                _execute_client_training_on_device(
                    net, args, trainset, valiset, testset, frozen_layer, not_com_layer,
                    device_id, prefix, round_idx, total_rounds, net_id, no_client_tqdm, tqdm_pos,
                    as_lora_active_layers=as_lora_active_layers,
                    gi_cmfv_active_layers=gi_cmfv_active_layers,
                )
            )

    if get_gpu_memory_guard() is None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    gc.collect()

    return (
        net_id,
        layers_params,
        train_loss,
        test_loss,
        train_acc,
        test_acc,
        val_metrics,
        test_metrics,
        arc_aslora_scores,
        arc_aslora_diagnostics,
    )


def compute_fed_avg_weights(nets_this_round, cli_train, n_parties, use_mk_correction=False, d_total_all=None):
    """
    FedAvg-style weights over selected clients.
    Standard: w_k = |D_k| / sum_{j in C_t} |D_j|
    Partial-participation (paper Eq.3): w_k = (M/K) * |D_k| / |D|, where |D| is total train size over all M clients.
    """
    client_ids = list(nets_this_round.keys())
    if use_mk_correction:
        m_clients = int(n_parties)
        k_selected = len(client_ids)
        d_total = float(
            d_total_all if d_total_all is not None
            else sum(len(cli_train[i]) for i in range(n_parties))
        )
        if d_total <= 0 or k_selected == 0:
            raise ValueError("Invalid D_total or empty client selection for M/K aggregation.")
        return [
            (m_clients / k_selected) * (len(cli_train[cid]) / d_total)
            for cid in client_ids
        ]
    total_selected = sum(len(cli_train[cid]) for cid in client_ids)
    return [len(cli_train[cid]) / total_selected for cid in client_ids]


def next_cyclic_client_batch(loaders, iterators, client_id):
    client_id = int(client_id)
    iterator = iterators.get(client_id)
    if iterator is None:
        iterator = iter(loaders[client_id])
        iterators[client_id] = iterator
    try:
        return next(iterator)
    except StopIteration:
        iterator = iter(loaders[client_id])
        iterators[client_id] = iterator
        try:
            return next(iterator)
        except StopIteration as exc:
            raise RuntimeError(
                f"Client {client_id} probe dataset is empty"
            ) from exc


def next_cyclic_client_batches(
    loaders,
    iterators,
    client_id,
    batch_count,
):
    if int(batch_count) < 1:
        raise ValueError("batch_count must be positive")
    return [
        next_cyclic_client_batch(loaders, iterators, client_id)
        for _ in range(int(batch_count))
    ]


def local_train_net(
        nets,
        args,
        cli_train,
        cli_vali,
        cli_test,
        frozen_layer,
        not_com_layer,
        round_idx,
        total_rounds,
        shared_source_model=None,
        as_lora_active_layers=None,
        gi_cmfv_active_layers=None,
):
    devices = _resolve_devices(args, len(nets))
    max_workers = args.max_workers or len(nets)
    log_lock = threading.Lock()
    results = {}
    gpu_semaphores = _gpu_semaphores(devices, args.per_gpu_max_parallel)
    gpu_guard = get_gpu_memory_guard()
    unique_devices = list(dict.fromkeys(devices))
    if gpu_guard is not None:
        gpu_guard.enter_compute(unique_devices, logger=logger)
    # 默认展示各客户端 tqdm（desc 中带轮次|客户端）；仅 --no_client_tqdm 时关闭
    no_client_tqdm = bool(args.no_client_tqdm)

    if max_workers > 1:
        limits = _parse_per_gpu_limits(args.per_gpu_max_parallel, devices)
        if limits is None:
            lim_desc = "不限制（0）"
        else:
            lim_desc = ", ".join(f"{d}={limits[d]}" for d in limits)
        logger.info(
            "并行训练: max_workers=%s；同卡并发上限=%s。"
            "说明: gpustat/nvidia-smi 按进程汇总显存，同一 Python 进程只占一行，与是否多线程并行无关；"
            "请看 GPU-Util 百分比或显存总占用是否随训练上升。"
            " 分词: 多线程共用同一 HuggingFace Tokenizer 时批量分词已在 trainer 内加锁串行，"
            "各客户端会先后出现「分词编码…完成」属正常。"
            " 若分词完成后仍卡在训练步，可把 --per_gpu_max_parallel 设为 1 对同卡 GPU 计算串行。"
            " 各客户端 tqdm=%s（多行可能交错，不需要时加 --no_client_tqdm）。",
            max_workers,
            lim_desc,
            "关" if no_client_tqdm else "开(默认)",
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, (net_id, net) in enumerate(nets.items()):
            dev = devices[i % len(devices)]
            futures.append(
                executor.submit(
                    train_single_client, net_id, net, args,
                    cli_train[net_id], cli_vali[net_id], cli_test[net_id],
                    frozen_layer, not_com_layer, dev, log_lock,
                    round_idx, total_rounds,
                    gpu_semaphores, no_client_tqdm, shared_source_model,
                    as_lora_active_layers,
                    gi_cmfv_active_layers,
                )
            )
        for future in concurrent.futures.as_completed(futures):
            (
                net_id,
                layers_params,
                train_loss,
                test_loss,
                train_acc,
                test_acc,
                val_metrics,
                test_metrics,
                arc_aslora_scores,
                arc_aslora_diagnostics,
            ) = future.result()
            results[net_id] = (
                layers_params,
                train_loss,
                test_loss,
                train_acc,
                test_acc,
                val_metrics,
                test_metrics,
                arc_aslora_scores,
                arc_aslora_diagnostics,
            )
            emit_progress(
                f"[客户端进度] {args.jl_method} | 轮次 {round_idx + 1}/{total_rounds} | "
                f"已完成 {len(results)}/{len(nets)} 个训练客户端 | 刚完成 client={net_id}",
                logger=logger,
            )

    # All client models are back on CPU here. Clear each device's allocator
    # cache once, after same-GPU workers have finished, while preserving any
    # live tensors held by the optional GPU memory guard.
    gc.collect()
    if torch.cuda.is_available():
        for device_id in unique_devices:
            if not str(device_id).startswith("cuda"):
                continue
            with torch.cuda.device(device_id):
                persistent_reservation = bool(
                    gpu_guard is not None
                    and gpu_guard.persistent_reservation
                )
                if not persistent_reservation:
                    torch.cuda.empty_cache()
                allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
            logger.info(
                "CUDA cache cleanup after client round: device=%s, "
                "allocated=%.2f GB, reserved=%.2f GB, persistent=%s",
                device_id,
                allocated_gb,
                reserved_gb,
                persistent_reservation,
            )

    net_params = [results[net_id][0] for net_id in nets.keys()]
    cli_train_loss = [results[net_id][1] for net_id in nets.keys()]
    cli_test_loss = [results[net_id][2] for net_id in nets.keys()]
    cli_train_acc = [results[net_id][3] for net_id in nets.keys()]
    cli_test_acc = [results[net_id][4] for net_id in nets.keys()]
    client_metrics = {
        net_id: {
            "train_loss": results[net_id][1],
            "test_loss": results[net_id][2],
            "train_acc": results[net_id][3],
            "test_acc": results[net_id][4],
            "val_threshold": results[net_id][5]["threshold"],
            "val_macro_f1": results[net_id][5]["macro_f1"],
            "test_macro_f1": results[net_id][6]["macro_f1"],
            "test_balanced_acc": results[net_id][6]["balanced_acc"],
            "test_macro_precision": results[net_id][6]["macro_precision"],
            "test_macro_recall": results[net_id][6]["macro_recall"],
            "test_macro_auc": results[net_id][6]["macro_auc"],
            "test_precision_cls1": results[net_id][6]["precision_cls1"],
            "test_recall_cls1": results[net_id][6]["recall_cls1"],
            "test_f1_cls1": results[net_id][6]["f1_cls1"],
            "test_evaluated": False,
        }
        for net_id in nets.keys()
    }
    arc_aslora_client_scores = {
        int(net_id): results[net_id][7]
        for net_id in nets.keys()
        if results[net_id][7] is not None
    }
    arc_aslora_client_diagnostics = {
        int(net_id): results[net_id][8]
        for net_id in nets.keys()
        if results[net_id][8] is not None
    }

    fed_train_loss = np.mean(cli_train_loss)
    fed_test_loss = np.mean(cli_test_loss)
    fed_train_acc = np.mean(cli_train_acc)
    fed_test_acc = np.mean(cli_test_acc)

    base_result = (
        net_params,
        fed_train_loss,
        fed_test_loss,
        fed_train_acc,
        fed_test_acc,
        client_metrics,
    )
    if str(getattr(args, "factor_selection_mode", "")) in {
        "arc_aslora",
        "gi_cmfv",
    }:
        return base_result + (
            arc_aslora_client_scores,
            arc_aslora_client_diagnostics,
        )
    return base_result


def init_nets(out_dim, n_parties,rank,model, dropout_policy="model_default", attention="eager"):
    nets = {net_i: None for net_i in range(n_parties)}

    for net_i in range(n_parties):
        if model=='roberta':
            net, loading_info = RobertaForSequenceClassification.from_pretrained(
                model_file, num_labels=out_dim, output_loading_info=True,
                attn_implementation=attention)
            net._jl_loading_info = loading_info
            if attention == "sdpa" and any(type(layer.attention.self).__name__ != "RobertaSdpaSelfAttention"
                                           for layer in net.roberta.encoder.layer):
                raise RuntimeError("Requested SDPA, but the loaded RoBERTa uses a different attention backend")
        elif model =='gpt':
            print(model)
            net = GPT2ForSequenceClassification.from_pretrained(model_file, num_labels=out_dim)
        # layer_params = net.roberta.encoder.layer[0].attention.self.named_parameters()
        # for name, param in layer_params:
        #
        #     print(param)
        if (if_lora):
            filter_names = [i for i in range(0, 12)]
            add_lora(net, filter_names, rank,model)
            nets[net_i] = net
        else:
            if finetune == "full":
                for p in net.parameters():
                    p.requires_grad = True
                nets[net_i] = net
            else:
                if finetune == "adapter":
                    config_dict ={
                        "delta_type": "adapter",
                        # "learning_rate": 1e-3,
                        # "num_train_epochs": 3,
                        "unfrozen_modules": [
                            "deltas",
                            "layer_norm",
                            "final_layer_norm",
                            "classifier",
                        ],
                        "bottleneck_dim": 16,
                    }
                elif (finetune == "bitfit"):
                    config_dict = {
                        # "num_train_epochs": 3,
                        "delta_type": "bitfit",
                        # "learning_rate": 3e-4,
                        "unfrozen_modules": [
                            "classifier",
                            "deltas"
                        ],
                    }
                elif(finetune == "prefix_tuning"):
                    config_dict = {
                        "delta_type": "prefix",
                        # "learning_rate": 1e-3,
                        "unfrozen_modules": [
                            "deltas",
                            "classifier",
                        ],
                        "prefix_token_num": 64
                    }
                else:
                    raise ValueError(f"Unsupported finetune mode without LoRA: {finetune!r}")
                from opendelta import AutoDeltaConfig, AutoDeltaModel
                delta_config = AutoDeltaConfig.from_dict(config_dict)
                delta_model = AutoDeltaModel.from_config(delta_config,backbone_model=net)
                delta_model.freeze_module(set_state_dict=True)
                nets[net_i] = delta_model
                delta_model.log()

        count = configure_dropout(nets[net_i], dropout_policy)
        emit_progress(f"[Model] attention={attention} | dropout={dropout_policy} | modules={count} | model_index={net_i}")

    return nets


# 此方法只针对attention的lkey、value层
def generate_frozen_layer(layer, matrix,model):
    all_layer_num = [i for i in range(12)]
    frozen_layer_num = [i for i in all_layer_num if i not in layer]
    # frozen_layer是冻结的层名，not_com_layer是联邦时不进行交换的层名，区别是lora在本地训练A、B都参与，但是交换时仅一个参与
    frozen_layer = []
    not_com_layer = []
    not_com_layer_mark = []
    # 如果frozen_layer_num不为空，则有些层需要全部冻结
    if (len(frozen_layer_num) != 0):
        if model =='roberta':
            frozen_layer = ['roberta.encoder.layer.' + str(i) + '.attention.self.' + postfix + factor for i in
                            frozen_layer_num
                            for postfix in ['key.', 'value.'] for factor in ['lora_a', 'lora_b']]
        elif model=='gpt':
            frozen_layer = ['transformer.h.' + str(i) + '.attn.c_attn.' + postfix + factor for i in frozen_layer_num
                             for postfix in ['k_proj.', 'v_proj.'] for factor in ['lora_a', 'lora_b']]
        not_com_layer = frozen_layer.copy()
        not_com_layer_mark = [str(i) + postfix + factor for postfix in ['.key.', '.value.'] for i in frozen_layer_num for
                             factor in ['0', '1']]
    # 如果matrix中存在0或者1，则需要根据matrix的值来确定A还是B需要冻结
    if(0 in matrix.values() or 1 in matrix.values()):
        for i in layer:
            if matrix[i] == 0:
                if model=='roberta':
                    not_com_layer.append('roberta.encoder.layer.' + str(i) + '.attention.self.key.lora_b')
                    not_com_layer.append('roberta.encoder.layer.' + str(i) + '.attention.self.value.lora_b')
                elif model=='gpt':
                    not_com_layer.append('transformer.h.' + str(i) + '.attn.c_attn.k_proj.lora_b')
                    not_com_layer.append('transformer.h.' + str(i) + '.attn.c_attn.v_proj.lora_b')
                not_com_layer_mark.append(str(i) +'.key.1')
                not_com_layer_mark.append(str(i) +'.value.1')

            elif matrix[i] == 1:
                if model=='roberta':
                    not_com_layer.append('roberta.encoder.layer.' + str(i) + '.attention.self.key.lora_a')
                    not_com_layer.append('roberta.encoder.layer.' + str(i) + '.attention.self.value.lora_a')
                elif model=='gpt':
                    not_com_layer.append('transformer.h.' + str(i) + '.attn.c_attn.k_proj.lora_a')
                    not_com_layer.append('transformer.h.' + str(i) + '.attn.c_attn.v_proj.lora_a')
                not_com_layer_mark.append(str(i) + '.key.0')
                not_com_layer_mark.append(str(i) + '.value.0')
            else:
                continue
    return frozen_layer, not_com_layer_mark,not_com_layer
def get_gpu_info(pid):
    """
    使用 nvidia-smi 获取指定 PID 的 GPU 使用情况。
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                parts = line.split(',')
                if len(parts) == 2 and int(parts[0]) == pid:
                    used_memory = int(parts[1])
                    print(used_memory)
                    return used_memory
    except Exception as e:
        print(f"Error: {e}")
    return None, None
# def record_resource_usage(pid,cpu_usages, memory_usages,gpu_usages,interval, stop_event):
#
#     while not stop_event.is_set():
#         used_memory = get_gpu_info(pid)
#         if isinstance(used_memory, int) :
#             gpu_usages.append(used_memory)
#         if used_memory is not None:
#             print(f"PID: {pid}, Used Memory: {used_memory} MiB")
#         else:
#             print(f"No GPU usage data found for PID: {pid}")
#         current_cpu_usage = process.cpu_percent(interval=1)
#         current_memory_usage = process.memory_info().rss
#         cpu_usages.append(current_cpu_usage)
#         memory_usages.append(current_memory_usage / (1024 * 1024))
#         time.sleep(interval)
def compute_sparsity(grad, epsilon=1e-5):
    """
    计算梯度的稀疏性
    :param grad: 梯度张量
    :param epsilon: 判断梯度接近零的阈值
    :return: 稀疏性比例
    """

    close_to_zero = (torch.abs(grad) < epsilon).float()
    sparsity = torch.mean(close_to_zero).item()

    return sparsity


def _safe_filename(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


COORDINATE_SCHEDULES = {
    "bba": [1, 1, 0],
    "bbb": [1, 1, 1],
    "bbba": [1, 1, 1, 0],
    "baba": [1, 0, 1, 0],
    "bbbaaa": [1, 1, 1, 0, 0, 0],
    "b10a1": [1] * 10 + [0],
    "b20a1": [1] * 20 + [0],
    "b10_bba": [],  # B×10 warmup, then B,B,A cycling
    "b10_b2a2": [],  # B×10 warmup, then B,B,A,A cycling
    "b20_b10a10": [],  # B×20 warmup, then B10A10 cycling
    "b10a10": [1] * 10 + [0] * 10,
    "b5a5": [1] * 5 + [0] * 5,
    "b20a20": [1] * 20 + [0] * 20,
    "bbaa": [1, 1, 0, 0],
    "trend_slope_switching": [],  # adaptive B/A via validation Macro F1 trend slope
    "none": [],  # FedAvg-Full: no LoRA coordinate descent
}

COORDINATE_SCHEDULE_LABELS = {
    "bba": "B,B,A",
    "bbb": "B,B,B",
    "bbba": "B,B,B,A",
    "baba": "B,A,B,A",
    "bbbaaa": "B,B,B,A,A,A",
    "b10a1": "B×10,A×1",
    "b20a1": "B×20,A×1",
    "b10_bba": "B×10, then B,B,A",
    "b10_b2a2": "B×10, then B,B,A,A",
    "b20_b10a10": "B×20, then B×10,A×10",
    "b10a10": "B×10,A×10",
    "b5a5": "B×5,A×5",
    "b20a20": "B×20,A×20",
    "bbaa": "B,B,A,A",
    "trend_slope_switching": "Trend-slope adaptive B/A (w=5,p=2,eps=0.001,m_min=5,m_max=20)",
    "none": "N/A (no LoRA CD)",
}

# Trend-slope adaptive factor switching (fixed hyperparameters per spec)
TREND_SLOPE_W = 5
TREND_SLOPE_P = 2
TREND_SLOPE_EPS = 0.001
TREND_SLOPE_M_MIN = 5
TREND_SLOPE_M_MAX = 20

# Active schedule list; set in __main__ from --coordinate_schedule.
_coordinate_schedule_list = COORDINATE_SCHEDULES["bba"]
_trend_slope_switching_active = False
_b10_warmup_schedule = None  # (warmup_rounds, tail_cycle) for b10_bba / b10_b2a2
B10_WARMUP_TAIL_CYCLES = {
    "b10_bba": (10, [1, 1, 0]),
    "b10_b2a2": (10, [1, 1, 0, 0]),
    "b20_b10a10": (20, [1] * 10 + [0] * 10),
}


def resolve_coordinate_schedule(name):
    key = str(name).lower().strip()
    if key not in COORDINATE_SCHEDULES:
        raise ValueError(
            f"Unknown coordinate_schedule={name!r}; "
            f"choose from {list(COORDINATE_SCHEDULES.keys())}"
        )
    return COORDINATE_SCHEDULES[key], COORDINATE_SCHEDULE_LABELS[key]


def _longest_same_factor_run(cycle):
    if not cycle:
        return 1
    longest_run = 1
    current_run = 1
    for idx in range(1, len(cycle)):
        if cycle[idx] == cycle[idx - 1]:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 1
    return longest_run


def set_active_coordinate_schedule(name):
    global _coordinate_schedule_list, _trend_slope_switching_active, _b10_warmup_schedule
    key = str(name).lower().strip()
    _coordinate_schedule_list, label = resolve_coordinate_schedule(name)
    _trend_slope_switching_active = key == "trend_slope_switching"
    _b10_warmup_schedule = B10_WARMUP_TAIL_CYCLES.get(key)
    return label


def get_coordinate_factor(round_idx, schedule=None):
    """
    RLCD-FedLoRA coordinate schedule: 1 = update/aggregate B, 0 = A.
    Cycles over the list set by --coordinate_schedule (default B,B,A).
    """
    if _b10_warmup_schedule is not None:
        warmup_rounds, tail_cycle = _b10_warmup_schedule
        if round_idx < warmup_rounds:
            return 1
        tail_idx = round_idx - warmup_rounds
        return tail_cycle[tail_idx % len(tail_cycle)]
    sched = schedule if schedule is not None else _coordinate_schedule_list
    return sched[round_idx % len(sched)]


def factor_name(factor_value):
    return "B" if factor_value == 1 else "A"


def resolve_server_validation_dataset(sst_dataset, cli_vali, logger):
    """Use dedicated server_vali when present; otherwise pooled client validation."""
    if "server_vali" in sst_dataset:
        cen_vali = sst_dataset["server_vali"]
        logger.info(
            f"Using dedicated server_vali (n={len(cen_vali)}) for global validation / RL reward."
        )
        if "server_test" in sst_dataset:
            logger.info(f"Held-out server_test available (n={len(sst_dataset['server_test'])}).")
        return cen_vali
    cen_vali = concatenate_datasets(cli_vali)
    logger.info(
        f"Using pooled client vali D_val for server eval / RL (n={len(cen_vali)})."
    )
    return cen_vali


def get_max_update_delay(schedule=None):
    """Paper: tau = gamma + 1 is the max delay between B/A factor updates (not layer selection)."""
    if _b10_warmup_schedule is not None:
        warmup_rounds, tail_cycle = _b10_warmup_schedule
        return int(max(warmup_rounds + 1, _longest_same_factor_run(tail_cycle) + 1))
    sched = schedule if schedule is not None else _coordinate_schedule_list
    if not sched:
        if _trend_slope_switching_active:
            return int(max(TREND_SLOPE_M_MAX, TREND_SLOPE_M_MIN + TREND_SLOPE_W + TREND_SLOPE_P))
        return 1
    longest_run = 1
    current_run = 1
    for idx in range(1, len(sched)):
        if sched[idx] == sched[idx - 1]:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 1
    return int(longest_run + 1)


def compute_trend_slope(f1_values):
    """Linear regression slope k over window indices 1..w for the last w Macro F1 values."""
    w = len(f1_values)
    if w < 2:
        return None
    i_vals = np.arange(1, w + 1, dtype=np.float64)
    f_vals = np.asarray(f1_values, dtype=np.float64)
    i_bar = float(np.mean(i_vals))
    f_bar = float(np.mean(f_vals))
    denom = float(np.sum((i_vals - i_bar) ** 2))
    if denom == 0.0:
        return 0.0
    k = float(np.sum((i_vals - i_bar) * (f_vals - f_bar)) / denom)
    return k


def init_trend_slope_switching_state():
    return {
        "current_factor": "B",
        "factor_rounds": 0,
        "factor_f1_history": [],
        "saturation_counter": 0,
    }


def factor_name_to_coordinate_value(factor_name):
    return 1 if str(factor_name).upper() == "B" else 0


def set_trainable_lora_factor(model, current_factor):
    """
    Only the active LoRA factor (A or B) remains trainable; backbone/classifier stay frozen.
    Per-layer masking for RLCD is still applied via generate_frozen_layer / freeze_layer.
    """
    target_suffix = "lora_b" if str(current_factor).upper() == "B" else "lora_a"
    freeze_suffix = "lora_a" if target_suffix == "lora_b" else "lora_b"
    unfrozen(model)
    for name, param in model.named_parameters():
        suffix = name.split(".")[-1]
        if suffix in {"lora_a", "lora_b"} and suffix == freeze_suffix:
            param.requires_grad = False
    return model


def should_switch_factor(factor_f1_history, factor_rounds, saturation_counter):
    """
    Switch when:
      - current factor has run m_max rounds (forced), or
      - m_min met, enough F1 history, and saturation_counter >= p (trend saturation).
    """
    if factor_rounds < TREND_SLOPE_M_MIN:
        return False, None
    if factor_rounds >= TREND_SLOPE_M_MAX:
        return True, "max_rounds"
    if len(factor_f1_history) >= TREND_SLOPE_W and int(saturation_counter) >= TREND_SLOPE_P:
        return True, "saturation"
    return False, None


def _apply_factor_switch(state):
    state["current_factor"] = "A" if state["current_factor"] == "B" else "B"
    state["factor_rounds"] = 0
    state["factor_f1_history"] = []
    state["saturation_counter"] = 0


def update_trend_slope_switching_state(state, macro_f1):
    """
    Append validation Macro F1, update saturation counter, optionally flip B<->A.
    Returns (switch_event, trend_slope_or_none, factor_used_this_round, saturation_after, switch_reason).
    """
    factor_used = state["current_factor"]
    state["factor_rounds"] += 1
    state["factor_f1_history"].append(float(macro_f1))
    rounds_this_phase = state["factor_rounds"]

    switch_event = 0
    switch_reason = None
    trend_slope = None
    saturation_after = state["saturation_counter"]

    if state["factor_rounds"] >= TREND_SLOPE_M_MIN and len(state["factor_f1_history"]) >= TREND_SLOPE_W:
        recent = state["factor_f1_history"][-TREND_SLOPE_W:]
        trend_slope = compute_trend_slope(recent)
        if trend_slope <= TREND_SLOPE_EPS:
            state["saturation_counter"] += 1
        else:
            state["saturation_counter"] = 0
        saturation_after = state["saturation_counter"]

    do_switch, switch_reason = should_switch_factor(
        state["factor_f1_history"], state["factor_rounds"], state["saturation_counter"]
    )
    if do_switch:
        _apply_factor_switch(state)
        switch_event = 1

    return switch_event, trend_slope, factor_used, saturation_after, switch_reason, rounds_this_phase


def build_round_matrix(selected_layers, factor_value, n_layers=12):
    matrix = {i: None for i in range(n_layers)}
    for layer_idx in selected_layers:
        matrix[int(layer_idx)] = int(factor_value)
    return matrix


def build_round_matrix_from_factor_map(selected_layers, factor_map, n_layers=12):
    matrix = {i: None for i in range(n_layers)}
    for layer_idx in selected_layers:
        matrix[int(layer_idx)] = factor_name_to_coordinate_value(factor_map[int(layer_idx)])
    return matrix


def lora_factor_param_names_for_layer(layer_idx, factor_name, model_name):
    suffix = "lora_b" if str(factor_name).upper() == "B" else "lora_a"
    if model_name == "roberta":
        return [
            f"roberta.encoder.layer.{layer_idx}.attention.self.key.{suffix}",
            f"roberta.encoder.layer.{layer_idx}.attention.self.value.{suffix}",
        ]
    if model_name == "gpt":
        return [
            f"transformer.h.{layer_idx}.attn.c_attn.k_proj.{suffix}",
            f"transformer.h.{layer_idx}.attn.c_attn.v_proj.{suffix}",
        ]
    raise ValueError(f"Unsupported model for LoRA factor names: {model_name}")


def expected_upload_param_names_from_factor_map(factor_map, model_name):
    expected = set()
    for layer_idx, factor in factor_map.items():
        expected.update(lora_factor_param_names_for_layer(int(layer_idx), factor, model_name))
    return expected


def prepare_round_factor_masking(selected_layers, round_matrix, model, rl_active=False):
    frozen_layer, not_com_layer_mark, not_com_layer = generate_frozen_layer(
        selected_layers, round_matrix, model
    )
    if rl_active:
        for layer_idx in selected_layers:
            factor_value = round_matrix.get(int(layer_idx))
            if factor_value is None:
                continue
            if factor_value == 1:
                frozen_layer.extend(
                    lora_factor_param_names_for_layer(int(layer_idx), "A", model)
                )
            else:
                frozen_layer.extend(
                    lora_factor_param_names_for_layer(int(layer_idx), "B", model)
                )
    return frozen_layer, not_com_layer_mark, not_com_layer


def _layer_score_value(layer_scores, layer_id):
    if isinstance(layer_scores, dict):
        return float(layer_scores[int(layer_id)])
    return float(layer_scores[int(layer_id)])


def top_k_layers(layer_scores, k, all_layer_ids=None):
    if all_layer_ids is None:
        if isinstance(layer_scores, dict):
            all_layer_ids = sorted(int(x) for x in layer_scores.keys())
        else:
            all_layer_ids = list(range(len(layer_scores)))
    ordered = sorted(
        [int(layer_id) for layer_id in all_layer_ids],
        key=lambda layer_id: (-_layer_score_value(layer_scores, layer_id), int(layer_id)),
    )
    return ordered[: int(k)]


def select_layers_fixed_20_80(
        all_layer_ids,
        layer_scores,
        budget,
        rng,
        layer_random_prob=0.20,
):
    all_layer_ids = [int(layer_id) for layer_id in all_layer_ids]
    budget = int(budget)
    if not 1 <= budget <= len(all_layer_ids):
        raise ValueError(
            f"Invalid layer budget {budget}; expected 1 <= budget <= {len(all_layer_ids)}"
        )

    draw = rng.random()
    if draw < float(layer_random_prob):
        selected_layers = rng.sample(list(all_layer_ids), budget)
        reason = "random_20pct"
    else:
        selected_layers = top_k_layers(
            layer_scores=layer_scores,
            k=budget,
            all_layer_ids=all_layer_ids,
        )
        reason = "score_greedy" if float(layer_random_prob) == 0 else "score_80pct"
    return [int(layer_id) for layer_id in selected_layers], reason, draw


def init_factor_scores_and_counts(total_layers):
    factor_scores = {
        int(layer_id): {"A": 0.0, "B": 0.0}
        for layer_id in range(int(total_layers))
    }
    factor_select_counts = {
        int(layer_id): {"A": 0, "B": 0}
        for layer_id in range(int(total_layers))
    }
    return factor_scores, factor_select_counts


def resolve_warmup_layer_selection(round_idx, warmup_rounds, total_layers):
    if int(round_idx) < int(warmup_rounds):
        selected_layers = list(range(int(total_layers)))
        return {
            "warmup_active": True,
            "selected_budget": len(selected_layers),
            "selected_layers": selected_layers,
            "layer_selection_reason": "warmup_all_layers",
            "layer_random_draw": None,
        }
    return {
        "warmup_active": False,
        "selected_budget": None,
        "selected_layers": None,
        "layer_selection_reason": None,
        "layer_random_draw": None,
    }


def resolve_all_layer_selection(total_layers):
    selected_layers = list(range(int(total_layers)))
    return {
        "selected_budget": len(selected_layers),
        "selected_layers": selected_layers,
        "layer_selection_reason": "all_layers_baseline",
        "layer_random_draw": None,
    }


FACTOR_SELECTOR_SEED_OFFSET = 7919
ADAPTIVE_CD_LAYER_METHODS = frozenset({"rl", "all", "fedfmsl", "spdcfl", "random_g"})
SCORE_HISTORY_FACTOR_MODES = frozenset({"adaptive_score", "product_delta"})


class GlobalAdaptiveFactorSelector:
    def __init__(
            self,
            epsilon: float = 0.20,
            beta: float = 0.20,
            max_staleness: int = 5,
            reward_normalization: str = "per_layer",
            score_clip: Optional[float] = None,
            seed: int = 0,
    ):
        self.score_a: float = 0.0
        self.score_b: float = 0.0
        self.count_a: int = 0
        self.count_b: int = 0
        self.last_selected_a: Optional[int] = None
        self.last_selected_b: Optional[int] = None
        self.epsilon: float = float(epsilon)
        self.beta: float = float(beta)
        self.max_staleness: int = int(max_staleness)
        self.reward_normalization: str = str(reward_normalization)
        self.score_clip: Optional[float] = None if score_clip is None else float(score_clip)
        self.random_state = random.Random(int(seed))

    def state_dict(self) -> dict:
        return {
            "score_a": float(self.score_a),
            "score_b": float(self.score_b),
            "count_a": int(self.count_a),
            "count_b": int(self.count_b),
            "last_selected_a": self.last_selected_a,
            "last_selected_b": self.last_selected_b,
            "epsilon": float(self.epsilon),
            "beta": float(self.beta),
            "max_staleness": int(self.max_staleness),
            "reward_normalization": str(self.reward_normalization),
            "score_clip": self.score_clip,
            "random_state": self.random_state.getstate(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.score_a = float(state.get("score_a", 0.0))
        self.score_b = float(state.get("score_b", 0.0))
        self.count_a = int(state.get("count_a", 0))
        self.count_b = int(state.get("count_b", 0))
        self.last_selected_a = state.get("last_selected_a")
        self.last_selected_b = state.get("last_selected_b")
        self.epsilon = float(state.get("epsilon", self.epsilon))
        self.beta = float(state.get("beta", self.beta))
        self.max_staleness = int(state.get("max_staleness", self.max_staleness))
        self.reward_normalization = str(state.get("reward_normalization", self.reward_normalization))
        self.score_clip = state.get("score_clip", self.score_clip)
        if state.get("random_state") is not None:
            self.random_state.setstate(state["random_state"])

    def _staleness(self, round_idx: int, last_selected: Optional[int]) -> float:
        if last_selected is None:
            return float("inf")
        return float(round_idx - last_selected)

    def _random_factor(self, rng: random.Random) -> str:
        return "A" if rng.random() < 0.5 else "B"

    def select_factor(
            self,
            round_idx: int,
            rng: Optional[random.Random] = None,
    ) -> Tuple[str, str]:
        rng = rng or self.random_state

        if self.count_a == 0 and self.count_b == 0:
            return "B", "forced_unseen_both_choose_b"
        if self.count_a == 0:
            return "A", "forced_unseen_a"
        if self.count_b == 0:
            return "B", "forced_unseen_b"

        stale_a = self._staleness(round_idx, self.last_selected_a)
        stale_b = self._staleness(round_idx, self.last_selected_b)
        force_a = stale_a >= self.max_staleness
        force_b = stale_b >= self.max_staleness
        if force_a or force_b:
            if force_a and not force_b:
                return "A", "forced_a_staleness"
            if force_b and not force_a:
                return "B", "forced_b_staleness"
            if stale_a > stale_b:
                return "A", "forced_a_staleness"
            return "B", "forced_b_staleness"

        draw = rng.random()
        if draw < self.epsilon:
            factor = self._random_factor(rng)
            return factor, f"epsilon_random_{factor.lower()}"

        if self.score_a > self.score_b:
            return "A", "score_greedy_a"
        if self.score_b > self.score_a:
            return "B", "score_greedy_b"
        if stale_a > stale_b:
            return "A", "tie_break_a"
        return "B", "tie_break_b"

    def update(
            self,
            factor: str,
            reward: float,
            num_selected_layers: int,
            round_idx: int,
    ) -> dict:
        factor = str(factor).upper()
        if factor not in {"A", "B"}:
            raise ValueError(f"Invalid factor {factor!r}; expected 'A' or 'B'.")
        if self.reward_normalization != "per_layer":
            raise ValueError(
                f"Unsupported factor_reward_normalization={self.reward_normalization!r}"
            )

        factor_utility = float(reward) / max(int(num_selected_layers), 1)
        if self.score_clip is not None:
            factor_utility = float(np.clip(factor_utility, -self.score_clip, self.score_clip))

        score_a_before = float(self.score_a)
        score_b_before = float(self.score_b)
        if factor == "A":
            self.score_a = (1.0 - self.beta) * self.score_a + self.beta * factor_utility
            self.count_a += 1
            self.last_selected_a = int(round_idx)
        else:
            self.score_b = (1.0 - self.beta) * self.score_b + self.beta * factor_utility
            self.count_b += 1
            self.last_selected_b = int(round_idx)

        total_count = self.count_a + self.count_b
        return {
            "factor": factor,
            "factor_utility": float(factor_utility),
            "score_a_before": score_a_before,
            "score_b_before": score_b_before,
            "score_a_after": float(self.score_a),
            "score_b_after": float(self.score_b),
            "count_a": int(self.count_a),
            "count_b": int(self.count_b),
            "last_selected_a": self.last_selected_a,
            "last_selected_b": self.last_selected_b,
            "factor_a_ratio_so_far": float(self.count_a / total_count) if total_count else 0.0,
            "factor_b_ratio_so_far": float(self.count_b / total_count) if total_count else 0.0,
        }


def build_global_factor_map(selected_layers, active_factor):
    factor = str(active_factor).upper()
    if factor not in {"A", "B"}:
        raise ValueError(f"Invalid factor {active_factor!r}; expected 'A' or 'B'.")
    return {int(layer_id): factor for layer_id in selected_layers}


def select_factor_for_layer(
        layer_id,
        factor_scores,
        factor_select_counts,
        rng,
        factor_random_prob=0.20,
):
    layer_id = int(layer_id)
    count_a = int(factor_select_counts[layer_id]["A"])
    count_b = int(factor_select_counts[layer_id]["B"])

    if count_a == 0 and count_b == 0:
        return {"factor": "B", "reason": "forced_B_both_unseen", "random_draw": None}
    if count_a == 0:
        return {"factor": "A", "reason": "forced_A_unseen", "random_draw": None}
    if count_b == 0:
        return {"factor": "B", "reason": "forced_B_unseen", "random_draw": None}

    draw = rng.random()
    if draw < float(factor_random_prob):
        factor = rng.choice(["A", "B"])
        return {"factor": factor, "reason": "random_20pct", "random_draw": draw}

    score_a = float(factor_scores[layer_id]["A"])
    score_b = float(factor_scores[layer_id]["B"])
    if score_a > score_b:
        factor = "A"
        reason = "score_80pct"
    elif score_b > score_a:
        factor = "B"
        reason = "score_80pct"
    else:
        factor = "B" if float(factor_random_prob) == 0 else rng.choice(["A", "B"])
        reason = "score_tie_B" if float(factor_random_prob) == 0 else "score_tie_random"
    return {"factor": factor, "reason": reason, "random_draw": draw}


def build_factor_map_for_selected_layers(
        selected_layers,
        factor_scores,
        factor_select_counts,
        rng,
        factor_random_prob=0.20,
):
    factor_map = {}
    factor_decision_metadata = {}
    for layer_id in selected_layers:
        result = select_factor_for_layer(
            layer_id=layer_id,
            factor_scores=factor_scores,
            factor_select_counts=factor_select_counts,
            rng=rng,
            factor_random_prob=factor_random_prob,
        )
        layer_id = int(layer_id)
        factor_map[layer_id] = result["factor"]
        factor_decision_metadata[layer_id] = result
    return factor_map, factor_decision_metadata


def build_product_delta_factor_map(
        selected_layers,
        round_idx,
        warmup_rounds,
        factor_scores,
        factor_select_counts,
        rng,
        factor_random_prob=0.0,
):
    selected_layers = [int(layer_id) for layer_id in selected_layers]
    if int(round_idx) < int(warmup_rounds):
        factor = "B" if int(round_idx) % 2 == 0 else "A"
        factor_map = build_global_factor_map(selected_layers, factor)
        metadata = {
            layer_id: {
                "factor": factor,
                "reason": "product_warmup_alternation",
                "random_draw": None,
            }
            for layer_id in selected_layers
        }
        return factor_map, metadata

    return build_factor_map_for_selected_layers(
        selected_layers=selected_layers,
        factor_scores=factor_scores,
        factor_select_counts=factor_select_counts,
        rng=rng,
        factor_random_prob=factor_random_prob,
    )


def validate_factor_upload_consistency(net_params, factor_map, model_name):
    if not factor_map:
        return
    expected_names = expected_upload_param_names_from_factor_map(factor_map, model_name)
    for client_idx, client_params in enumerate(net_params):
        client_names = set(client_params.keys())
        unexpected = sorted(client_names - expected_names)
        missing = sorted(expected_names - client_names)
        if unexpected or missing:
            raise ValueError(
                "Inconsistent factor uploads for client "
                f"{client_idx}: unexpected={unexpected}, missing={missing}"
            )


def aggregate_uploaded_params(net_params, fed_avg_freqs):
    if len(net_params) != len(fed_avg_freqs):
        raise ValueError(
            f"Mismatched aggregation inputs: {len(net_params)} client payloads vs "
            f"{len(fed_avg_freqs)} weights"
        )

    aggregated_params = {}
    aggregated_grad = {}
    param_calculate = 0
    param_upload = 0
    for net_id, net in enumerate(net_params):
        for name, param in net.items():
            weight = fed_avg_freqs[net_id]
            if name not in aggregated_params:
                aggregated_params[name] = param.clone().detach() * weight
                aggregated_grad[name] = (
                    param.grad.clone().detach() * weight if param.grad is not None else None
                )
            else:
                aggregated_params[name] += param.clone().detach() * weight
                if param.grad is not None:
                    if aggregated_grad[name] is None:
                        aggregated_grad[name] = param.grad.clone().detach() * weight
                    else:
                        aggregated_grad[name] += param.grad.clone().detach() * weight
            param_calculate += param.numel()
            param_upload += param.numel()
    return aggregated_params, aggregated_grad, param_calculate, param_upload


def update_factor_select_counts(factor_select_counts, factor_map, aggregated_params, model_name):
    aggregated_keys = set(aggregated_params.keys())
    for layer_id, factor in factor_map.items():
        expected = set(lora_factor_param_names_for_layer(int(layer_id), factor, model_name))
        missing = sorted(expected - aggregated_keys)
        if missing:
            raise ValueError(
                f"Aggregation missing expected active factor parameters for layer {layer_id} "
                f"factor {factor}: {missing}"
            )
    for layer_id, factor in factor_map.items():
        factor_select_counts[int(layer_id)][str(factor).upper()] += 1


def compute_selected_factor_delta_norms(prev_lora_state, curr_model, factor_map, model_name):
    if prev_lora_state is None or not factor_map:
        return 0.0, {}

    curr_lora_state = clone_lora_factor_state(curr_model)
    per_layer_delta = {}
    total_q = 0.0
    for layer_id, factor in factor_map.items():
        layer_id = int(layer_id)
        layer_total = 0.0
        for name in lora_factor_param_names_for_layer(layer_id, factor, model_name):
            before = prev_lora_state.get(name)
            after = curr_lora_state.get(name)
            if before is None or after is None:
                continue
            layer_total += float(torch.norm(after.float() - before.float(), p='fro').item())
        per_layer_delta[layer_id] = layer_total
        total_q += layer_total
    return float(total_q), per_layer_delta


def compute_layer_factor_rms_delta(
        layer_id,
        selected_factor,
        global_state_before,
        global_state_after,
        model_name,
):
    squared_delta_sum = 0.0
    total_numel = 0
    parameter_names = lora_factor_param_names_for_layer(layer_id, selected_factor, model_name)
    if not parameter_names:
        raise RuntimeError(
            f"No LoRA parameters found for layer={layer_id}, factor={selected_factor}"
        )

    for parameter_name in parameter_names:
        if parameter_name not in global_state_before or parameter_name not in global_state_after:
            raise RuntimeError(
                f"Missing LoRA parameter snapshot for {parameter_name} "
                f"(layer={layer_id}, factor={selected_factor})"
            )
        before = global_state_before[parameter_name]
        after = global_state_after[parameter_name]
        if tuple(before.shape) != tuple(after.shape):
            raise RuntimeError(
                f"Shape mismatch for {parameter_name}: before={tuple(before.shape)}, "
                f"after={tuple(after.shape)}"
            )

        delta = (
            after.detach().to(dtype=torch.float64, device="cpu")
            - before.detach().to(dtype=torch.float64, device="cpu")
        )
        squared_delta_sum += torch.sum(delta * delta).item()
        total_numel += delta.numel()

    if total_numel <= 0:
        raise RuntimeError(
            f"Invalid total_numel for layer={layer_id}, factor={selected_factor}"
        )
    return math.sqrt(squared_delta_sum / total_numel)


def assign_rms_weighted_factor_credit(
        reward,
        selected_layers,
        factor_map,
        global_state_before,
        global_state_after,
        model_name,
        eps=1e-12,
):
    if not selected_layers:
        raise ValueError("selected_layers must not be empty when assigning credit")

    rms_deltas = {}
    for layer_id in selected_layers:
        factor = factor_map[layer_id]
        rms_deltas[layer_id] = compute_layer_factor_rms_delta(
            layer_id=layer_id,
            selected_factor=factor,
            global_state_before=global_state_before,
            global_state_after=global_state_after,
            model_name=model_name,
        )

    total_rms_delta = float(sum(rms_deltas.values()))
    if total_rms_delta <= eps:
        equal_credit = reward / len(selected_layers)
        credits = {layer_id: float(equal_credit) for layer_id in selected_layers}
        weights = {layer_id: float(1.0 / len(selected_layers)) for layer_id in selected_layers}
        fallback_used = True
    else:
        weights = {
            layer_id: float(rms_deltas[layer_id] / total_rms_delta)
            for layer_id in selected_layers
        }
        credits = {
            layer_id: float(reward * weights[layer_id])
            for layer_id in selected_layers
        }
        fallback_used = False

    return {
        "credits": credits,
        "weights": weights,
        "rms_deltas": rms_deltas,
        "total_rms_delta": total_rms_delta,
        "fallback_used": fallback_used,
        "credit_mode": "rms",
    }


def assign_equal_factor_credit(reward, selected_layers):
    if not selected_layers:
        raise ValueError("selected_layers must not be empty when assigning credit")

    equal_credit = float(reward) / len(selected_layers)
    credits = {layer_id: equal_credit for layer_id in selected_layers}
    weights = {layer_id: 1.0 / len(selected_layers) for layer_id in selected_layers}
    return {
        "credits": credits,
        "weights": weights,
        "rms_deltas": {layer_id: 0.0 for layer_id in selected_layers},
        "total_rms_delta": 0.0,
        "fallback_used": False,
        "credit_mode": "equal",
    }


def compute_effective_product_delta_norms(
        global_state_before,
        global_state_after,
        selected_layers=None,
):
    """Return per-layer ||Delta(A@B)|| over the layer's key/value LoRA modules."""
    selected = (
        None
        if selected_layers is None
        else {int(layer_id) for layer_id in selected_layers}
    )
    squared_norms = {}
    for a_name, before_a in global_state_before.items():
        if a_name.split(".")[-1] != "lora_a":
            continue
        layer_id = int(a_name.split(".")[-5])
        if selected is not None and layer_id not in selected:
            continue

        prefix = a_name.rsplit(".", 1)[0]
        b_name = f"{prefix}.lora_b"
        before_b = global_state_before.get(b_name)
        after_a = global_state_after.get(a_name)
        after_b = global_state_after.get(b_name)
        if before_b is None or after_a is None or after_b is None:
            raise RuntimeError(
                f"Incomplete LoRA A/B snapshots for effective product {prefix}"
            )

        before_product = _compose_lora_delta(
            before_a.to(dtype=torch.float32, device="cpu"),
            before_b.to(dtype=torch.float32, device="cpu"),
        )
        after_product = _compose_lora_delta(
            after_a.to(dtype=torch.float32, device="cpu"),
            after_b.to(dtype=torch.float32, device="cpu"),
        )
        if before_product is None or after_product is None:
            raise RuntimeError(
                f"Incompatible LoRA A/B shapes for effective product {prefix}"
            )

        product_delta = after_product - before_product
        squared_norms[layer_id] = squared_norms.get(layer_id, 0.0) + float(
            torch.sum(product_delta * product_delta).item()
        )

    if selected is not None:
        missing = sorted(selected - set(squared_norms))
        if missing:
            raise RuntimeError(
                f"No effective LoRA A@B products found for selected layers {missing}"
            )
    return {
        int(layer_id): math.sqrt(max(float(squared_norm), 0.0))
        for layer_id, squared_norm in squared_norms.items()
    }


def assign_product_delta_weighted_credit(
        reward,
        selected_layers,
        global_state_before,
        global_state_after,
        eps=1e-12,
):
    """Distribute round reward in proportion to each layer's effective A@B change."""
    selected_layers = [int(layer_id) for layer_id in selected_layers]
    if not selected_layers:
        raise ValueError("selected_layers must not be empty when assigning credit")

    product_deltas = compute_effective_product_delta_norms(
        global_state_before=global_state_before,
        global_state_after=global_state_after,
        selected_layers=selected_layers,
    )
    total_product_delta = float(
        sum(product_deltas[layer_id] for layer_id in selected_layers)
    )
    if total_product_delta <= float(eps):
        weights = {
            layer_id: float(1.0 / len(selected_layers))
            for layer_id in selected_layers
        }
        fallback_used = True
    else:
        weights = {
            layer_id: float(product_deltas[layer_id] / total_product_delta)
            for layer_id in selected_layers
        }
        fallback_used = False

    credits = {
        layer_id: float(reward) * weights[layer_id]
        for layer_id in selected_layers
    }
    return {
        "credits": credits,
        "weights": weights,
        "product_deltas": product_deltas,
        "total_product_delta": total_product_delta,
        "fallback_used": fallback_used,
        "credit_mode": "product_delta",
    }


def _json_ready_factor_state(state_dict):
    out = {}
    for layer_id, payload in state_dict.items():
        out[str(layer_id)] = {
            key: float(value) if isinstance(value, (int, float, np.floating)) else value
            for key, value in payload.items()
        }
    return out


def _restore_int_keyed_factor_state(raw_state, total_layers, value_kind):
    restored = {}
    for layer_id in range(int(total_layers)):
        key = str(layer_id)
        raw_payload = raw_state.get(key, raw_state.get(layer_id, {})) if raw_state else {}
        if value_kind == "score":
            restored[layer_id] = {
                "A": float(raw_payload.get("A", 0.0)),
                "B": float(raw_payload.get("B", 0.0)),
            }
        else:
            restored[layer_id] = {
                "A": int(raw_payload.get("A", 0)),
                "B": int(raw_payload.get("B", 0)),
            }
    return restored


def save_adaptive_factor_checkpoint(log_dir, payload):
    if not payload:
        return
    pkl_path = os.path.join(log_dir, "adaptive_factor_state.pkl")
    state_path = os.path.join(log_dir, "adaptive_factor_state.json")
    rng_path = os.path.join(log_dir, "server_rng_state.pkl")
    arc_rng_path = os.path.join(log_dir, "arc_aslora_rng_state.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f)
    serializable = {
        "factor_selection_mode": payload["factor_selection_mode"],
        "warmup_rounds": int(payload["warmup_rounds"]),
        "layer_random_prob": float(payload["layer_random_prob"]),
        "factor_random_prob": float(payload["factor_random_prob"]),
        "factor_score_beta": float(payload["factor_score_beta"]),
        "factor_lr": float(payload["factor_lr"]),
        "factor_scores": _json_ready_factor_state(payload["factor_scores"]) if payload.get("factor_scores") is not None else None,
        "factor_select_counts": _json_ready_factor_state(payload["factor_select_counts"]) if payload.get("factor_select_counts") is not None else None,
        "factor_selector_state": None,
        "arc_aslora_selector_state": None,
        "outcome_alignment_selector_state": None,
        "cmfv_selector_state": None,
        "gi_cmfv_selector_state": None,
        "cfvg_incumbent_factors": (
            {
                str(layer_id): str(factor)
                for layer_id, factor in payload[
                    "cfvg_incumbent_factors"
                ].items()
            }
            if payload.get("cfvg_incumbent_factors") is not None
            else None
        ),
    }
    if payload.get("factor_selector_state") is not None:
        factor_selector_state = payload["factor_selector_state"]
        serializable["factor_selector_state"] = {
            "score_a": float(factor_selector_state["score_a"]),
            "score_b": float(factor_selector_state["score_b"]),
            "count_a": int(factor_selector_state["count_a"]),
            "count_b": int(factor_selector_state["count_b"]),
            "last_selected_a": factor_selector_state["last_selected_a"],
            "last_selected_b": factor_selector_state["last_selected_b"],
            "epsilon": float(factor_selector_state["epsilon"]),
            "beta": float(factor_selector_state["beta"]),
            "max_staleness": int(factor_selector_state["max_staleness"]),
            "reward_normalization": str(factor_selector_state["reward_normalization"]),
            "score_clip": factor_selector_state["score_clip"],
        }
    if payload.get("arc_aslora_selector_state") is not None:
        arc_state = payload["arc_aslora_selector_state"]
        serializable["arc_aslora_selector_state"] = {
            "total_layers": int(arc_state["total_layers"]),
            "warmup_rounds": int(arc_state["warmup_rounds"]),
            "ema_beta": float(arc_state["ema_beta"]),
            "temperature_0": float(arc_state["temperature_0"]),
            "temperature_min": float(arc_state["temperature_min"]),
            "temperature_gamma": float(arc_state["temperature_gamma"]),
            "score_eps": float(arc_state["score_eps"]),
            "ema_scores": _json_ready_factor_state(arc_state["ema_scores"]),
            "last_temperature": float(arc_state["last_temperature"]),
        }
    if payload.get("outcome_alignment_selector_state") is not None:
        outcome_state = payload["outcome_alignment_selector_state"]
        serializable["outcome_alignment_selector_state"] = {
            "total_layers": int(outcome_state["total_layers"]),
            "min_observations": int(outcome_state["min_observations"]),
            "ema_decay": float(outcome_state["ema_decay"]),
            "alignment_weight": float(outcome_state["alignment_weight"]),
            "ucb_coefficient": float(outcome_state["ucb_coefficient"]),
            "reward_scale_decay": float(outcome_state["reward_scale_decay"]),
            "credit_clip": float(outcome_state["credit_clip"]),
            "utilities": _json_ready_factor_state(outcome_state["utilities"]),
            "counts": _json_ready_factor_state(outcome_state["counts"]),
            "reward_abs_ema": float(outcome_state["reward_abs_ema"]),
            "rounds_updated": int(outcome_state["rounds_updated"]),
        }
    if payload.get("cmfv_selector_state") is not None:
        serializable["cmfv_selector_state"] = payload["cmfv_selector_state"]
    if payload.get("gi_cmfv_selector_state") is not None:
        serializable["gi_cmfv_selector_state"] = payload[
            "gi_cmfv_selector_state"
        ]
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    with open(rng_path, "wb") as f:
        pickle.dump(payload["server_rng_state"], f)
    if payload.get("arc_aslora_selector_state") is not None:
        with open(arc_rng_path, "wb") as f:
            pickle.dump(payload["arc_aslora_selector_state"]["random_state"], f)


def load_adaptive_factor_checkpoint(
        ckpt_dir,
        total_layers,
        logger=None,
):
    pkl_path = os.path.join(ckpt_dir, "adaptive_factor_state.pkl")
    state_path = os.path.join(ckpt_dir, "adaptive_factor_state.json")
    rng_path = os.path.join(ckpt_dir, "server_rng_state.pkl")
    arc_rng_path = os.path.join(ckpt_dir, "arc_aslora_rng_state.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    if not os.path.exists(state_path):
        if logger is not None:
            logger.warning(
                "Adaptive factor checkpoint fields missing in %s; initializing factor scores/counts to zeros.",
                ckpt_dir,
            )
        return None

    with open(state_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    rng_state = None
    if os.path.exists(rng_path):
        with open(rng_path, "rb") as f:
            rng_state = pickle.load(f)
    elif logger is not None:
        logger.warning(
            "server_rng_state.pkl missing in %s; resetting adaptive selection RNG to seed initialization.",
            ckpt_dir,
        )
    arc_state = raw.get("arc_aslora_selector_state")
    if arc_state is not None:
        arc_state["ema_scores"] = {
            int(layer_id): {
                factor: None if value is None else float(value)
                for factor, value in factor_scores.items()
            }
            for layer_id, factor_scores in arc_state["ema_scores"].items()
        }
        if os.path.exists(arc_rng_path):
            with open(arc_rng_path, "rb") as f:
                arc_state["random_state"] = pickle.load(f)
        else:
            arc_state = None
            if logger is not None:
                logger.warning(
                    "arc_aslora_rng_state.pkl missing in %s; ARC+AS-LoRA selector state ignored.",
                    ckpt_dir,
                )
    return {
        "factor_selection_mode": raw.get("factor_selection_mode", "adaptive_score"),
        "warmup_rounds": int(raw.get("warmup_rounds", 2)),
        "layer_random_prob": float(raw.get("layer_random_prob", 0.20)),
        "factor_random_prob": float(raw.get("factor_random_prob", 0.20)),
        "factor_score_beta": float(raw.get("factor_score_beta", 1.0)),
        "factor_lr": float(raw.get("factor_lr", 0.0)),
        "factor_scores": _restore_int_keyed_factor_state(
            raw.get("factor_scores"), total_layers, value_kind="score"
        ) if raw.get("factor_scores") is not None else None,
        "factor_select_counts": _restore_int_keyed_factor_state(
            raw.get("factor_select_counts"), total_layers, value_kind="count"
        ) if raw.get("factor_select_counts") is not None else None,
        "factor_selector_state": raw.get("factor_selector_state"),
        "arc_aslora_selector_state": arc_state,
        "outcome_alignment_selector_state": raw.get(
            "outcome_alignment_selector_state"
        ),
        "cmfv_selector_state": raw.get("cmfv_selector_state"),
        "gi_cmfv_selector_state": raw.get("gi_cmfv_selector_state"),
        "cfvg_incumbent_factors": (
            {
                int(layer_id): str(factor)
                for layer_id, factor in raw[
                    "cfvg_incumbent_factors"
                ].items()
            }
            if raw.get("cfvg_incumbent_factors") is not None
            else None
        ),
        "server_rng_state": rng_state,
    }


def clone_lora_factor_state(model):
    state = {}
    for name, param in model.named_parameters():
        if name.split('.')[-1] in {"lora_a", "lora_b"}:
            state[name] = param.detach().cpu().clone()
    return state


def _compose_lora_delta(lora_a, lora_b):
    if lora_a.ndim != 2 or lora_b.ndim != 2:
        return None
    if lora_a.shape[1] == lora_b.shape[0]:
        return torch.matmul(lora_a, lora_b)
    if lora_b.shape[1] == lora_a.shape[0]:
        return torch.matmul(lora_b, lora_a)
    return None


def compute_layer_effective_delta_norms(prev_lora_state, curr_model):
    if prev_lora_state is None:
        return {}
    curr_lora_state = clone_lora_factor_state(curr_model)
    layer_norms = {}
    for name, prev_a in prev_lora_state.items():
        if name.split('.')[-1] != "lora_a":
            continue
        prefix = name.rsplit('.', 1)[0]
        b_name = f"{prefix}.lora_b"
        prev_b = prev_lora_state.get(b_name)
        curr_a = curr_lora_state.get(name)
        curr_b = curr_lora_state.get(b_name)
        if prev_b is None or curr_a is None or curr_b is None:
            continue
        prev_delta = _compose_lora_delta(prev_a.float(), prev_b.float())
        curr_delta = _compose_lora_delta(curr_a.float(), curr_b.float())
        if prev_delta is None or curr_delta is None:
            continue
        layer_idx = int(name.split('.')[-5])
        delta_norm = torch.norm(curr_delta - prev_delta, p='fro').item()
        layer_norms[layer_idx] = layer_norms.get(layer_idx, 0.0) + float(delta_norm)
    return layer_norms


def select_layers_epsilon_greedy(
        layer_scores,
        num_layers,
        epsilon_layer,
        total_layers=12,
        layer_select_mode="max",
):
    """
    Paper epsilon-greedy on contribution dictionary O_t.
    layer_select_mode='max': TopK by largest c_{t,l}; 'min': smallest first.
    """
    num_layers = max(1, min(int(num_layers), int(total_layers)))
    layer_scores = np.asarray(layer_scores, dtype=float)
    if random.random() < 1 - epsilon_layer:
        order = np.argsort(layer_scores)
        if str(layer_select_mode).lower() == "max":
            return [int(i) for i in order[::-1][:num_layers].tolist()]
        return [int(i) for i in order[:num_layers].tolist()]
    return [int(i) for i in random.sample(range(total_layers), num_layers)]


def save_client_metric_plots(
        log_dir,
        client_names,
        train_acc_history,
        test_acc_history,
        train_loss_history,
        test_loss_history,
        server_acc_history,
        server_loss_history,
        dqn_loss_history,
):
    plot_dir = os.path.join(log_dir, "client_metric_plots")
    os.makedirs(plot_dir, exist_ok=True)

    train_acc_history = np.asarray(train_acc_history, dtype=float)
    test_acc_history = np.asarray(test_acc_history, dtype=float)
    train_loss_history = np.asarray(train_loss_history, dtype=float)
    test_loss_history = np.asarray(test_loss_history, dtype=float)
    rounds = np.arange(1, train_acc_history.shape[0] + 1)

    for client_id, client_name in enumerate(client_names):
        safe_name = _safe_filename(client_name)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(rounds, train_acc_history[:, client_id], label="train_acc")
        axes[0].plot(rounds, test_acc_history[:, client_id], label="test_acc")
        axes[0].set_title(f"Client {client_id} ({client_name}) Accuracy")
        axes[0].set_xlabel("Communication Round")
        axes[0].set_ylabel("Accuracy")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(rounds, train_loss_history[:, client_id], label="train_loss")
        axes[1].plot(rounds, test_loss_history[:, client_id], label="test_loss")
        axes[1].set_title(f"Client {client_id} ({client_name}) Loss")
        axes[1].set_xlabel("Communication Round")
        axes[1].set_ylabel("Loss")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"client_{client_id}_{safe_name}_metrics.png"), dpi=200)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for client_id, client_name in enumerate(client_names):
        axes[0].plot(rounds, test_acc_history[:, client_id], label=f"{client_id}-{client_name}")
        axes[1].plot(rounds, test_loss_history[:, client_id], label=f"{client_id}-{client_name}")
    axes[0].set_title("All Clients Test Accuracy")
    axes[0].set_xlabel("Communication Round")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("All Clients Test Loss")
    axes[1].set_xlabel("Communication Round")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "all_clients_test_metrics.png"), dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    server_rounds = np.arange(1, len(server_acc_history) + 1)
    axes[0].plot(server_rounds, server_acc_history, label="server_val_acc")
    axes[0].set_title("Server Validation Accuracy")
    axes[0].set_xlabel("Communication Round")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(server_rounds, server_loss_history, label="server_val_loss")
    axes[1].set_title("Server Validation Loss")
    axes[1].set_xlabel("Communication Round")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "server_validation_metrics.png"), dpi=200)
    plt.close(fig)

    if len(dqn_loss_history) > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(np.arange(1, len(dqn_loss_history) + 1), dqn_loss_history, label="double_dqn_loss")
        ax.set_title("Double DQN Loss")
        ax.set_xlabel("Double DQN Update Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "double_dqn_loss.png"), dpi=200)
        plt.close(fig)


def _bbb_cycle_segments(max_round, cycle_len=3):
  """Return (start, end, cycle_index) inclusive round bands for BBB-style schedules."""
  if max_round < 1:
    return []
  segments = []
  start = 1
  cycle = 0
  for r in range(2, max_round + 2):
    if r > max_round or (r - 1) % cycle_len == 0:
      end = min(r - 1, max_round)
      if end >= start:
        segments.append((start, end, cycle))
      start = r
      cycle += 1
  return segments


def _client_macro_f1_series(client_round_metrics_history, n_clients, metric_key="test_macro_f1"):
  """Build per-client macro-F1 arrays indexed by round (1-based keys in history)."""
  grouped = {client_id: {} for client_id in range(n_clients)}
  for row in client_round_metrics_history:
    client_id = int(row["client_id"])
    if client_id not in grouped:
      continue
    grouped[client_id][int(row["round"])] = float(row[metric_key])
  if not grouped:
    return [], {}, [], []
  max_round = max(
    round_idx
    for per_client in grouped.values()
    for round_idx in per_client
  )
  rounds = list(range(1, max_round + 1))
  per_client_series = {}
  mean_vals = []
  std_vals = []
  for round_idx in rounds:
    vals = [
      grouped[cid][round_idx]
      for cid in sorted(grouped)
      if round_idx in grouped[cid]
    ]
    mean_vals.append(float(np.mean(vals)) if vals else np.nan)
    std_vals.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
  for client_id in sorted(grouped):
    per_client_series[client_id] = [
      grouped[client_id].get(round_idx, np.nan) for round_idx in rounds
    ]
  return rounds, per_client_series, mean_vals, std_vals


def save_macro_f1_convergence_plots(
    log_dir,
    client_names,
    server_macro_f1_history,
    client_round_metrics_history,
    coordinate_schedule_name="bba",
    target_rounds=None,
):
  """Plot pooled-val and per-client test macro-F1 immediately after training."""
  plot_dir = os.path.join(log_dir, "macro_f1_plots")
  os.makedirs(plot_dir, exist_ok=True)

  server_f1 = np.asarray(server_macro_f1_history, dtype=float)
  n_completed = len(server_f1)
  if n_completed == 0:
    logger.warning("macro-F1 plot skipped: empty server_macro_f1_history.")
    return plot_dir

  n_clients = len(client_names)
  rounds_cli, per_client_series, client_mean, client_std = _client_macro_f1_series(
    client_round_metrics_history, n_clients, metric_key="test_macro_f1"
  )
  rounds_srv = np.arange(1, n_completed + 1)

  pooled_color = "#d62728"
  client_color = "#2ca02c"
  fig, ax = plt.subplots(figsize=(12, 6))

  if str(coordinate_schedule_name).lower() == "bbb":
    palette = ["#f0f4ff", "#fff4f0"]
    for seg_start, seg_end, cycle_idx in _bbb_cycle_segments(n_completed, cycle_len=3):
      ax.axvspan(
        seg_start - 0.5,
        seg_end + 0.5,
        color=palette[cycle_idx % len(palette)],
        alpha=0.35,
        zorder=0,
      )

  if rounds_cli:
    for client_id, series in per_client_series.items():
      label = (
        f"client {client_id} ({client_names[client_id]})"
        if client_id < len(client_names)
        else f"client {client_id}"
      )
      ax.plot(rounds_cli, series, alpha=0.35, linewidth=1.0, color=client_color)
    mean_arr = np.asarray(client_mean, dtype=float)
    std_arr = np.asarray(client_std, dtype=float)
    ax.plot(
      rounds_cli,
      mean_arr,
      color=client_color,
      linewidth=2.2,
      label="Client test macro-F1 (mean)",
      zorder=3,
    )
    ax.fill_between(
      rounds_cli,
      mean_arr - std_arr,
      mean_arr + std_arr,
      color=client_color,
      alpha=0.15,
      zorder=2,
    )

  ax.plot(
    rounds_srv,
    server_f1,
    color=pooled_color,
    linewidth=2.4,
    marker="o",
    markersize=3,
    label="Pooled client vali macro-F1",
    zorder=4,
  )
  best_idx = int(np.argmax(server_f1))
  ax.scatter(
    [best_idx + 1],
    [server_f1[best_idx]],
    color=pooled_color,
    s=70,
    zorder=5,
    label=f"Best pooled F1={server_f1[best_idx]:.4f} @R{best_idx + 1}",
  )

  title_rounds = target_rounds or n_completed
  ax.set_title(
    f"Macro F1 convergence — {coordinate_schedule_name.upper()} "
    f"({n_completed}/{title_rounds} rounds)"
  )
  ax.set_xlabel("Communication Round")
  ax.set_ylabel("Macro F1")
  ax.grid(True, alpha=0.3)
  ax.legend(loc="best", fontsize=9)
  if str(coordinate_schedule_name).lower() == "bbb":
    ax.text(
      0.01,
      0.02,
      "Shaded bands: BBB 3-round cycles (B-only updates)",
      transform=ax.transAxes,
      fontsize=8,
      color="gray",
    )
  fig.tight_layout()
  out_png = os.path.join(plot_dir, "macro_f1_convergence.png")
  root_png = os.path.join(log_dir, "macro_f1_convergence.png")
  fig.savefig(out_png, dpi=200)
  fig.savefig(root_png, dpi=200)
  plt.close(fig)

  summary_csv = os.path.join(plot_dir, "macro_f1_round_summary.csv")
  with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(
      ["round", "pooled_val_macro_f1", "client_test_macro_f1_mean", "client_test_macro_f1_std"]
    )
    for round_idx in range(1, n_completed + 1):
      cli_i = round_idx - 1
      writer.writerow([
        round_idx,
        float(server_f1[round_idx - 1]),
        float(client_mean[cli_i]) if cli_i < len(client_mean) else "",
        float(client_std[cli_i]) if cli_i < len(client_std) else "",
      ])

  summary = {
    "n_rounds_completed": n_completed,
    "best_pooled_val_macro_f1": float(server_f1[best_idx]),
    "best_pooled_val_round": int(best_idx + 1),
    "coordinate_schedule": coordinate_schedule_name,
    "plot_path": out_png,
  }
  with open(os.path.join(plot_dir, "macro_f1_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
  logger.info(f"Macro-F1 convergence plot saved to {out_png}")
  print(f"Macro-F1 convergence plot saved to {out_png}", flush=True)
  return plot_dir


# LoRA aggregation baselines (FedAvg-LoRA / FFA-LoRA / FLoRA); set LORA_BASELINE_MODE env before launch.
LORA_AGG_MODE = None
prepare_rl_lora_round = None
aggregate_lora_server = None

# Where-to-fine-tune baselines (FedFMSL / SPD-CFL); set LAYER_SELECT_METHOD env before launch.
LAYER_SELECT_METHOD = "rl"
layer_selector = None


def _init_lora_baseline_hooks():
    import os
    import sys

    mode = os.environ.get("LORA_BASELINE_MODE", "").strip()
    if not mode:
        return
    from jl_baseline_hooks import _baseline_callables

    global LORA_AGG_MODE, prepare_rl_lora_round, aggregate_lora_server
    prepare_fn, aggregate_fn = _baseline_callables(mode)
    LORA_AGG_MODE = mode
    prepare_rl_lora_round = prepare_fn
    aggregate_lora_server = aggregate_fn

    # runpy.run_path executes this file as __main__; patch that namespace explicitly.
    main_mod = sys.modules.get("__main__")
    if main_mod is not None:
        main_mod.LORA_AGG_MODE = mode
        main_mod.prepare_rl_lora_round = prepare_fn
        main_mod.aggregate_lora_server = aggregate_fn
    try:
        import jl_engine as fed

        fed.LORA_AGG_MODE = mode
        fed.prepare_rl_lora_round = prepare_fn
        fed.aggregate_lora_server = aggregate_fn
    except ImportError:
        pass


_init_lora_baseline_hooks()


def _init_layer_selection_hooks():
    import os
    import sys

    mode = os.environ.get("LAYER_SELECT_METHOD", "").strip().lower()
    if mode not in ("fedfmsl", "spdcfl"):
        return

    global LAYER_SELECT_METHOD
    LAYER_SELECT_METHOD = mode

    targets = []
    main_mod = sys.modules.get("__main__")
    if main_mod is not None:
        targets.append(main_mod)
    try:
        import jl_engine as fed

        if fed not in targets:
            targets.append(fed)
    except ImportError:
        pass
    for mod in targets:
        mod.LAYER_SELECT_METHOD = mode


_init_layer_selection_hooks()


if __name__ == "__main__":
    def load_layers_from_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            layers_list = json.load(f)
        return layers_list


    loaded_layers_list = None
    # Get command line arguments
    args = get_args()
    # Seed CUDA before the memory guard creates its first CUDA allocation.
    set_seed(args.init_seed)
    if args.gpu_memory_reserve:
        init_gpu_memory_guard(
            args.gpu_memory_reserve,
            logger=None,
            early_claim=True,
            training_headroom_gb=float(args.gpu_memory_training_headroom),
            start_watchdog=False,
            persistent_reservation=bool(args.gpu_memory_persistent_reserve),
        )
    is_fedavg_full = (not args.lora) and (args.finetune == "full")
    if is_fedavg_full:
        args.coordinate_schedule = "none"
        args.layer_select_method = "none"
        args.if_rl = False
    factor_selection_mode = str(getattr(args, "factor_selection_mode", "global_adaptive")).lower().strip()
    if factor_selection_mode not in {
        "fixed_baba",
        "global_adaptive",
        "adaptive_score",
        "product_delta",
        "arc_aslora",
        "outcome_alignment",
        "cmfv",
        "gi_cmfv",
        "qgs",
        "cfvg",
    }:
        raise ValueError(f"Unsupported factor_selection_mode={factor_selection_mode!r}")
    factor_warmup_rounds = int(getattr(args, "factor_warmup_rounds", 2))
    layer_random_prob = float(getattr(args, "layer_random_prob", 0.20))
    layer_score_prob = float(getattr(args, "layer_score_prob", 0.80))
    factor_epsilon = float(getattr(args, "factor_epsilon", 0.20))
    factor_score_beta = float(getattr(args, "factor_score_beta", 0.20))
    factor_max_staleness = int(getattr(args, "factor_max_staleness", 5))
    factor_reward_normalization = str(getattr(args, "factor_reward_normalization", "per_layer"))
    factor_score_clip = getattr(args, "factor_score_clip", None)
    factor_credit_mode = str(getattr(args, "factor_credit_mode", "equal")).lower().strip()
    if factor_credit_mode not in {"equal", "rms"}:
        raise ValueError(f"Unsupported factor_credit_mode={factor_credit_mode!r}")
    factor_random_prob = factor_epsilon
    factor_score_prob = 1.0 - factor_epsilon
    factor_lr = float(args.lr)
    aslora_ema_beta = float(args.aslora_ema_beta)
    aslora_fd_eps = float(args.aslora_fd_eps)
    aslora_norm_eps = float(args.aslora_norm_eps)
    aslora_temperature_0 = float(args.aslora_temperature_0)
    aslora_temperature_min = float(args.aslora_temperature_min)
    aslora_temperature_gamma = float(args.aslora_temperature_gamma)
    alignment_min_observations = int(args.alignment_min_observations)
    alignment_ema_decay = float(args.alignment_ema_decay)
    alignment_local_weight = float(args.alignment_local_weight)
    alignment_ucb_coefficient = float(args.alignment_ucb_coefficient)
    alignment_reward_scale_decay = float(args.alignment_reward_scale_decay)
    alignment_credit_clip = float(args.alignment_credit_clip)
    alignment_validation_batch_size = int(args.alignment_validation_batch_size)
    cmfv_validation_batch_size = int(args.cmfv_validation_batch_size)
    cmfv_curvature_fd_eps = float(args.cmfv_curvature_fd_eps)
    cmfv_coherence_prior = float(args.cmfv_coherence_prior)
    cmfv_coherence_ema_decay = float(args.cmfv_coherence_ema_decay)
    cmfv_count_discount = float(args.cmfv_count_discount)
    cmfv_confidence_z = float(args.cmfv_confidence_z)
    cmfv_min_task_fit = float(args.cmfv_min_task_fit)
    cmfv_exploration_beta = float(args.cmfv_exploration_beta)
    cmfv_posterior_temperature = float(args.cmfv_posterior_temperature)
    gi_cmfv_validation_batch_size = int(
        args.gi_cmfv_validation_batch_size
    )
    gi_cmfv_rank_rtol = float(args.gi_cmfv_rank_rtol)
    gi_cmfv_ema_decay = float(args.gi_cmfv_ema_decay)
    gi_cmfv_count_discount = float(args.gi_cmfv_count_discount)
    gi_cmfv_server_weight = float(args.gi_cmfv_server_weight)
    gi_cmfv_disagreement_penalty = float(
        args.gi_cmfv_disagreement_penalty
    )
    gi_cmfv_temperature_0 = float(args.gi_cmfv_temperature_0)
    gi_cmfv_temperature_min = float(args.gi_cmfv_temperature_min)
    gi_cmfv_temperature_gamma = float(args.gi_cmfv_temperature_gamma)
    qgs_validation_batch_size = int(args.qgs_validation_batch_size)
    qgs_rank_rtol = float(args.qgs_rank_rtol)
    cfvg_probe_train_batch_size = int(args.cfvg_probe_train_batch_size)
    cfvg_probe_validation_batch_size = int(
        args.cfvg_probe_validation_batch_size
    )
    cfvg_probe_steps = int(args.cfvg_probe_steps)
    cfvg_probe_validation_batches = int(
        args.cfvg_probe_validation_batches
    )
    cfvg_confidence_z = float(args.cfvg_confidence_z)
    assert abs(layer_random_prob + layer_score_prob - 1.0) < 1e-8
    assert abs(factor_random_prob + factor_score_prob - 1.0) < 1e-8
    if factor_selection_mode not in {
        "arc_aslora",
        "product_delta",
        "cmfv",
        "gi_cmfv",
        "qgs",
        "cfvg",
    }:
        assert factor_warmup_rounds == 2
    coordinate_schedule_label = set_active_coordinate_schedule(args.coordinate_schedule)
    epsilon_layer = float(args.epsilon_layer)
    epsilon_matrix = float(args.epsilon_matrix)
    layer_select_mode = str(args.layer_select_mode).lower()
    layer_select_method = str(
        os.environ.get("LAYER_SELECT_METHOD", args.layer_select_method) or "rl"
    ).lower().strip()
    if layer_select_method not in ("rl", "all", "fedfmsl", "spdcfl", "random_g", "none"):
        layer_select_method = "rl"
    if layer_select_method == "rl" and layer_select_mode != "max":
        print(
            f"Overriding layer_select_mode={layer_select_mode} to 'max' for RLCD-FedLoRA.",
            flush=True,
        )
        layer_select_mode = "max"
        args.layer_select_mode = "max"
    if factor_selection_mode == "global_adaptive":
        if is_fedavg_full or layer_select_method != "rl" or not args.if_rl or not args.lora or LORA_AGG_MODE:
            raise ValueError(
                "global_adaptive mode is only supported for RLCD-FedLoRA "
                "(if_rl=True, lora=True, layer_select_method=rl, no LoRA baseline hooks)."
            )
        if factor_reward_normalization != "per_layer":
            raise ValueError(
                f"Unsupported factor_reward_normalization={factor_reward_normalization!r}; "
                "only 'per_layer' is currently supported."
            )
    if factor_selection_mode == "adaptive_score":
        if (
            is_fedavg_full
            or layer_select_method not in ADAPTIVE_CD_LAYER_METHODS
            or not args.if_rl
            or not args.lora
            or LORA_AGG_MODE
        ):
            raise ValueError(
                "adaptive_score mode is only supported for RLCD-FedLoRA with "
                "layer_select_method in {rl, all, fedfmsl, spdcfl, random_g} "
                "(if_rl=True, lora=True, no LoRA baseline hooks)."
            )
        if factor_credit_mode not in {"equal", "rms"}:
            raise ValueError(f"Unsupported factor_credit_mode={factor_credit_mode!r} for adaptive_score")
        effective_alr = factor_lr * float(args.lr_a_scale)
        effective_blr = factor_lr
        if abs(effective_alr - effective_blr) > 1e-12:
            raise ValueError(
                "adaptive_score mode requires equal A/B learning rates: "
                f"alr={effective_alr}, blr={effective_blr}. "
                "Set --lr_a_scale 1.0 or use fixed_baba mode."
            )
        args.lr = factor_lr
        args.lr_a_scale = 1.0
    if factor_selection_mode == "product_delta":
        if (
            is_fedavg_full
            or layer_select_method != "rl"
            or not args.if_rl
            or not args.lora
            or LORA_AGG_MODE
        ):
            raise ValueError(
                "product_delta requires ARC-FedLoRA with if_rl=True, lora=True, "
                "layer_select_method=rl, and no LoRA baseline hooks."
            )
        if abs(factor_lr * float(args.lr_a_scale) - factor_lr) > 1e-12:
            raise ValueError(
                "product_delta requires equal A/B learning rates; "
                "set --lr_a_scale 1.0"
            )
        if factor_warmup_rounds < 0:
            raise ValueError("product_delta factor_warmup_rounds must be non-negative")
        args.lr = factor_lr
        args.lr_a_scale = 1.0
    if factor_selection_mode == "arc_aslora":
        if (
            is_fedavg_full
            or layer_select_method != "rl"
            or not args.if_rl
            or not args.lora
            or LORA_AGG_MODE
        ):
            raise ValueError(
                "arc_aslora is a controlled ARC-FedLoRA baseline and requires "
                "if_rl=True, lora=True, layer_select_method=rl, and no baseline hooks."
            )
        if bool(args.shared_backbone_lora):
            raise ValueError(
                "arc_aslora currently requires the standard client-model runtime so FD scoring "
                "can restore each client's parameters and gradients exactly."
            )
        if abs(factor_lr * float(args.lr_a_scale) - factor_lr) > 1e-12:
            raise ValueError("arc_aslora requires equal A/B learning rates; set --lr_a_scale 1.0")
        if factor_warmup_rounds < 0:
            raise ValueError("arc_aslora factor_warmup_rounds must be non-negative")
        if not 0.0 <= aslora_ema_beta < 1.0:
            raise ValueError("aslora_ema_beta must be in [0, 1)")
        if aslora_fd_eps <= 0.0 or aslora_norm_eps <= 0.0:
            raise ValueError("AS-LoRA finite-difference epsilons must be positive")
        if aslora_temperature_0 <= 0.0 or aslora_temperature_min <= 0.0:
            raise ValueError("AS-LoRA temperatures must be positive")
        if not 0.0 < aslora_temperature_gamma <= 1.0:
            raise ValueError("aslora_temperature_gamma must be in (0, 1]")
        args.lr = factor_lr
        args.lr_a_scale = 1.0
    if factor_selection_mode == "outcome_alignment":
        if (
            is_fedavg_full
            or layer_select_method != "rl"
            or not args.if_rl
            or not args.lora
            or LORA_AGG_MODE
        ):
            raise ValueError(
                "outcome_alignment requires ARC-FedLoRA with if_rl=True, lora=True, "
                "layer_select_method=rl, and no LoRA baseline hooks."
            )
        if abs(factor_lr * float(args.lr_a_scale) - factor_lr) > 1e-12:
            raise ValueError(
                "outcome_alignment requires equal A/B learning rates; "
                "set --lr_a_scale 1.0"
            )
        if alignment_min_observations < 1:
            raise ValueError("alignment_min_observations must be at least 1")
        if not 0.0 <= alignment_ema_decay < 1.0:
            raise ValueError("alignment_ema_decay must be in [0, 1)")
        if not 0.0 <= alignment_local_weight <= 1.0:
            raise ValueError("alignment_local_weight must be in [0, 1]")
        if alignment_ucb_coefficient < 0.0:
            raise ValueError("alignment_ucb_coefficient must be non-negative")
        if not 0.0 <= alignment_reward_scale_decay < 1.0:
            raise ValueError("alignment_reward_scale_decay must be in [0, 1)")
        if alignment_credit_clip <= 0.0:
            raise ValueError("alignment_credit_clip must be positive")
        if alignment_validation_batch_size < 1:
            raise ValueError("alignment_validation_batch_size must be positive")
        args.lr = factor_lr
        args.lr_a_scale = 1.0
    if factor_selection_mode == "cmfv":
        if (
            is_fedavg_full
            or layer_select_method != "rl"
            or not args.if_rl
            or not args.lora
            or LORA_AGG_MODE
        ):
            raise ValueError(
                "cmfv requires if_rl=True, lora=True, layer_select_method=rl, "
                "and no LoRA baseline hooks."
            )
        if abs(factor_lr * float(args.lr_a_scale) - factor_lr) > 1e-12:
            raise ValueError(
                "cmfv requires equal A/B learning rates; set --lr_a_scale 1.0"
            )
        if cmfv_validation_batch_size < 1:
            raise ValueError("cmfv_validation_batch_size must be positive")
        if cmfv_curvature_fd_eps <= 0.0:
            raise ValueError("cmfv_curvature_fd_eps must be positive")
        if not 0.0 <= cmfv_coherence_prior <= 1.0:
            raise ValueError("cmfv_coherence_prior must be in [0, 1]")
        if not 0.0 <= cmfv_coherence_ema_decay < 1.0:
            raise ValueError("cmfv_coherence_ema_decay must be in [0, 1)")
        if not 0.0 < cmfv_count_discount <= 1.0:
            raise ValueError("cmfv_count_discount must be in (0, 1]")
        if cmfv_confidence_z < 0.0 or cmfv_min_task_fit < 0.0:
            raise ValueError(
                "cmfv_confidence_z and cmfv_min_task_fit must be non-negative"
            )
        if cmfv_exploration_beta < 0.0:
            raise ValueError("cmfv_exploration_beta must be non-negative")
        if cmfv_posterior_temperature <= 0.0:
            raise ValueError("cmfv_posterior_temperature must be positive")
        args.lr = factor_lr
        args.lr_a_scale = 1.0
    if factor_selection_mode == "gi_cmfv":
        if (
            is_fedavg_full
            or layer_select_method != "rl"
            or not args.if_rl
            or not args.lora
            or LORA_AGG_MODE
        ):
            raise ValueError(
                "gi_cmfv requires if_rl=True, lora=True, "
                "layer_select_method=rl, and no baseline hooks."
            )
        if bool(args.shared_backbone_lora):
            raise ValueError(
                "gi_cmfv requires the standard client-model runtime so every "
                "client can compute counterfactual validation gradients."
            )
        if abs(factor_lr * float(args.lr_a_scale) - factor_lr) > 1e-12:
            raise ValueError(
                "gi_cmfv requires equal A/B learning rates; "
                "set --lr_a_scale 1.0"
            )
        if factor_warmup_rounds < 0:
            raise ValueError(
                "gi_cmfv factor_warmup_rounds must be non-negative"
            )
        if gi_cmfv_validation_batch_size < 1:
            raise ValueError(
                "gi_cmfv_validation_batch_size must be positive"
            )
        if gi_cmfv_rank_rtol <= 0.0:
            raise ValueError("gi_cmfv_rank_rtol must be positive")
        if not 0.0 <= gi_cmfv_ema_decay < 1.0:
            raise ValueError("gi_cmfv_ema_decay must be in [0, 1)")
        if not 0.0 < gi_cmfv_count_discount <= 1.0:
            raise ValueError(
                "gi_cmfv_count_discount must be in (0, 1]"
            )
        if not 0.0 <= gi_cmfv_server_weight <= 1.0:
            raise ValueError("gi_cmfv_server_weight must be in [0, 1]")
        if gi_cmfv_disagreement_penalty < 0.0:
            raise ValueError(
                "gi_cmfv_disagreement_penalty must be non-negative"
            )
        if (
            gi_cmfv_temperature_0 <= 0.0
            or gi_cmfv_temperature_min <= 0.0
            or gi_cmfv_temperature_min > gi_cmfv_temperature_0
            or not 0.0 < gi_cmfv_temperature_gamma <= 1.0
        ):
            raise ValueError(
                "GI-CMFV temperatures require 0 < min <= initial and "
                "temperature_gamma in (0, 1]"
            )
        args.lr = factor_lr
        args.lr_a_scale = 1.0
    if factor_selection_mode == "qgs":
        if (
            is_fedavg_full
            or layer_select_method != "rl"
            or not args.if_rl
            or not args.lora
            or LORA_AGG_MODE
        ):
            raise ValueError(
                "qgs requires if_rl=True, lora=True, "
                "layer_select_method=rl, and no baseline hooks."
            )
        if abs(factor_lr * float(args.lr_a_scale) - factor_lr) > 1e-12:
            raise ValueError(
                "qgs requires equal A/B learning rates; "
                "set --lr_a_scale 1.0"
            )
        if factor_warmup_rounds != 0:
            raise ValueError(
                "qgs is a prior-free deterministic selector and requires "
                "--factor_warmup_rounds 0"
            )
        if qgs_validation_batch_size < 1:
            raise ValueError("qgs_validation_batch_size must be positive")
        if qgs_rank_rtol <= 0.0:
            raise ValueError("qgs_rank_rtol must be positive")
        args.lr = factor_lr
        args.lr_a_scale = 1.0
    if factor_selection_mode == "cfvg":
        if (
            is_fedavg_full
            or layer_select_method != "rl"
            or not args.if_rl
            or not args.lora
            or LORA_AGG_MODE
        ):
            raise ValueError(
                "cfvg requires if_rl=True, lora=True, "
                "layer_select_method=rl, and no baseline hooks."
            )
        if bool(args.shared_backbone_lora):
            raise ValueError(
                "cfvg requires the standard client-model runtime so its "
                "counterfactual optimizer matches client training."
            )
        if abs(factor_lr * float(args.lr_a_scale) - factor_lr) > 1e-12:
            raise ValueError(
                "cfvg requires equal A/B learning rates; "
                "set --lr_a_scale 1.0"
            )
        if factor_warmup_rounds != 0:
            raise ValueError(
                "cfvg observes both interventions every round and requires "
                "--factor_warmup_rounds 0"
            )
        if cfvg_probe_train_batch_size < 1:
            raise ValueError("cfvg_probe_train_batch_size must be positive")
        if cfvg_probe_validation_batch_size < 1:
            raise ValueError(
                "cfvg_probe_validation_batch_size must be positive"
            )
        if cfvg_probe_steps < 1:
            raise ValueError("cfvg_probe_steps must be positive")
        if cfvg_probe_validation_batches < 1:
            raise ValueError(
                "cfvg_probe_validation_batches must be positive"
            )
        if cfvg_confidence_z < 0.0:
            raise ValueError("cfvg_confidence_z must be non-negative")
        args.lr = factor_lr
        args.lr_a_scale = 1.0
    if args.if_rl and args.cd is not None:
        print(
            f"Ignoring cd={args.cd} because RL layer selection is enabled.",
            flush=True,
        )
        args.cd = None
    LAYER_SELECT_METHOD = layer_select_method
    if is_fedavg_full:
        layer_select_method = "none"
        LAYER_SELECT_METHOD = "none"
        layer_selector = None
    elif layer_select_method == "fedfmsl":
        from jl_layer_selection import FedFMSLLayerSelector

        layer_selector = FedFMSLLayerSelector(
            queue_size=args.fedfmsl_queue_size,
            delta_threshold=args.fedfmsl_delta,
        )
    elif layer_select_method == "spdcfl":
        from jl_layer_selection import SPDCFLLayerSelector

        layer_selector = SPDCFLLayerSelector(
            dropout_step=args.spdcfl_dropout_step,
            ema_decay=args.spdcfl_ema_decay,
        )
    else:
        layer_selector = None
    if not is_fedavg_full and (not args.if_rl) and args.cd is not None:
        loaded_layers_list = load_layers_from_file('data/layers_output.json')
        print(loaded_layers_list)
    # 多线程时每个线程若再用多核 BLAS，会严重超订 CPU 看起来像卡死；限制为 1
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)
    now = datetime.now()
    change = int(args.change)
    _baseline_mode = LORA_AGG_MODE or os.environ.get("LORA_BASELINE_MODE", "").strip()
    baseline_tag = f"{_baseline_mode}_lora_" if _baseline_mode else ""
    if is_fedavg_full:
        baseline_tag = "fedavg_full_"
    if layer_select_method == "rl":
        if factor_selection_mode in {
            "adaptive_score",
            "product_delta",
            "global_adaptive",
            "arc_aslora",
            "outcome_alignment",
            "cmfv",
            "gi_cmfv",
            "qgs",
            "cfvg",
        }:
            if factor_selection_mode == "gi_cmfv":
                where_tag = (
                    f"layers_{layer_select_mode}_{factor_selection_mode}_"
                    f"w{factor_warmup_rounds}_"
                )
            else:
                where_tag = (
                    f"layers_{layer_select_mode}_{factor_selection_mode}_"
                )
        else:
            where_tag = f"layers_{layer_select_mode}_"
    elif layer_select_method == "all":
        where_tag = f"all_layers_{factor_selection_mode}_"
    elif layer_select_method == "none":
        where_tag = "no_rl_"
    elif layer_select_method in {"fedfmsl", "spdcfl"} and factor_selection_mode == "adaptive_score":
        where_tag = f"where_{layer_select_method}_{factor_selection_mode}_"
    else:
        where_tag = f"where_{layer_select_method}_"
    sched_name = "none" if is_fedavg_full else args.coordinate_schedule
    if is_fedavg_full:
        log_dir = (
            "log/"
            + now.strftime("log_%Y%m%d_%H%M%S")
            + "_"
            + baseline_tag
            + f"sched_{sched_name}_"
            + f"{args.sample_fraction}_"
            + f"{args.finetune}_"
            + str(args.n_parties)
            + "_"
            + str(args.epochs)
            + "_"
            + str(args.data)
            + "_tok512"
        )
    else:
        log_dir = (
            "log/"
            + now.strftime("log_%Y%m%d_%H%M%S")
            + "_"
            + baseline_tag
            + f"sched_{sched_name}_"
            + f"{args.if_rl}_"
            + f"{epsilon_layer}_"
            + f"{epsilon_matrix}_"
            + where_tag
            + f"{args.sample_fraction}_"
            + f"{args.finetune}_"
            + f"{args.cd}_"
            + str(args.n_parties)
            + "_"
            + str(args.epochs)
            + "_dqn"
            + str(args.data)
            + "_"
            + str(args.initial_rank)
            + "_"
            + str(change)
            + "轮后训练A"
        )
    log_dir = args.jl_output
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir,"record.log")
    fh = logging.FileHandler(log_filename)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    fh.setFormatter(formatter)
    # 添加handler到logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    if args.if_rl and args.cd is None:
        if layer_select_method == "rl":
            logger.info("RL layer selection active; legacy cd layer-selection override disabled.")
        else:
            logger.info(
                "Adaptive CD workflow active with layer_select_method=%s; "
                "legacy cd layer-selection override disabled.",
                layer_select_method,
            )
    logger.info(
        f"Layer selection method: {layer_select_method}"
    )
    if is_fedavg_full:
        logger.info("FedAvg-Full mode: full RoBERTa fine-tuning, no LoRA, no RL, no B/A CD.")
        print("FedAvg-Full mode: full RoBERTa fine-tuning, no LoRA, no RL, no B/A CD.", flush=True)
    elif layer_select_method == "rl":
        if args.jl_controller_variant:
            logger.info('Controller study: %s | %s', args.jl_controller_variant,
                        VARIANTS[args.jl_controller_variant])
        else:
            logger.info(
                f"  Double DQN + epsilon_layer={epsilon_layer}, mode={layer_select_mode} "
                f"({'largest' if layer_select_mode == 'max' else 'smallest'} contribution first)"
            )
    elif layer_select_method == "fedfmsl":
        logger.info(
            f"  FedFMSL SAL heuristic (queue={args.fedfmsl_queue_size}, delta={args.fedfmsl_delta})"
        )
    elif layer_select_method == "spdcfl":
        logger.info(
            f"  SPD-CFL SGC budget (dropout_step={args.spdcfl_dropout_step}, ema={args.spdcfl_ema_decay})"
        )
    elif layer_select_method == "all":
        logger.info(
            "  All-layer controlled baseline: all 12 transformer layers are selected "
            "every round; no DQN layer action or layer exploration is used."
        )
    elif layer_select_method == "random_g":
        logger.info(
            "Random-G ablation: two all-layer warm-up rounds, then uniform G in 1..12; "
            "private count RNG; unchanged 20/80 layer selection and adaptive factors; no DDQN."
        )
    if LORA_AGG_MODE:
        logger.info(f"LoRA aggregation baseline mode: {LORA_AGG_MODE}")
        print(f"LoRA aggregation baseline mode: {LORA_AGG_MODE}")
    if factor_selection_mode == "adaptive_score":
        logger.info(
            "Adaptive factor selection active: warmup_rounds=%d, layer_random_prob=%.2f, "
            "factor_random_prob=%.2f, factor_score_beta=%.2f, factor_credit_mode=%s, factor_lr=%.6f "
            "(legacy epsilon_layer=%.3f ignored for layer-set exploration in this mode)",
            factor_warmup_rounds,
            layer_random_prob,
            factor_random_prob,
            factor_score_beta,
            factor_credit_mode,
            factor_lr,
            epsilon_layer,
        )
        print(
            "Adaptive factor selection active: "
            f"warmup_rounds={factor_warmup_rounds}, "
            f"layer_random_prob={layer_random_prob}, "
            f"factor_random_prob={factor_random_prob}, "
            f"factor_score_beta={factor_score_beta}, "
            f"factor_credit_mode={factor_credit_mode}, "
            f"factor_lr={factor_lr}",
            flush=True,
        )
    elif factor_selection_mode == "product_delta":
        logger.info(
            "ARC product-delta factor selection active: warmup_rounds=%d "
            "(B/A alternating on ARC-selected layers), factor_random_prob=%.2f, "
            "factor_score_beta=%.2f, reward_credit=effective_A@B_product_delta, "
            "factor_lr=%.6f",
            factor_warmup_rounds,
            factor_random_prob,
            factor_score_beta,
            factor_lr,
        )
        print(
            "ARC product-delta factor selection active: "
            f"warmup_rounds={factor_warmup_rounds}, "
            f"factor_random_prob={factor_random_prob}, "
            f"factor_score_beta={factor_score_beta}, "
            f"factor_lr={factor_lr}",
            flush=True,
        )
    elif factor_selection_mode == "global_adaptive":
        logger.info(
            "Global adaptive factor selection active: warmup_rounds=%d, factor_epsilon=%.2f, "
            "factor_score_beta=%.2f, factor_max_staleness=%d, factor_reward_normalization=%s, "
            "factor_score_clip=%s",
            factor_warmup_rounds,
            factor_epsilon,
            factor_score_beta,
            factor_max_staleness,
            factor_reward_normalization,
            factor_score_clip,
        )
        print(
            "Global adaptive factor selection active: "
            f"warmup_rounds={factor_warmup_rounds}, "
            f"factor_epsilon={factor_epsilon}, "
            f"factor_score_beta={factor_score_beta}, "
            f"factor_max_staleness={factor_max_staleness}, "
            f"factor_reward_normalization={factor_reward_normalization}, "
            f"factor_score_clip={factor_score_clip}",
            flush=True,
        )
    elif factor_selection_mode == "arc_aslora":
        logger.info(
            "ARC+AS-LoRA controlled baseline: warmup_rounds=%d, ema_beta=%.3f, "
            "fd_eps=%.3g, norm_eps=%.3g, T0=%.3f, Tmin=%.3f, gamma=%.3f, "
            "score_aggregation=uniform, parameter_aggregation=data_size_weighted, "
            "softmax_rule=official_relative_difference",
            factor_warmup_rounds,
            aslora_ema_beta,
            aslora_fd_eps,
            aslora_norm_eps,
            aslora_temperature_0,
            aslora_temperature_min,
            aslora_temperature_gamma,
        )
        logger.info(
            "AS-LoRA release/paper difference: release defaults T0=1.0,Tmin=0.1 and "
            "normalizes (SA-SB) by |SA|+|SB|; paper text reports T0=2.0,Tmin=0.2 "
            "and raw-score softmax. Release behavior is used per baseline protocol."
        )
        if args.evaluate_client_test_every_round:
            logger.info(
                "Client-test reporting enabled: each client test split is evaluated every round; "
                "these metrics are excluded from aggregation, ARC layer feedback, AS-LoRA factor "
                "feedback, and server-validation checkpoint selection."
            )
        print(
            "ARC+AS-LoRA active: "
            f"warmup={factor_warmup_rounds}, ema_beta={aslora_ema_beta}, "
            f"fd_eps={aslora_fd_eps}, T0={aslora_temperature_0}, "
            f"Tmin={aslora_temperature_min}, gamma={aslora_temperature_gamma}",
            flush=True,
        )
    elif factor_selection_mode == "outcome_alignment":
        logger.info(
            "ARC outcome-alignment factor selection active: min_observations=%d, "
            "ema_decay=%.3f, local_weight=%.3f, ucb_coefficient=%.3f, "
            "reward_scale_decay=%.3f, credit_clip=%.3f, validation_batch_size=%d",
            alignment_min_observations,
            alignment_ema_decay,
            alignment_local_weight,
            alignment_ucb_coefficient,
            alignment_reward_scale_decay,
            alignment_credit_clip,
            alignment_validation_batch_size,
        )
        logger.info(
            "Outcome alignment uses server-validation gradients and actual aggregated "
            "LoRA updates; client test metrics and AS-LoRA finite differences are excluded."
        )
        print(
            "ARC outcome-alignment active: "
            f"min_observations={alignment_min_observations}, "
            f"ema_decay={alignment_ema_decay}, "
            f"local_weight={alignment_local_weight}, "
            f"ucb={alignment_ucb_coefficient}",
            flush=True,
        )
    elif factor_selection_mode == "cmfv":
        logger.info(
            "Natural CMFV coordinate decision active: validation_batch_size=%d, "
            "curvature_fd_eps=%.3g, coherence_prior=%.3f, coherence_ema_decay=%.3f, "
            "count_discount=%.3f, confidence_z=%.3f, min_task_fit=%.3g, "
            "exploration_beta=%.3f, posterior_temperature=%.3f",
            cmfv_validation_batch_size,
            cmfv_curvature_fd_eps,
            cmfv_coherence_prior,
            cmfv_coherence_ema_decay,
            cmfv_count_discount,
            cmfv_confidence_z,
            cmfv_min_task_fit,
            cmfv_exploration_beta,
            cmfv_posterior_temperature,
        )
        logger.info(
            "Natural CMFV keeps DDQN responsible only for the layer budget; "
            "factor task fit is the curvature-adjusted one-step validation gain and "
            "A/B selection uses an uncapped posterior advantage probability."
        )
        print(
            "Natural CMFV active: "
            f"coherence_prior={cmfv_coherence_prior}, "
            f"ema_decay={cmfv_coherence_ema_decay}, "
            f"discount={cmfv_count_discount}, "
            f"exploration_beta={cmfv_exploration_beta}, "
            f"posterior_temperature={cmfv_posterior_temperature}",
            flush=True,
        )
    elif factor_selection_mode == "gi_cmfv":
        logger.info(
            "GI-CMFV active: validation_batch_size=%d, rank_rtol=%.3g, "
            "server_weight=%.3f, disagreement_penalty=%.3f, "
            "temperature=(%.3f, %.3f, gamma=%.3f), "
            "factor_prior=none, factor_warmup_rounds=%d",
            gi_cmfv_validation_batch_size,
            gi_cmfv_rank_rtol,
            gi_cmfv_server_weight,
            gi_cmfv_disagreement_penalty,
            gi_cmfv_temperature_0,
            gi_cmfv_temperature_min,
            gi_cmfv_temperature_gamma,
            factor_warmup_rounds,
        )
        logger.info(
            "GI-CMFV evidence memory: ema_decay=%.3f, count_discount=%.3f",
            gi_cmfv_ema_decay,
            gi_cmfv_count_discount,
        )
        logger.info(
            "GI-CMFV hierarchy: DDQN selects the layer budget, ARC/CD selects "
            "the concrete layers, and GI-CMFV selects A/B only. B is valued "
            "inside col(A); A receives credit only for new effective-weight "
            "directions outside the B-update tangent. Both factors are "
            "counterfactually observed on every participating client's "
            "validation split, with uniform client evidence aggregation."
        )
        print(
            "GI-CMFV active: gauge-invariant A-revision/B-exploitation value, "
            "dual-factor client-validation evidence, disagreement-aware "
            "prior-free relative-softmax exploration",
            flush=True,
        )
    elif factor_selection_mode == "qgs":
        logger.info(
            "QGS-FS active: validation_batch_size=%d, rank_rtol=%.3g, "
            "objective=uniform participating-client validation mean, "
            "smoothness=shared effective-weight Lipschitz bound, "
            "factor_prior=none, factor_warmup_rounds=0",
            qgs_validation_batch_size,
            qgs_rank_rtol,
        )
        logger.info(
            "QGS-FS uses full quotient-tangent projections H_A=G P_B and "
            "H_B=P_A G, then deterministically selects the larger "
            "projected-gradient energy. Client test metrics, EMA, "
            "temperature, exploration, and empirical penalties are excluded."
        )
        print(
            "QGS-FS active: quotient Gauss-Southwell selection from the "
            "uniform participating-client validation-gradient mean",
            flush=True,
        )
    elif factor_selection_mode == "cfvg":
        logger.info(
            "CFVG-FS active: probe_train_batch_size=%d, "
            "probe_steps=%d, probe_validation_batch_size=%d, "
            "probe_validation_batches=%d, confidence_z=%.3f, "
            "optimizer=%s, lr=%.6g, lora_weight_decay=%.6g, "
            "objective=uniform-client class-balanced validation mean, "
            "aggregation=global-base-plus-weighted-client-deltas, "
            "factor_prior=none, factor_warmup_rounds=0",
            cfvg_probe_train_batch_size,
            cfvg_probe_steps,
            cfvg_probe_validation_batch_size,
            cfvg_probe_validation_batches,
            cfvg_confidence_z,
            args.optimizer,
            factor_lr,
            float(args.lora_weight_decay),
        )
        logger.info(
            "CFVG-FS v3 counterfactually executes separate multi-step A-only "
            "and B-only trajectories, federates deltas with the actual round "
            "weights, excludes structurally zero-update coordinates, and "
            "switches an incumbent only when paired-client validation evidence "
            "is decisive. Test metrics, factor priors, warm-up, quotas, and "
            "random exploration are excluded."
        )
        print(
            "CFVG-FS v3 active: actionability-aware multi-step dual optimizer "
            "trajectories with paired federated validation-gain selection",
            flush=True,
        )
    rl = args.if_rl
    finetune = args.finetune
    cd = args.cd
    if_lora = args.lora
    if args.if_rl and if_lora and not LORA_AGG_MODE and not is_fedavg_full:
        logger.info(f"RLCD coordinate schedule: {coordinate_schedule_label} ({args.coordinate_schedule})")
        print(f"RLCD coordinate schedule: {coordinate_schedule_label}")
    gpu_guard = get_gpu_memory_guard()
    if gpu_guard is not None:
        gpu_guard.start_watchdog()
        logger.info(
            "GPU memory reserve active on %s (training headroom %.0f GB, baseline hold %.0f GB); "
            "persistent=%s; watchdog keeps idle footprint, compute shrinks holds only.",
            ", ".join(f"{d}={gb:.0f}GB" for d, gb in gpu_guard.targets.items()),
            gpu_guard.training_headroom_gb,
            max(0.0, next(iter(gpu_guard.targets.values())) - gpu_guard.training_headroom_gb)
            if gpu_guard.targets else 0.0,
            gpu_guard.persistent_reservation,
        )
    out_dim = args.out_dim
    model = args.model
    logger.info(
        f"Training mode: lora={if_lora}, if_rl={rl}, finetune={finetune}, "
        f"early_stop_macro_f1={args.early_stop_macro_f1}"
    )
    print(
        f"Training mode: lora={if_lora}, if_rl={rl}, finetune={finetune}, "
        f"early_stop_macro_f1={args.early_stop_macro_f1}",
        flush=True,
    )
    if (if_lora):
        rank = args.initial_rank
    else:
        rank = None
    # Create directory to save log and model
    if factor_selection_mode in {"gi_cmfv", "qgs", "cfvg"}:
        argument_output_path = os.path.join(log_dir, "arguments.json")
    else:
        argument_path = (
            f"{args.n_parties}_arguments-%s.json"
            % datetime.now().strftime("%Y-%m-%d-%H%M-%S")
        )
        argument_output_path = os.path.join(args.logdir, argument_path)
    os.makedirs(args.logdir, exist_ok=True)
    print(os.getcwd())
    data_file=''
    if args.jl_data_file:
        data_file = args.jl_data_file
    elif(args.data=="20news"):
        data_file = str(Path(args.datadir) / '20news_processed_10.pkl')
    elif args.data == "yelp":
        data_file = str(Path(args.datadir) / 'yelp10.pkl')
    elif args.data == "eICU":
        data_file = str(Path(args.datadir) / 'eicu_10.pkl')
    elif args.data == "CFPB":
        data_file = str(Path(args.datadir) / 'complaints.pkl')
    elif args.data == "mimic_careunit":
        data_file = 'data/mimic_careunit_clients.pkl'
    elif args.data == "mimic_careunit_tok512":
        data_file = 'data/mimic_careunit_clients_tokenized_512.pkl'
    elif args.data == "mimic_icu10_tok512":
        data_file = 'data/mimic_icu10_server500_tokenized_512.pkl'
        # Prefer pre-merge pickle (server_vali=100 + server_test=400) when available.
        split_bak = Path(data_file + ".bak")
        if split_bak.exists():
            with split_bak.open("rb") as bak_f:
                bak_dataset = pickle.load(bak_f)
            if "server_test" in bak_dataset:
                data_file = str(split_bak)
                logger.info(f"Using split server data from {data_file}")
    else:
        raise ValueError(f"Unknown dataset: {args.data}")
    with open(data_file, 'rb') as file:
        sst_dataset = pickle.load(file)
    print(sst_dataset.keys())
    if args.data == "mimic_icu10_tok512" and args.client_partition == "icu5_large":
        from data.icu_client_partition import subset_icu10_to_large5

        sst_dataset = subset_icu10_to_large5(sst_dataset)
        logger.info(
            "client_partition=icu5_large: selected clients %s",
            sst_dataset.get("client_names", []),
        )
    cli_train = sst_dataset['train']
    cli_test = sst_dataset['test']
    cli_vali = sst_dataset['vali']
    client_names = sst_dataset.get('client_names', [f"client_{i}" for i in range(len(cli_train))])
    if args.data in {"mimic_careunit", "mimic_careunit_tok512", "mimic_icu10_tok512"} and args.n_parties != len(cli_train):
        print(f"Reset n_parties from {args.n_parties} to {len(cli_train)} for {args.data} clients.")
        args.n_parties = len(cli_train)
    client_names = client_names[:args.n_parties]
    use_mk_aggregation = bool(getattr(args, "use_mk_aggregation", False)) or args.data == "mimic_icu10_tok512"
    d_total_all_clients = sst_dataset.get("total_client_train_size")
    if d_total_all_clients is None:
        d_total_all_clients = sum(len(cli_train[i]) for i in range(len(cli_train)))
    cen_vali = resolve_server_validation_dataset(sst_dataset, cli_vali, logger)
    if 'server_vali' not in sst_dataset:
        raise ValueError('JL requires a dedicated server_vali split for controller feedback')
    cen_test = concatenate_datasets(cli_test)
    jl_protocol = EvaluationProtocol(args, log_dir, cli_vali, cli_test, client_names)
    write_json(Path(log_dir) / 'jl_arguments.json', vars(args))
    outcome_alignment_validation_loader = None
    cmfv_validation_loader = None
    gi_cmfv_validation_loader = None
    qgs_validation_batches = {}
    cfvg_train_loaders = {}
    cfvg_validation_loaders = {}
    cfvg_train_iterators = {}
    cfvg_validation_iterators = {}
    cfvg_client_class_weights = {}
    if factor_selection_mode == "outcome_alignment":
        outcome_alignment_validation_loader = prepare_tensor_dataloader(
            cen_vali,
            model_name=model,
            max_length=args.max_length,
            progress_prefix="[outcome-alignment] ",
            split_name="server validation",
            batch_size=alignment_validation_batch_size,
            shuffle=False,
        )
        logger.info(
            "Prepared deterministic server-validation loader for outcome alignment: "
            "n=%d, batch_size=%d",
            len(outcome_alignment_validation_loader.dataset),
            alignment_validation_batch_size,
        )
    elif factor_selection_mode == "cmfv":
        cmfv_validation_loader = prepare_tensor_dataloader(
            cen_vali,
            model_name=model,
            max_length=args.max_length,
            progress_prefix="[cmfv] ",
            split_name="server validation",
            batch_size=cmfv_validation_batch_size,
            shuffle=False,
        )
        logger.info(
            "Prepared deterministic server-validation loader for CMFV: "
            "n=%d, batch_size=%d",
            len(cmfv_validation_loader.dataset),
            cmfv_validation_batch_size,
        )
    elif factor_selection_mode == "gi_cmfv":
        gi_cmfv_validation_loader = prepare_tensor_dataloader(
            cen_vali,
            model_name=model,
            max_length=args.max_length,
            progress_prefix="[gi-cmfv] ",
            split_name="server validation",
            batch_size=gi_cmfv_validation_batch_size,
            shuffle=False,
        )
        logger.info(
            "Prepared deterministic server-validation loader for GI-CMFV: "
            "n=%d, batch_size=%d",
            len(gi_cmfv_validation_loader.dataset),
            gi_cmfv_validation_batch_size,
        )
    elif factor_selection_mode == "qgs":
        for client_id in range(args.n_parties):
            client_loader = prepare_tensor_dataloader(
                cli_vali[client_id],
                model_name=model,
                max_length=args.max_length,
                progress_prefix=f"[qgs|client {client_id}] ",
                split_name="client validation",
                batch_size=qgs_validation_batch_size,
                shuffle=False,
            )
            try:
                qgs_validation_batches[int(client_id)] = next(
                    iter(client_loader)
                )
            except StopIteration as exc:
                raise RuntimeError(
                    f"QGS client {client_id} validation split is empty"
                ) from exc
        logger.info(
            "Prepared deterministic QGS validation batches: clients=%s, "
            "batch_size=%d",
            sorted(qgs_validation_batches),
            qgs_validation_batch_size,
        )
    elif factor_selection_mode == "cfvg":
        for client_id in range(args.n_parties):
            cfvg_train_loaders[int(client_id)] = prepare_tensor_dataloader(
                cli_train[client_id],
                model_name=model,
                max_length=args.max_length,
                progress_prefix=f"[cfvg|client {client_id}] ",
                split_name="probe train",
                batch_size=cfvg_probe_train_batch_size,
                shuffle=False,
            )
            cfvg_validation_loaders[int(client_id)] = (
                prepare_tensor_dataloader(
                    cli_vali[client_id],
                    model_name=model,
                    max_length=args.max_length,
                    progress_prefix=f"[cfvg|client {client_id}] ",
                    split_name="probe validation",
                    batch_size=cfvg_probe_validation_batch_size,
                    shuffle=False,
                )
            )
            cfvg_client_class_weights[int(client_id)] = (
                compute_class_weights(
                    cli_train[client_id]["label"],
                    max_positive_weight=3.0,
                )
            )
        logger.info(
            "Prepared rotating deterministic CFVG probe loaders: clients=%s, "
            "train_batch_size=%d, validation_batch_size=%d",
            sorted(cfvg_train_loaders),
            cfvg_probe_train_batch_size,
            cfvg_probe_validation_batch_size,
        )
    if use_mk_aggregation:
        logger.info(
            f"M/K aggregation enabled: M={args.n_parties}, "
            f"|D|={d_total_all_clients}, sample_fraction={args.sample_fraction}"
        )
    # Save arguments
    with open(argument_output_path, "w") as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    server_rng = random.Random(args.init_seed)



    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []

    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(sorted(random.sample(party_list, n_party_per_round)))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)


    logger.info("Initializing nets")
    global_models = init_nets(out_dim, 1,rank,model, dropout_policy=args.jl_dropout, attention=args.jl_attention)
    global_model = global_models[0]
    jl_rank_controller, jl_rank_history, jl_rank_changes = None, [], []
    if args.jl_rank_policy == 'server_validation_feedback':
        jl_rank_controller = ValidationFeedbackRank(
            rank, args.min_rank, args.max_rank, args.jl_rank_step, args.jl_rank_patience,
            args.jl_rank_decrease_patience,
            args.jl_rank_min_delta, args.factor_warmup_rounds)
        for module in global_model.modules():
            if hasattr(module, 'lora_a') and args.max_rank > min(module.lora_a.shape[0], module.lora_b.shape[1]):
                raise ValueError('Flex rank cap exceeds model adapter dimensions')
    from jl_protocol import record_initialization
    record_initialization(global_model, args, log_dir,
                          ('fresh ' + VARIANTS[args.jl_controller_variant]['kind'] + ' controller study'
                           if args.jl_controller_variant else
                           'fresh DDQN, no offline checkpoint' if layer_select_method == 'rl' and rl else None))
    shared_lora_runtime = use_shared_backbone_lora(args)
    if shared_lora_runtime:
        SHARED_BACKBONE_REGISTRY.clear()
        nets = create_lora_client_states(global_model, args.n_parties)
        logger.info("Shared LoRA backbone runtime enabled: one frozen backbone per GPU, per-client LoRA state only.")
        print("Shared LoRA backbone runtime enabled.", flush=True)
    else:
        nets = init_nets(out_dim, args.n_parties,rank,model, dropout_policy=args.jl_dropout, attention=args.jl_attention)
    if gpu_guard is not None:
        gpu_guard.enter_idle(_gpu_guard_devices(args), logger=logger)
    n_comm_rounds = args.comm_round

    net_id = 0

    total_layers = 12
    layer_contribution_scores = np.zeros(total_layers, dtype=float)
    factor_scores, factor_select_counts = init_factor_scores_and_counts(total_layers)
    jl_component = ComponentAblation(args.jl_component_ablation, args.init_seed, args.jl_output)
    adaptive_factor_selection_log = []
    global_factor_selection_log = []
    arc_aslora_selection_log = []
    outcome_alignment_selection_log = []
    cmfv_selection_log = []
    gi_cmfv_selection_log = []
    qgs_selection_log = []
    cfvg_selection_log = []
    cfvg_incumbent_factors = {}
    factor_selector = GlobalAdaptiveFactorSelector(
        epsilon=factor_epsilon,
        beta=factor_score_beta,
        max_staleness=factor_max_staleness,
        reward_normalization=factor_reward_normalization,
        score_clip=factor_score_clip,
        seed=args.init_seed + FACTOR_SELECTOR_SEED_OFFSET,
    )
    arc_aslora_selector = ASLoRAComponentSelector(
        total_layers=total_layers,
        warmup_rounds=factor_warmup_rounds,
        ema_beta=aslora_ema_beta,
        temperature_0=aslora_temperature_0,
        temperature_min=aslora_temperature_min,
        temperature_gamma=aslora_temperature_gamma,
        score_eps=aslora_norm_eps,
        seed=args.init_seed + FACTOR_SELECTOR_SEED_OFFSET + 1,
    )
    outcome_alignment_selector = OutcomeAlignmentFactorSelector(
        total_layers=total_layers,
        min_observations=alignment_min_observations,
        ema_decay=alignment_ema_decay,
        alignment_weight=alignment_local_weight,
        ucb_coefficient=alignment_ucb_coefficient,
        reward_scale_decay=alignment_reward_scale_decay,
        credit_clip=alignment_credit_clip,
    )
    cmfv_selector = CMFVSelector(
        total_layers=total_layers,
        coherence_prior=cmfv_coherence_prior,
        ema_decay=cmfv_coherence_ema_decay,
        count_discount=cmfv_count_discount,
        confidence_z=cmfv_confidence_z,
        min_task_fit=cmfv_min_task_fit,
        exploration_beta=cmfv_exploration_beta,
        posterior_temperature=cmfv_posterior_temperature,
        seed=args.init_seed + FACTOR_SELECTOR_SEED_OFFSET + 2,
    )
    gi_cmfv_selector = GICMFVSelector(
        total_layers=total_layers,
        ema_decay=gi_cmfv_ema_decay,
        count_discount=gi_cmfv_count_discount,
        warmup_rounds=factor_warmup_rounds,
        server_weight=gi_cmfv_server_weight,
        disagreement_penalty=gi_cmfv_disagreement_penalty,
        temperature_0=gi_cmfv_temperature_0,
        temperature_min=gi_cmfv_temperature_min,
        temperature_gamma=gi_cmfv_temperature_gamma,
        seed=args.init_seed + FACTOR_SELECTOR_SEED_OFFSET + 3,
    )
    qgs_selector = QGSSelector(total_layers=total_layers)
    arc_aslora_best = {
        "round": None,
        "validation_macro_f1": float("-inf"),
        "validation_threshold": None,
        "model_state": None,
    }
    if factor_selection_mode == "arc_aslora" and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(torch.device(args.device))
        except (RuntimeError, ValueError) as exc:
            logger.warning("Could not reset CUDA peak-memory stats: %s", exc)
    sparsity_A = {i: torch.tensor(0) for i in range(12)}
    sparsity_B = {i: torch.tensor(0) for i in range(12)}
    imprv_per_layer = None
    initial_num_layers = max(1, min(int(args.initial_num_layers), total_layers))
    # Warm-up: first two rounds train all layers (matches B20A20 best run, t*=48).
    next_round_layer = list(range(total_layers))
    next_round_budget = len(next_round_layer)
    next_round_layer_reason = "warmup_all_layers"
    next_round_layer_random_draw = None
    random_g_schedule = None
    if layer_select_method == "random_g":
        from jl_random_g import make_random_g_schedule
        random_g_schedule = make_random_g_schedule(
            args.init_seed, n_comm_rounds, total_layers, factor_warmup_rounds)
        write_json(Path(log_dir) / "jl_random_g_schedule.json", {
            "seed": args.init_seed, "policy": "uniform integer inclusive 1..12 after warm-up",
            "warmup_rounds": factor_warmup_rounds, "budgets": random_g_schedule,
            "count_rng": "private random.Random(seed), precomputed before training",
        })

    cli_test_acc = {i:None for i in range(12)}
    param_upload = []
    param_caculate = []
    action_size = 12
    agent = None
    jl_study = (ControllerStudy(args.jl_controller_variant, args.init_seed, n_comm_rounds,
                                output=args.jl_output, actions=action_size)
                if args.jl_controller_variant else None)
    if rl and layer_select_method == "rl" and jl_study is None:
        agent = DQNAgent(3, action_size)

        if args.rl_checkpoint_dir:
            ckpt = Path(args.rl_checkpoint_dir)
            q_path = ckpt / "q_network.pth"
            t_path = ckpt / "target_network.pth"
            m_path = ckpt / "memory.pkl"
            if q_path.exists() and t_path.exists() and m_path.exists():
                agent.load_models(str(q_path), str(t_path))
                agent.load_memory(str(m_path))
                agent.epsilon = float(args.online_epsilon_start)
                logger.info(
                    f"Loaded offline RL checkpoint from {ckpt} "
                    f"(memory={len(agent.memory)}, online_epsilon={agent.epsilon})"
                )
                print(
                    f"Loaded offline RL checkpoint: {ckpt} | memory={len(agent.memory)} | "
                    f"epsilon={agent.epsilon}"
                )
                if factor_selection_mode in SCORE_HISTORY_FACTOR_MODES:
                    adaptive_state = load_adaptive_factor_checkpoint(
                        str(ckpt), total_layers=total_layers, logger=logger
                    )
                    if adaptive_state is not None:
                        factor_scores = adaptive_state["factor_scores"]
                        factor_select_counts = adaptive_state["factor_select_counts"]
                        if adaptive_state.get("server_rng_state") is not None:
                            server_rng.setstate(adaptive_state["server_rng_state"])
                        logger.info(
                            "Loaded adaptive factor checkpoint from %s "
                            "(mode=%s, warmup_rounds=%d, factor_lr=%.6f)",
                            ckpt,
                            adaptive_state["factor_selection_mode"],
                            adaptive_state["warmup_rounds"],
                            adaptive_state["factor_lr"],
                        )
                elif factor_selection_mode == "global_adaptive":
                    adaptive_state = load_adaptive_factor_checkpoint(
                        str(ckpt), total_layers=total_layers, logger=logger
                    )
                    if adaptive_state is not None:
                        if adaptive_state.get("factor_selector_state") is not None:
                            factor_selector.load_state_dict(adaptive_state["factor_selector_state"])
                        elif logger is not None:
                            logger.warning(
                                "factor_selector_state missing in %s; using freshly initialized global selector.",
                                ckpt,
                            )
                        if adaptive_state.get("server_rng_state") is not None:
                            server_rng.setstate(adaptive_state["server_rng_state"])
                        logger.info(
                            "Loaded global adaptive factor selector checkpoint from %s",
                            ckpt,
                        )
                elif factor_selection_mode == "arc_aslora":
                    adaptive_state = load_adaptive_factor_checkpoint(
                        str(ckpt), total_layers=total_layers, logger=logger
                    )
                    if adaptive_state is not None and adaptive_state.get("arc_aslora_selector_state"):
                        arc_aslora_selector.load_state_dict(
                            adaptive_state["arc_aslora_selector_state"]
                        )
                        logger.info(
                            "Loaded ARC+AS-LoRA EMA/temperature/factor RNG checkpoint from %s",
                            ckpt,
                        )
                elif factor_selection_mode == "outcome_alignment":
                    adaptive_state = load_adaptive_factor_checkpoint(
                        str(ckpt), total_layers=total_layers, logger=logger
                    )
                    if (
                        adaptive_state is not None
                        and adaptive_state.get("outcome_alignment_selector_state")
                    ):
                        outcome_alignment_selector.load_state_dict(
                            adaptive_state["outcome_alignment_selector_state"]
                        )
                        logger.info(
                            "Loaded outcome-alignment selector checkpoint from %s",
                            ckpt,
                        )
                elif factor_selection_mode == "cmfv":
                    adaptive_state = load_adaptive_factor_checkpoint(
                        str(ckpt), total_layers=total_layers, logger=logger
                    )
                    if (
                        adaptive_state is not None
                        and adaptive_state.get("cmfv_selector_state")
                    ):
                        cmfv_selector.load_state_dict(
                            adaptive_state["cmfv_selector_state"]
                        )
                        logger.info("Loaded CMFV selector checkpoint from %s", ckpt)
                elif factor_selection_mode == "gi_cmfv":
                    adaptive_state = load_adaptive_factor_checkpoint(
                        str(ckpt), total_layers=total_layers, logger=logger
                    )
                    if (
                        adaptive_state is not None
                        and adaptive_state.get("gi_cmfv_selector_state")
                    ):
                        gi_cmfv_selector.load_state_dict(
                            adaptive_state["gi_cmfv_selector_state"]
                        )
                        logger.info(
                            "Loaded GI-CMFV selector checkpoint from %s", ckpt
                        )
                elif factor_selection_mode == "cfvg":
                    adaptive_state = load_adaptive_factor_checkpoint(
                        str(ckpt), total_layers=total_layers, logger=logger
                    )
                    if (
                        adaptive_state is not None
                        and adaptive_state.get("cfvg_incumbent_factors")
                    ):
                        cfvg_incumbent_factors.update(
                            adaptive_state["cfvg_incumbent_factors"]
                        )
                        logger.info(
                            "Loaded CFVG incumbent map from %s", ckpt
                        )
            else:
                raise FileNotFoundError(
                    f"--rl_checkpoint_dir={ckpt} must contain q_network.pth, target_network.pth, memory.pkl"
                )
        elif not args.reset_memory:
            if os.path.exists('q_network.pth') and os.path.exists('target_network.pth') and os.path.exists('memory.pkl'):
                agent.load_models('q_network.pth', 'target_network.pth')
                agent.load_memory('memory.pkl')
                logger.info("Loaded RL models/memory from project root (legacy paths).")
                if factor_selection_mode in SCORE_HISTORY_FACTOR_MODES:
                    adaptive_state = load_adaptive_factor_checkpoint(
                        os.getcwd(), total_layers=total_layers, logger=logger
                    )
                    if adaptive_state is not None:
                        factor_scores = adaptive_state["factor_scores"]
                        factor_select_counts = adaptive_state["factor_select_counts"]
                        if adaptive_state.get("server_rng_state") is not None:
                            server_rng.setstate(adaptive_state["server_rng_state"])
                elif factor_selection_mode == "global_adaptive":
                    adaptive_state = load_adaptive_factor_checkpoint(
                        os.getcwd(), total_layers=total_layers, logger=logger
                    )
                    if adaptive_state is not None:
                        if adaptive_state.get("factor_selector_state") is not None:
                            factor_selector.load_state_dict(adaptive_state["factor_selector_state"])
                        elif logger is not None:
                            logger.warning(
                                "factor_selector_state missing in %s; using freshly initialized global selector.",
                                os.getcwd(),
                            )
                        if adaptive_state.get("server_rng_state") is not None:
                            server_rng.setstate(adaptive_state["server_rng_state"])
                elif factor_selection_mode == "arc_aslora":
                    adaptive_state = load_adaptive_factor_checkpoint(
                        os.getcwd(), total_layers=total_layers, logger=logger
                    )
                    if adaptive_state is not None and adaptive_state.get("arc_aslora_selector_state"):
                        arc_aslora_selector.load_state_dict(
                            adaptive_state["arc_aslora_selector_state"]
                        )
                elif factor_selection_mode == "outcome_alignment":
                    adaptive_state = load_adaptive_factor_checkpoint(
                        os.getcwd(), total_layers=total_layers, logger=logger
                    )
                    if (
                        adaptive_state is not None
                        and adaptive_state.get("outcome_alignment_selector_state")
                    ):
                        outcome_alignment_selector.load_state_dict(
                            adaptive_state["outcome_alignment_selector_state"]
                        )
                elif factor_selection_mode == "cmfv":
                    adaptive_state = load_adaptive_factor_checkpoint(
                        os.getcwd(), total_layers=total_layers, logger=logger
                    )
                    if (
                        adaptive_state is not None
                        and adaptive_state.get("cmfv_selector_state")
                    ):
                        cmfv_selector.load_state_dict(
                            adaptive_state["cmfv_selector_state"]
                        )
                elif factor_selection_mode == "gi_cmfv":
                    adaptive_state = load_adaptive_factor_checkpoint(
                        os.getcwd(), total_layers=total_layers, logger=logger
                    )
                    if (
                        adaptive_state is not None
                        and adaptive_state.get("gi_cmfv_selector_state")
                    ):
                        gi_cmfv_selector.load_state_dict(
                            adaptive_state["gi_cmfv_selector_state"]
                        )
                elif factor_selection_mode == "cfvg":
                    adaptive_state = load_adaptive_factor_checkpoint(
                        os.getcwd(), total_layers=total_layers, logger=logger
                    )
                    if (
                        adaptive_state is not None
                        and adaptive_state.get("cfvg_incumbent_factors")
                    ):
                        cfvg_incumbent_factors.update(
                            adaptive_state["cfvg_incumbent_factors"]
                        )
            else:
                print("RL models or memory not found. Starting from scratch.")
        else:
            print("Resetting RL memory and starting agent from scratch.")

    # if (os.path.exists('memory.pkl')):
    #     agent.load_models('q_network.pth', 'target_network.pth')
    #     agent.load_memory('memory.pkl')
    # ppo = PPO(3, action_size)
    # if(os.path.exists('policy.pth')):
    #     ppo.load_models('policy.pth', 'value.pth')
    trajectory_state = []
    trajectory_action = []
    trajectory_reward = []
    next_state = []
    old_action_probs = []
    dones = []
    convergence_round = []
    cli_train_loss_history = []
    cli_test_loss_history = []
    cli_train_acc_history = []
    cli_test_acc_history = []
    client_train_loss_history = []
    client_test_loss_history = []
    client_train_acc_history = []
    client_test_acc_history = []
    client_round_metrics_history = []
    sever_test_acc_history = []
    server_test_loss_history = []
    server_macro_f1_history = []
    server_threshold_history = []
    factor_switching_log = []
    factor_switch_rounds = []
    trend_slope_state = init_trend_slope_switching_state() if _trend_slope_switching_active else None

    # 记录初始资源使用情况
    max_cpu_usages_history = []
    max_memory_usages_history = []
    avg_cpu_usages_history = []
    avg_memory_usages_history = []
    max_gpu_usages_history = []
    avg_gpu_usages_history = []
    interval = 5
    dqn_loss_history = []
    # 非rl的lora每轮确定更新A还是B，0是更新A，1是更新B
    flag = 0
    all_sparsities = {i: 0 for i in range(12)}
    next_round_matrix = {i: None for i in range(12)}
    training_wall_start = time.perf_counter()
    report_progress(0, n_comm_rounds, 0, 0, label=args.jl_method,
                    note=FINAL_TEST_NOTE, logger=logger)
    logger.info(
        f"RLCD setup: initial_num_layers={initial_num_layers}, "
        f"coordinate_schedule={coordinate_schedule_label}, "
        f"initial_layers={next_round_layer}"
    )
    if _trend_slope_switching_active:
        logger.info(
            f"Trend-slope adaptive factor switching: w={TREND_SLOPE_W}, p={TREND_SLOPE_P}, "
            f"epsilon={TREND_SLOPE_EPS}, m_min={TREND_SLOPE_M_MIN}, m_max={TREND_SLOPE_M_MAX}, "
            f"initial_factor={trend_slope_state['current_factor']}"
        )
        print(
            f"Trend-slope switching active (B start, w={TREND_SLOPE_W}, p={TREND_SLOPE_P}, "
            f"eps={TREND_SLOPE_EPS}, m_min={TREND_SLOPE_M_MIN}, m_max={TREND_SLOPE_M_MAX})"
        )
    for round in range(n_comm_rounds):
        round_wall_start = time.perf_counter()
        emit_progress(f"[训练阶段] {args.jl_method} | 轮次 {round + 1}/{n_comm_rounds} | 准备本轮更新",
                      logger=logger)
        jl_rank_reward_baseline = None
        if jl_rank_controller is not None and jl_rank_controller.rank != rank:
            emit_progress(f"[验证阶段] {args.jl_method} | 变秩后的服务器奖励基线", logger=logger)
            # Isolate expansion initialization from client dropout/sampling RNGs.
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.manual_seed(args.init_seed + 100000 + round)
                jl_resize_details = resize_common_rank(global_model, jl_rank_controller.rank)
            logger.info('JL common rank changes before round %d: %d -> %d',
                        round + 1, rank, jl_rank_controller.rank)
            # Compression changes predictions; do not credit its effect to CD training.
            _gpu_enter_compute(args, logger=logger)
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                _, jl_baseline_loss, jl_baseline_metrics = test(
                    global_model, cen_vali, args, verbose=False, device=args.device,
                    finetune=args.finetune, logger=logger, return_metrics=True,
                    max_length=args.max_length, progress_prefix='[JL rank baseline] ', disable_tqdm=True)
            jl_rank_controller.rebase(jl_baseline_loss)
            jl_rank_reward_baseline = float(jl_baseline_metrics['macro_f1'])
            jl_rank_changes.append(dict(round=round + 1, old_rank=rank,
                                        new_rank=jl_rank_controller.rank,
                                        baseline_server_loss=float(jl_baseline_loss),
                                        baseline_server_macro_f1=jl_rank_reward_baseline,
                                        adapters=jl_resize_details))
            write_json(Path(log_dir) / 'jl_rank_changes.json', jl_rank_changes)
            rank = jl_rank_controller.rank
        next_round_matrix = {i: None for i in range(12)}
        factor_map = {}
        factor_decision_metadata = {}
        selected_budget = 0
        layer_selection_reason = ""
        layer_random_draw = None
        warmup_active = False
        factor_scores_before_round = None
        factor_counts_before_round = None
        arc_aslora_aggregated_scores = None
        outcome_alignment_diagnostics = None
        outcome_alignment_update = None
        cmfv_task_fit = None
        cmfv_coherence_observations = None
        cmfv_coherence_update = None
        cmfv_post_metadata = None
        gi_cmfv_server_scores = None
        gi_cmfv_server_diagnostics = None
        gi_cmfv_client_update = None
        qgs_scores = None
        qgs_diagnostics = None
        cfvg_scores = None
        cfvg_diagnostics = None
        update_sparsities = {}
        frozen_layer, not_com_layer_mark, not_com_layer=[],[],[]
        if round >10 and args.if_sa:
            logger.info("初始阶段结束，进入退火阶段")
            args.if_sa_initial = False
            args.if_sa_anneal = True
        if round >80 and args.if_sa:
            logger.info("退火阶段结束，秩固定")
            args.if_sa_anneal=False
        # cpu_usages = []
        # memory_usages = []
        # gpu_usages = []
        # pid = os.getpid()
        # process = psutil.Process(pid)
        # # 创建一个事件对象来停止记录线程
        # stop_event = threading.Event()
        # # 启动记录线程
        # recording_thread = threading.Thread(target=record_resource_usage, args=(pid,cpu_usages,memory_usages,gpu_usages,interval, stop_event))
        # recording_thread.start()

        param_upload_per_round = 0
        param_caculate_per_round = 0
        logger.info("in comm round:" + str(round))

        party_list_this_round = party_list_rounds[round]

        # Download global model from (virtual) central server
        global_w = global_model.state_dict()
        nets_this_round = {k: nets[k] for k in party_list_this_round}
        if shared_lora_runtime:
            for net in nets_this_round.values():
                sync_lora_client_state_from_global(net, global_model)
        else:
            for net in nets_this_round.values():
                if jl_rank_controller is not None:
                    match_checkpoint_layout(net, global_w, lora_layout(global_model))
                net.load_state_dict(global_w)
        # 每轮初始化先将全局模型以及各个客户端模型解冻
        if(if_lora and (args.if_sa_initial==False)):
            logger.info("冷冻除lora外的所有层")
            global_model = unfrozen(global_model)
            if not shared_lora_runtime:
                nets_this_round = unfrozen(nets_this_round)

        # Train local model with local data
        if (not if_lora) and (finetune == "full"):
            frozen_layer = []
            not_com_layer = []
            not_com_layer_mark = []
            logger.info("FedAvg-Full: train and aggregate all parameters (no RL / no LoRA CD).")
        elif ((rl or cd is not None)):
            global_adaptive_active = (
                rl and factor_selection_mode == "global_adaptive" and prepare_rl_lora_round is None
            )
            adaptive_factor_active = (
                layer_select_method in ADAPTIVE_CD_LAYER_METHODS
                and factor_selection_mode == "adaptive_score"
                and prepare_rl_lora_round is None
            )
            product_delta_active = (
                rl
                and layer_select_method == "rl"
                and factor_selection_mode == "product_delta"
                and prepare_rl_lora_round is None
            )
            arc_aslora_active = (
                rl
                and layer_select_method == "rl"
                and factor_selection_mode == "arc_aslora"
                and prepare_rl_lora_round is None
            )
            outcome_alignment_active = (
                rl
                and layer_select_method == "rl"
                and factor_selection_mode == "outcome_alignment"
                and prepare_rl_lora_round is None
            )
            cmfv_active = (
                rl
                and layer_select_method == "rl"
                and factor_selection_mode == "cmfv"
                and prepare_rl_lora_round is None
            )
            gi_cmfv_active = (
                rl
                and layer_select_method == "rl"
                and factor_selection_mode == "gi_cmfv"
                and prepare_rl_lora_round is None
            )
            qgs_active = (
                rl
                and layer_select_method == "rl"
                and factor_selection_mode == "qgs"
                and prepare_rl_lora_round is None
            )
            cfvg_active = (
                rl
                and layer_select_method == "rl"
                and factor_selection_mode == "cfvg"
                and prepare_rl_lora_round is None
            )
            factor_map = {}
            factor_decision_metadata = {}
            current_round_factor = None
            active_factor = None
            factor_selection_reason = ""
            factor_selector_before = None
            factor_selector_update = None
            warmup_state = resolve_warmup_layer_selection(round, factor_warmup_rounds, total_layers)
            warmup_active = bool(
                round < factor_warmup_rounds and not cmfv_active
            )
            if (
                warmup_active
                and not arc_aslora_active
                and not gi_cmfv_active
                and not product_delta_active
            ):
                next_round_layer = warmup_state["selected_layers"]
                selected_budget = warmup_state["selected_budget"]
                layer_selection_reason = warmup_state["layer_selection_reason"]
                layer_random_draw = warmup_state["layer_random_draw"]
            elif rl:
                # RLCD-FedLoRA: layer set from Double DQN + epsilon-greedy (after warm-up).
                next_round_layer = list(next_round_layer)
                selected_budget = int(next_round_budget)
                layer_selection_reason = next_round_layer_reason
                layer_random_draw = next_round_layer_random_draw
            else:
                next_round_layer = loaded_layers_list[round - 2]
                selected_budget = len(next_round_layer)
                layer_selection_reason = "loaded_layers_list"
                layer_random_draw = None

            if rl and prepare_rl_lora_round is not None:
                frozen_layer, not_com_layer_mark, not_com_layer, next_round_matrix = prepare_rl_lora_round(
                    round, next_round_layer, model, logger
                )
            elif rl:
                if cmfv_active:
                    if cmfv_validation_loader is None:
                        raise RuntimeError("CMFV validation loader was not initialized")
                    _gpu_enter_compute(args, logger=logger)
                    cmfv_task_fit = compute_server_task_fit(
                        model=global_model,
                        validation_loader=cmfv_validation_loader,
                        device=args.device,
                        finetune=args.finetune,
                        learning_rate=factor_lr,
                        curvature_fd_eps=cmfv_curvature_fd_eps,
                        eps=1e-12,
                    )
                    (
                        next_round_layer,
                        factor_map,
                        factor_decision_metadata,
                    ) = cmfv_selector.select_coordinates(
                        task_fit_by_layer=cmfv_task_fit["per_layer"],
                        budget=next_round_budget,
                        round_one_based=round + 1,
                    )
                    selected_budget = len(next_round_layer)
                    layer_selection_reason = "cmfv_value_topk"
                    layer_random_draw = None
                    for layer_idx in range(total_layers):
                        decision = factor_decision_metadata[layer_idx]
                        layer_contribution_scores[layer_idx] = float(
                            decision["layer_value"]
                        )
                        factor_scores[layer_idx] = {
                            "A": float(decision["factor_value_A"]),
                            "B": float(decision["factor_value_B"]),
                        }
                    next_round_matrix = build_round_matrix_from_factor_map(
                        next_round_layer, factor_map, n_layers=12
                    )
                elif gi_cmfv_active:
                    if gi_cmfv_validation_loader is None:
                        raise RuntimeError(
                            "GI-CMFV validation loader was not initialized"
                        )
                    _gpu_enter_compute(args, logger=logger)
                    (
                        gi_cmfv_server_scores,
                        gi_cmfv_server_diagnostics,
                    ) = compute_server_gi_cmfv_scores(
                        model=global_model,
                        validation_loader=gi_cmfv_validation_loader,
                        active_layers=next_round_layer,
                        device=args.device,
                        finetune=args.finetune,
                        rank_rtol=gi_cmfv_rank_rtol,
                    )
                    (
                        factor_map,
                        factor_decision_metadata,
                    ) = gi_cmfv_selector.select_factor_map(
                        active_layers=next_round_layer,
                        server_scores=gi_cmfv_server_scores,
                        round_one_based=round + 1,
                    )
                    for layer_idx in next_round_layer:
                        decision = factor_decision_metadata[int(layer_idx)]
                        factor_scores[int(layer_idx)] = {
                            "A": float(
                                decision["A"]["risk_adjusted_value"]
                            ),
                            "B": float(
                                decision["B"]["risk_adjusted_value"]
                            ),
                        }
                    next_round_matrix = build_round_matrix_from_factor_map(
                        next_round_layer, factor_map, n_layers=12
                    )
                elif qgs_active:
                    selected_validation_batches = {
                        int(client_id): qgs_validation_batches[int(client_id)]
                        for client_id in party_list_this_round
                    }
                    _gpu_enter_compute(args, logger=logger)
                    qgs_scores, qgs_diagnostics = (
                        compute_federated_qgs_scores(
                            model=global_model,
                            validation_batches=selected_validation_batches,
                            active_layers=next_round_layer,
                            device=args.device,
                            finetune=args.finetune,
                            rank_rtol=qgs_rank_rtol,
                        )
                    )
                    factor_map, factor_decision_metadata = (
                        qgs_selector.select_factor_map(
                            active_layers=next_round_layer,
                            scores=qgs_scores,
                        )
                    )
                    for layer_idx in next_round_layer:
                        layer_idx = int(layer_idx)
                        factor_scores[layer_idx] = {
                            "A": float(qgs_scores[layer_idx]["A"]),
                            "B": float(qgs_scores[layer_idx]["B"]),
                        }
                    next_round_matrix = build_round_matrix_from_factor_map(
                        next_round_layer, factor_map, n_layers=12
                    )
                elif cfvg_active:
                    selected_train_batches = {
                        int(client_id): next_cyclic_client_batches(
                            cfvg_train_loaders,
                            cfvg_train_iterators,
                            client_id,
                            cfvg_probe_steps,
                        )
                        for client_id in party_list_this_round
                    }
                    selected_validation_batches = {
                        int(client_id): next_cyclic_client_batches(
                            cfvg_validation_loaders,
                            cfvg_validation_iterators,
                            client_id,
                            cfvg_probe_validation_batches,
                        )
                        for client_id in party_list_this_round
                    }
                    probe_weight_values = compute_fed_avg_weights(
                        nets_this_round,
                        cli_train,
                        args.n_parties,
                        use_mk_correction=use_mk_aggregation,
                        d_total_all=d_total_all_clients,
                    )
                    probe_aggregation_weights = {
                        int(client_id): float(weight)
                        for client_id, weight in zip(
                            nets_this_round.keys(),
                            probe_weight_values,
                        )
                    }
                    _gpu_enter_compute(args, logger=logger)
                    (
                        factor_map,
                        cfvg_scores,
                        cfvg_diagnostics,
                    ) = compute_counterfactual_federated_validation_gains(
                        model=global_model,
                        training_batches=selected_train_batches,
                        validation_batches=selected_validation_batches,
                        client_class_weights={
                            int(client_id): cfvg_client_class_weights[
                                int(client_id)
                            ]
                            for client_id in party_list_this_round
                        },
                        active_layers=next_round_layer,
                        aggregation_weights=probe_aggregation_weights,
                        device=args.device,
                        learning_rate=factor_lr,
                        optimizer_name=args.optimizer,
                        lora_weight_decay=float(args.lora_weight_decay),
                        finetune=args.finetune,
                        probe_steps=cfvg_probe_steps,
                        incumbent_factors=cfvg_incumbent_factors,
                        confidence_z=cfvg_confidence_z,
                    )
                    cfvg_incumbent_factors.update(factor_map)
                    factor_decision_metadata = cfvg_scores
                    for layer_idx in next_round_layer:
                        layer_idx = int(layer_idx)
                        factor_scores[layer_idx] = {
                            "A": float(cfvg_scores[layer_idx]["gain_A"]),
                            "B": float(cfvg_scores[layer_idx]["gain_B"]),
                        }
                    next_round_matrix = build_round_matrix_from_factor_map(
                        next_round_layer, factor_map, n_layers=12
                    )
                elif global_adaptive_active:
                    factor_selector_before = factor_selector.state_dict()
                    if warmup_active:
                        if round == 0:
                            active_factor = "B"
                            factor_selection_reason = "warmup_b_first"
                        else:
                            active_factor = "A"
                            factor_selection_reason = "warmup_a"
                    else:
                        active_factor, factor_selection_reason = factor_selector.select_factor(round)
                    factor_map = build_global_factor_map(next_round_layer, active_factor)
                    current_round_factor = factor_name_to_coordinate_value(active_factor)
                    next_round_matrix = build_round_matrix_from_factor_map(
                        next_round_layer, factor_map, n_layers=12
                    )
                elif product_delta_active:
                    factor_scores_before_round = copy.deepcopy(factor_scores)
                    factor_counts_before_round = copy.deepcopy(factor_select_counts)
                    factor_map, factor_decision_metadata = build_product_delta_factor_map(
                        selected_layers=next_round_layer,
                        round_idx=round,
                        warmup_rounds=factor_warmup_rounds,
                        factor_scores=factor_scores,
                        factor_select_counts=factor_select_counts,
                        rng=server_rng,
                        factor_random_prob=factor_random_prob,
                    )
                    next_round_matrix = build_round_matrix_from_factor_map(
                        next_round_layer, factor_map, n_layers=12
                    )
                elif adaptive_factor_active:
                    factor_scores_before_round = copy.deepcopy(factor_scores)
                    factor_counts_before_round = copy.deepcopy(factor_select_counts)
                    factor_map, factor_decision_metadata = build_factor_map_for_selected_layers(
                        selected_layers=next_round_layer,
                        factor_scores=factor_scores,
                        factor_select_counts=factor_select_counts,
                        rng=server_rng,
                        factor_random_prob=factor_random_prob,
                    )
                    factor_map, factor_decision_metadata = jl_component.factors(
                        (factor_map, factor_decision_metadata), next_round_layer, round + 1)
                    jl_component.record(round + 1, next_round_layer, factor_map,
                                        layer_selection_reason, factor_decision_metadata)
                    next_round_matrix = build_round_matrix_from_factor_map(
                        next_round_layer, factor_map, n_layers=12
                    )
                elif arc_aslora_active:
                    factor_map, factor_decision_metadata = (
                        arc_aslora_selector.select_factor_map(
                            active_layers=next_round_layer,
                            round_one_based=round + 1,
                        )
                    )
                    next_round_matrix = build_round_matrix_from_factor_map(
                        next_round_layer, factor_map, n_layers=12
                    )
                elif outcome_alignment_active:
                    factor_map, factor_decision_metadata = (
                        outcome_alignment_selector.select_factor_map(
                            active_layers=next_round_layer
                        )
                    )
                    next_round_matrix = build_round_matrix_from_factor_map(
                        next_round_layer, factor_map, n_layers=12
                    )
                else:
                    if _trend_slope_switching_active:
                        current_round_factor = factor_name_to_coordinate_value(
                            trend_slope_state["current_factor"]
                        )
                    else:
                        current_round_factor = get_coordinate_factor(round)
                    next_round_matrix = build_round_matrix(next_round_layer, current_round_factor, n_layers=12)
                    if _trend_slope_switching_active:
                        set_trainable_lora_factor(global_model, trend_slope_state["current_factor"])
            elif ((round+1)>change):
                next_round_matrix = {i:0 for i in range(12)}
            else:
                next_round_matrix = {i:1 for i in range(12)}

            logger.info(f'当前轮训练层：{next_round_layer}')
            if rl and prepare_rl_lora_round is not None:
                logger.info(f'LoRA baseline ({LORA_AGG_MODE}): {next_round_matrix}')
            elif rl:
                if cmfv_active:
                    logger.info(
                        "CMFV selection round=%d, selected_budget=%d, "
                        "selected_layers=%s, factor_map=%s, "
                        "num_A_layers=%d, num_B_layers=%d, validation_loss=%.6f",
                        round + 1,
                        selected_budget,
                        next_round_layer,
                        factor_map,
                        sum(1 for factor in factor_map.values() if factor == "A"),
                        sum(1 for factor in factor_map.values() if factor == "B"),
                        cmfv_task_fit["validation_loss"],
                    )
                    logger.info(
                        "CMFV values round=%d: %s",
                        round + 1,
                        {
                            layer_idx: {
                                "A": factor_decision_metadata[layer_idx][
                                    "factor_value_A"
                                ],
                                "B": factor_decision_metadata[layer_idx][
                                    "factor_value_B"
                                ],
                                "layer": factor_decision_metadata[layer_idx][
                                    "layer_value"
                                ],
                                "score": factor_decision_metadata[layer_idx][
                                    "layer_decision_score"
                                ],
                                "pA": factor_decision_metadata[layer_idx][
                                    "probability_A"
                                ],
                                "draw": factor_decision_metadata[layer_idx][
                                    "factor_random_draw"
                                ],
                                "selected_factor": factor_map.get(layer_idx, ""),
                            }
                            for layer_idx in range(total_layers)
                        },
                    )
                elif gi_cmfv_active:
                    logger.info(
                        "GI-CMFV selection round=%d, factor_warmup_active=%s, "
                        "selected_budget=%d, "
                        "selected_layers=%s, layer_selection_reason=%s, "
                        "factor_map=%s, num_A_layers=%d, num_B_layers=%d, "
                        "server_validation_loss=%.6f",
                        round + 1,
                        bool(round < factor_warmup_rounds),
                        selected_budget,
                        next_round_layer,
                        layer_selection_reason,
                        factor_map,
                        sum(
                            1 for factor in factor_map.values()
                            if factor == "A"
                        ),
                        sum(
                            1 for factor in factor_map.values()
                            if factor == "B"
                        ),
                        gi_cmfv_server_diagnostics["validation_loss"],
                    )
                    logger.info(
                        "GI-CMFV values round=%d: %s",
                        round + 1,
                        {
                            layer_idx: {
                                "A_revision": gi_cmfv_server_scores[layer_idx][
                                    "A"
                                ],
                                "B_exploitation": gi_cmfv_server_scores[
                                    layer_idx
                                ]["B"],
                                "A_full": gi_cmfv_server_scores[layer_idx][
                                    "A_full"
                                ],
                                "A_overlap": gi_cmfv_server_scores[layer_idx][
                                    "A_overlap"
                                ],
                                "pA": factor_decision_metadata[layer_idx][
                                    "probability_A"
                                ],
                                "draw": factor_decision_metadata[layer_idx][
                                    "random_draw"
                                ],
                                "selected_factor": factor_map[layer_idx],
                            }
                            for layer_idx in next_round_layer
                        },
                    )
                elif qgs_active:
                    logger.info(
                        "QGS-FS selection round=%d, selected_budget=%d, "
                        "selected_layers=%s, layer_selection_reason=%s, "
                        "participating_clients=%s, factor_map=%s, "
                        "num_A_layers=%d, num_B_layers=%d, "
                        "client_mean_validation_loss=%.6f",
                        round + 1,
                        selected_budget,
                        next_round_layer,
                        layer_selection_reason,
                        qgs_diagnostics["client_ids"],
                        factor_map,
                        sum(
                            1 for factor in factor_map.values()
                            if factor == "A"
                        ),
                        sum(
                            1 for factor in factor_map.values()
                            if factor == "B"
                        ),
                        qgs_diagnostics["validation_loss"],
                    )
                    logger.info(
                        "QGS-FS values round=%d: %s",
                        round + 1,
                        {
                            layer_idx: {
                                "A": qgs_scores[layer_idx]["A"],
                                "B": qgs_scores[layer_idx]["B"],
                                "A_fraction": qgs_scores[layer_idx][
                                    "A_fraction"
                                ],
                                "B_fraction": qgs_scores[layer_idx][
                                    "B_fraction"
                                ],
                                "margin_A_minus_B": (
                                    factor_decision_metadata[layer_idx][
                                        "score_margin_A_minus_B"
                                    ]
                                ),
                                "selected_factor": factor_map[layer_idx],
                            }
                            for layer_idx in next_round_layer
                        },
                    )
                elif cfvg_active:
                    logger.info(
                        "CFVG-FS selection round=%d, selected_budget=%d, "
                        "selected_layers=%s, layer_selection_reason=%s, "
                        "participating_clients=%s, factor_map=%s, "
                        "num_A_layers=%d, num_B_layers=%d, "
                        "base_validation_loss=%.6f, "
                        "selected_map_validation_loss=%.6f, "
                        "predicted_map_gain=%.6f",
                        round + 1,
                        selected_budget,
                        next_round_layer,
                        layer_selection_reason,
                        cfvg_diagnostics["client_ids"],
                        factor_map,
                        sum(
                            1 for factor in factor_map.values()
                            if factor == "A"
                        ),
                        sum(
                            1 for factor in factor_map.values()
                            if factor == "B"
                        ),
                        cfvg_diagnostics["base_validation_loss"],
                        cfvg_diagnostics["selected_map_validation_loss"],
                        cfvg_diagnostics["selected_map_validation_gain"],
                    )
                    logger.info(
                        "CFVG-FS values round=%d: %s",
                        round + 1,
                        {
                            layer_idx: {
                                "gain_A": cfvg_scores[layer_idx]["gain_A"],
                                "gain_B": cfvg_scores[layer_idx]["gain_B"],
                                "loss_A": cfvg_scores[layer_idx][
                                    "candidate_loss_A"
                                ],
                                "loss_B": cfvg_scores[layer_idx][
                                    "candidate_loss_B"
                                ],
                                "update_norm_A": cfvg_scores[layer_idx][
                                    "update_norm_A"
                                ],
                                "update_norm_B": cfvg_scores[layer_idx][
                                    "update_norm_B"
                                ],
                                "paired_advantage_A": cfvg_scores[layer_idx][
                                    "paired_advantage_A"
                                ],
                                "paired_standard_error": cfvg_scores[
                                    layer_idx
                                ]["paired_standard_error"],
                                "confidence_decisive": cfvg_scores[layer_idx][
                                    "confidence_decisive"
                                ],
                                "actionability_decisive": cfvg_scores[
                                    layer_idx
                                ]["actionability_decisive"],
                                "actionability_override": cfvg_scores[
                                    layer_idx
                                ]["actionability_override"],
                                "incumbent_factor": cfvg_scores[layer_idx][
                                    "incumbent_factor"
                                ],
                                "decision_reason": cfvg_scores[layer_idx][
                                    "decision_reason"
                                ],
                                "selected_factor": factor_map[layer_idx],
                            }
                            for layer_idx in next_round_layer
                        },
                    )
                elif global_adaptive_active:
                    logger.info(
                        "Global adaptive factor selection round=%d, factor_selection_mode=%s, "
                        "warmup_active=%s, active_factor=%s, factor_selection_reason=%s, "
                        "factor_score_a_before=%.6f, factor_score_b_before=%.6f, "
                        "factor_count_a=%d, factor_count_b=%d, factor_last_selected_a=%s, "
                        "factor_last_selected_b=%s, factor_epsilon=%.2f, factor_score_beta=%.2f, "
                        "num_selected_layers=%d, selected_layer_ids=%s",
                        round + 1,
                        factor_selection_mode,
                        warmup_active,
                        active_factor,
                        factor_selection_reason,
                        factor_selector_before["score_a"] if factor_selector_before is not None else factor_selector.score_a,
                        factor_selector_before["score_b"] if factor_selector_before is not None else factor_selector.score_b,
                        factor_selector_before["count_a"] if factor_selector_before is not None else factor_selector.count_a,
                        factor_selector_before["count_b"] if factor_selector_before is not None else factor_selector.count_b,
                        factor_selector_before["last_selected_a"] if factor_selector_before is not None else factor_selector.last_selected_a,
                        factor_selector_before["last_selected_b"] if factor_selector_before is not None else factor_selector.last_selected_b,
                        factor_selector.epsilon,
                        factor_selector.beta,
                        len(next_round_layer),
                        next_round_layer,
                    )
                elif product_delta_active:
                    logger.info(
                        "ARC product-delta selection round=%d, warmup_active=%s, "
                        "selected_budget=%d, selected_layers=%s, "
                        "layer_selection_reason=%s, layer_random_draw=%s, "
                        "factor_map=%s, num_A_layers=%d, num_B_layers=%d, "
                        "post_warmup_factor_random_prob=%.2f",
                        round + 1,
                        warmup_active,
                        selected_budget,
                        next_round_layer,
                        layer_selection_reason,
                        layer_random_draw,
                        factor_map,
                        sum(1 for factor in factor_map.values() if factor == "A"),
                        sum(1 for factor in factor_map.values() if factor == "B"),
                        factor_random_prob,
                    )
                elif adaptive_factor_active:
                    logger.info(
                        "Adaptive factor selection round=%d, warmup_active=%s, selected_budget=%d, "
                        "selected_layers=%s, layer_selection_reason=%s, layer_random_draw=%s, "
                        "factor_map=%s, num_A_layers=%d, num_B_layers=%d, factor_random_prob=%.2f",
                        round + 1,
                        warmup_active,
                        selected_budget,
                        next_round_layer,
                        layer_selection_reason,
                        layer_random_draw,
                        factor_map,
                        sum(1 for factor in factor_map.values() if factor == "A"),
                        sum(1 for factor in factor_map.values() if factor == "B"),
                        factor_random_prob,
                    )
                elif arc_aslora_active:
                    logger.info(
                        "ARC+AS-LoRA selection round=%d, warmup_active=%s, selected_budget=%d, "
                        "selected_layers=%s, layer_selection_reason=%s, factor_map=%s, "
                        "temperature=%.6f",
                        round + 1,
                        warmup_active,
                        selected_budget,
                        next_round_layer,
                        layer_selection_reason,
                        factor_map,
                        arc_aslora_selector.last_temperature,
                    )
                elif outcome_alignment_active:
                    logger.info(
                        "Outcome-alignment selection round=%d, warmup_active=%s, "
                        "selected_budget=%d, selected_layers=%s, "
                        "layer_selection_reason=%s, factor_map=%s, "
                        "num_A_layers=%d, num_B_layers=%d",
                        round + 1,
                        warmup_active,
                        selected_budget,
                        next_round_layer,
                        layer_selection_reason,
                        factor_map,
                        sum(1 for factor in factor_map.values() if factor == "A"),
                        sum(1 for factor in factor_map.values() if factor == "B"),
                    )
                else:
                    logger.info(
                        f'当前轮统一更新因子：{factor_name(current_round_factor)} '
                        f'(schedule={coordinate_schedule_label})'
                    )
                logger.info(f'当前轮训练矩阵：{next_round_matrix}')
            else:
                logger.info(f'当前轮训练矩阵：{next_round_matrix}')
            if not (rl and prepare_rl_lora_round is not None):
                frozen_layer, not_com_layer_mark, not_com_layer = prepare_round_factor_masking(
                    next_round_layer, next_round_matrix, model, rl_active=rl
                )
        # 不使用强化学习的lora每轮只交换A或B
        # elif if_lora==True:
        #     next_round_layer = [i for i in range(12)]
        #     next_round_matrix = {i: flag for i in range(12)}
        #     frozen_layer,frozen_layer_mark = generate_frozen_layer(next_round_layer,next_round_matrix,model)
        #     logger.info(f'当前轮训练层：{next_round_matrix}')
        #     logger.info(f'当前轮冻结层：{frozen_layer_mark}')
        #     flag = 1-flag
        elif args.if_ffa == True:
            next_round_layer = [i for i in range(12)]
            next_round_matrix = {i: 1 for i in range(12)}
            frozen_layer, not_com_layer_mark, not_com_layer = generate_frozen_layer(next_round_layer, next_round_matrix,
                                                                                    model)
            frozen_layer = []
            for i in range(12):
                if model == 'roberta':
                    frozen_layer.append('roberta.encoder.layer.' + str(i) + '.attention.self.key.lora_a')
                    frozen_layer.append('roberta.encoder.layer.' + str(i) + '.attention.self.value.lora_a')
                elif model == 'gpt':
                    frozen_layer.append('transformer.h'+ str(i) + '.attn.c_attn.k_proj.lora_a')
                    frozen_layer.append('transformer.h'+ str(i) + '.attn.c_attn.v_proj.lora_a')
        elif args.if_sa ==True and args.if_sa_initial == True:
            # 初始阶段全参与训练
            frozen_layer=[]
            not_com_layer_mark = []
            not_com_layer = []
        elif args.if_sa ==True and args.if_sa_anneal == True:
            # 退火阶段只训练lora
            next_round_layer = [i for i in range(12)]
            # next_round_matrix = {i: flag for i in range(12)}
            frozen_layer, not_com_layer_mark, not_com_layer = generate_frozen_layer(next_round_layer, next_round_matrix,
                                                                                    model)
        else:

            next_round_layer = [i for i in range(12)]
            # next_round_matrix = {i: flag for i in range(12)}
            frozen_layer, not_com_layer_mark,not_com_layer = generate_frozen_layer(next_round_layer,next_round_matrix,model)
            flag = 1 - flag
        if (not if_lora) and (finetune == "full"):
            pass  # already logged above
        else:
            logger.info(f'当前轮不交换的层标记：{not_com_layer_mark}')
            logger.info(f'当前轮不交换的层：{not_com_layer}')
            logger.info(f'当前轮冻结层：{frozen_layer}')
        _gpu_enter_compute(args, logger=logger)
        local_result = local_train_net(
            nets_this_round,
            args,
            cli_train,
            cli_vali,
            cli_test,
            frozen_layer,
            not_com_layer,
            round_idx=round,
            total_rounds=n_comm_rounds,
            shared_source_model=global_model if shared_lora_runtime else None,
            as_lora_active_layers=(next_round_layer if factor_selection_mode == "arc_aslora" else None),
            gi_cmfv_active_layers=(
                next_round_layer
                if factor_selection_mode == "gi_cmfv"
                else None
            ),
        )
        if factor_selection_mode in {"arc_aslora", "gi_cmfv"}:
            (
                net_params,
                fed_train_loss,
                fed_test_loss,
                fed_train_acc,
                fed_test_acc,
                client_metrics,
                arc_aslora_client_scores,
                arc_aslora_client_diagnostics,
            ) = local_result
            if factor_selection_mode == "arc_aslora":
                arc_aslora_aggregated_scores = (
                    arc_aslora_selector.update_from_clients(
                        arc_aslora_client_scores,
                        active_layers=next_round_layer,
                    )
                )
                logger.info(
                    "ARC+AS-LoRA uniform score aggregation round=%d, clients=%s, raw_scores=%s, "
                    "ema_scores=%s, diagnostics=%s",
                    round + 1,
                    sorted(arc_aslora_client_scores),
                    arc_aslora_aggregated_scores,
                    {
                        layer_id: arc_aslora_selector.ema_scores[layer_id]
                        for layer_id in next_round_layer
                    },
                    arc_aslora_client_diagnostics,
                )
            else:
                gi_cmfv_client_update = (
                    gi_cmfv_selector.update_from_clients(
                        client_scores=arc_aslora_client_scores,
                        active_layers=next_round_layer,
                        round_one_based=round + 1,
                    )
                )
                logger.info(
                    "GI-CMFV uniform counterfactual evidence aggregation "
                    "round=%d, clients=%s, update=%s, diagnostics=%s",
                    round + 1,
                    sorted(arc_aslora_client_scores),
                    gi_cmfv_client_update,
                    arc_aslora_client_diagnostics,
                )
        else:
            (
                net_params,
                fed_train_loss,
                fed_test_loss,
                fed_train_acc,
                fed_test_acc,
                client_metrics,
            ) = local_result
        cli_train_loss_history.append(fed_train_loss)
        cli_test_loss_history.append(fed_test_loss)
        cli_test_acc_history.append(fed_test_acc)
        cli_train_acc_history.append(fed_train_acc)
        train_loss_row = np.full(args.n_parties, np.nan)
        test_loss_row = np.full(args.n_parties, np.nan)
        train_acc_row = np.full(args.n_parties, np.nan)
        test_acc_row = np.full(args.n_parties, np.nan)
        for client_id, metrics in client_metrics.items():
            train_loss_row[client_id] = metrics["train_loss"]
            test_loss_row[client_id] = metrics["test_loss"]
            train_acc_row[client_id] = metrics["train_acc"]
            test_acc_row[client_id] = metrics["test_acc"]
            client_round_metrics_history.append({
                "round": round + 1,
                "client_id": int(client_id),
                "client_name": client_names[client_id] if client_id < len(client_names) else f"client_{client_id}",
                "train_loss": float(metrics["train_loss"]),
                "train_acc": float(metrics["train_acc"]),
                "val_threshold": float(metrics["val_threshold"]),
                "val_macro_f1": float(metrics["val_macro_f1"]),
                "test_loss": float(metrics["test_loss"]),
                "test_acc": float(metrics["test_acc"]),
                "test_macro_f1": float(metrics["test_macro_f1"]),
                "test_balanced_acc": float(metrics["test_balanced_acc"]),
                "test_macro_precision": float(metrics["test_macro_precision"]),
                "test_macro_recall": float(metrics["test_macro_recall"]),
                "test_macro_auc": float(metrics["test_macro_auc"]),
                "test_precision_cls1": float(metrics["test_precision_cls1"]),
                "test_recall_cls1": float(metrics["test_recall_cls1"]),
                "test_f1_cls1": float(metrics["test_f1_cls1"]),
                "test_evaluated": bool(metrics.get("test_evaluated", True)),
            })
        client_train_loss_history.append(train_loss_row)
        client_test_loss_history.append(test_loss_row)
        client_train_acc_history.append(train_acc_row)
        client_test_acc_history.append(test_acc_row)
        write_csv(Path(log_dir) / 'jl_local_train_metrics.csv', [
            {k: r[k] for k in ('round', 'client_id', 'client_name', 'train_loss', 'train_acc')}
            for r in client_round_metrics_history
        ])

        global_model = freeze_layer(global_model, frozen_layer) if if_lora else global_model
        emit_progress(f"[聚合阶段] {args.jl_method} | 轮次 {round + 1}/{n_comm_rounds}", logger=logger)
        prev_global_lora_state = clone_lora_factor_state(global_model) if if_lora else None
        # for name, param in global_model.named_parameters():
        #     if name == 'roberta.encoder.layer.0.attention.self.key.lora_a' or name == 'roberta.encoder.layer.11.attention.self.value.lora_a':
        #         print(name, param.requires_grad)
        total_data_points = sum([len(cli_train[net_id]) for net_id, net in nets_this_round.items()])
        fed_avg_freqs = compute_fed_avg_weights(
            nets_this_round,
            cli_train,
            args.n_parties,
            use_mk_correction=use_mk_aggregation,
            d_total_all=d_total_all_clients,
        )
        logger.info(
            f"Aggregation weights (selected clients {list(nets_this_round.keys())}): "
            f"{[f'{w:.6f}' for w in fed_avg_freqs]} (sum={sum(fed_avg_freqs):.6f})"
        )
        if factor_selection_mode == "cmfv" and factor_map:
            validate_factor_upload_consistency(net_params, factor_map, model)
            cmfv_coherence_observations = compute_client_factor_coherence(
                client_payloads=net_params,
                state_before=prev_global_lora_state,
                factor_map=factor_map,
                aggregation_weights=fed_avg_freqs,
                eps=1e-12,
            )
            logger.info(
                "CMFV observed effective-space coherence round=%d: %s",
                round + 1,
                cmfv_coherence_observations,
            )
        aggregated_params = {}
        aggregated_grad={}
        if aggregate_lora_server is not None:
            aggregated_params, aggregated_grad = aggregate_lora_server(
                net_params,
                fed_avg_freqs,
                global_model=global_model,
                next_round_layer=next_round_layer,
                model_name=model,
                rank=rank,
                lora_alpha=8,
            )
            n_upload_clients = len(net_params)
            for name, param in aggregated_params.items():
                param_caculate_per_round += param.numel()
                # Each selected client uploads its local delta (K uplinks + K downlinks).
                param_upload_per_round += param.numel() * n_upload_clients
        else:
            if factor_selection_mode in {
                "adaptive_score",
                "product_delta",
                "arc_aslora",
                "outcome_alignment",
                "cmfv",
                "gi_cmfv",
                "qgs",
                "cfvg",
            } and factor_map:
                validate_factor_upload_consistency(net_params, factor_map, model)
            # M/K weights need not sum to one; keep the shared starting
            # factors unchanged and weight only the local updates.
            logger.info(
                "Parameter aggregation round=%d: global_base_plus_weighted_client_deltas",
                round + 1,
            )
            (
                aggregated_params,
                aggregated_grad,
                agg_param_calculate,
                agg_param_upload,
            ) = aggregate_uploaded_deltas(
                net_params,
                fed_avg_freqs,
                prev_global_lora_state if if_lora else global_model.state_dict(),
            )
            param_caculate_per_round += agg_param_calculate
            param_upload_per_round += agg_param_upload
        if factor_selection_mode in {"arc_aslora", "gi_cmfv"}:
            # Two scalar scores (A and B) per active layer and participating client.
            param_upload_per_round += 2 * len(next_round_layer) * len(nets_this_round)
        param_upload.append(param_upload_per_round)
        del net_params

        # 强化学习参数准备
        update_A = {}
        update_B = {}
        # 当前轮A矩阵需要更新的层数
        G_A = set()
        # 当前轮B矩阵需要更新的层数
        G_B = set()
        sparsities = []

        if if_lora:
            for name, param in global_model.named_parameters():
                # print(f'params device: {param.device}')
                # 插入lora_a和lora_b的有两个矩阵key、value，要把这俩加一起才能算一个lora_a的变化量

                if name in aggregated_params:
                    # 如果强化学习，在计算a、b的提升量，否则可省略
                    if(rl):
                        layer_number = int(name.split('.')[-5])
                        if (name.split('.')[-1] == 'lora_a'):
                            # 表示当前需要更新的层数及A矩阵，即对应imprv的索引
                            # G_A = [(0,0),(1,0)....,(11,0)]
                            G_A.add((layer_number, 0))
                            if (layer_number in update_A.keys()):
                                # value中的lora_a矩阵变化量
                                update_A[layer_number] += torch.norm(param.data - aggregated_params[name], 2)
                                grad_a = aggregated_grad.get(name)
                                if grad_a is not None:
                                    sparsity_A[layer_number] += grad_a
                                param_caculate_per_round+= param.numel()
                            else:
                                # key中的lora_a矩阵变化量
                                update_A[layer_number] = torch.norm(param.data - aggregated_params[name], 2)
                                grad_a = aggregated_grad.get(name)
                                if grad_a is not None:
                                    sparsity_A[layer_number] = torch.zeros_like(grad_a)
                                    sparsity_A[layer_number] += grad_a
                                else:
                                    sparsity_A[layer_number] = torch.tensor(0.0)
                                param_caculate_per_round += param.numel()
                        if (name.split('.')[-1] == 'lora_b'):
                            G_B.add((layer_number, 1))
                            grad_b = aggregated_grad.get(name)
                            if (layer_number in update_B.keys()):
                                param_caculate_per_round += param.numel()
                                if grad_b is not None:
                                    sparsity_B[layer_number] += grad_b
                                update_B[layer_number] += torch.norm(param.data - aggregated_params[name], 2)
                            else:
                                update_B[layer_number] = torch.norm(param.data - aggregated_params[name], 2)
                                sparsity_B[layer_number] = grad_b if grad_b is not None else torch.tensor(0.0)
                                param_caculate_per_round += param.numel()
                    else:
                        layer_number = int(name.split('.')[-5])
                        if (name.split('.')[-1] == 'lora_a'):
                            # 表示当前需要更新的层数及A矩阵，即对应imprv的索引
                            # G_A = [(0,0),(1,0)....,(11,0)]
                            G_A.add((layer_number, 0))
                        else:
                            G_B.add((layer_number, 1))
                    # 替换服务器参数
                    param.data = aggregated_params[name].data.clone()
                elif name not in aggregated_params.keys() and( name.split('.')[-2] == 'key' or name.split('.')[-2] == 'value' ):
                    layer_number = int(name.split('.')[-5])
                    if name.split('.')[-1] == 'lora_a':
                        update_A[layer_number] = 0
                    else:
                        update_B[layer_number] = 0
        else:
            if hasattr(global_model, "backbone_model"):
                global_param_iter = chain(
                    global_model.named_parameters(), global_model.backbone_model.named_parameters()
                )
            else:
                global_param_iter = global_model.named_parameters()
            for name, param in global_param_iter:
                param.to('cpu')
                if name in aggregated_params:
                    param.data = aggregated_params[name].data.clone()
                    param_caculate_per_round += param.numel()
        if(rl or cd !=None):
            for name,grad in sparsity_A.items():
                if isinstance(grad,torch.Tensor):
                    sparsity_A[name] = torch.sum(torch.abs(grad)).item()
                else:
                    sparsity_A[name] = grad
            for name,grad in sparsity_B.items():
                if isinstance(grad,torch.Tensor):
                    sparsity_B[name] = torch.sum(torch.abs(grad)).item()
                else:
                    sparsity_B[name] = grad
            print(f"sparsity_A:{sparsity_A}")
            print(f"sparsity_B:{sparsity_B}")
            # print(G_A+G_B)
            # Evaluating the global model
            for name,grad in aggregated_grad.items():
                if grad is not None:
                    layer_number = int(name.split('.')[-5])
                    sparsity = compute_sparsity(grad)
                    sparsities.append(sparsity)
                    if layer_number not in update_sparsities.keys():
                        update_sparsities[layer_number] =abs(grad.sum().item())
                    else:
                        update_sparsities[layer_number] +=abs(grad.sum().item())

            sparsities=torch.tensor(sparsities)
            all_sparsities.update(update_sparsities)
            logger.info(f"sparsities:{sparsities}")
            logger.info(f"all_sparsities:{all_sparsities}")
            mean = torch.mean(sparsities)
            var = torch.var(sparsities)
            max_val = torch.max(sparsities)
            min_val = torch.min(sparsities)

        logger.info(f"当前需要更新的层：{G_A | G_B}")
        _gpu_enter_compute(args, logger=logger)
        srv_prefix = _fed_progress_prefix(round, n_comm_rounds, tag="全局验证集")
        emit_progress(f"[验证阶段] {args.jl_method} | {srv_prefix.strip()} | server_vali 奖励评估",
                      logger=logger)
        test_acc,test_loss,server_metrics = test(
            global_model, cen_vali, args, verbose=True,
            device=args.device, finetune=args.finetune, logger=logger,
            progress_prefix=srv_prefix,
            disable_tqdm=True,
            return_metrics=True,
            max_length=args.max_length,
        )
        logger.info(f'{srv_prefix.strip()}本轮结束后全局验证集测试精度（server vali）：{test_acc}')
        logger.info(
            f'{srv_prefix.strip()}本轮结束后全局验证集 macro-F1：{server_metrics["macro_f1"]}，'
            f'阈值：{server_metrics["threshold"]}'
        )
        if _trend_slope_switching_active:
            (
                switch_event,
                trend_slope,
                factor_used,
                saturation_counter,
                switch_reason,
                factor_rounds,
            ) = update_trend_slope_switching_state(trend_slope_state, server_metrics["macro_f1"])
            slope_str = "None" if trend_slope is None else f"{trend_slope:.6f}"
            logger.info(
                f"[trend_slope_switching] round={round + 1}, factor={factor_used}, "
                f"factor_rounds={factor_rounds}, macro_f1={server_metrics['macro_f1']:.6f}, "
                f"trend_slope={slope_str}, saturation_counter={saturation_counter}, "
                f"switch_event={switch_event}, switch_reason={switch_reason}"
            )
            if switch_event:
                logger.info(
                    f"[trend_slope_switching] factor switched at round {round + 1} "
                    f"({switch_reason}): {factor_used} -> {trend_slope_state['current_factor']}"
                )
                factor_switch_rounds.append(round + 1)
                set_trainable_lora_factor(global_model, trend_slope_state["current_factor"])
            factor_switching_log.append({
                "round": round + 1,
                "current_factor": factor_used,
                "factor_rounds": factor_rounds,
                "macro_f1": float(server_metrics["macro_f1"]),
                "trend_slope": trend_slope,
                "saturation_counter": saturation_counter,
                "switch_event": switch_event,
                "switch_reason": switch_reason or "",
            })
        sever_test_acc_history.append(test_acc)
        server_test_loss_history.append(test_loss)
        server_macro_f1_history.append(server_metrics["macro_f1"])
        server_threshold_history.append(server_metrics["threshold"])
        if jl_rank_controller is not None:
            jl_rank_event = jl_rank_controller.observe(round + 1, test_loss)
            jl_rank_history.append(jl_rank_event)
            write_csv(Path(log_dir) / 'jl_rank_history.csv', jl_rank_history)
        jl_record = jl_protocol.validate(
            global_model, round + 1,
            selected_clients=json.dumps(party_list_this_round),
            selected_layers=json.dumps(next_round_layer),
            G=len(next_round_layer) if if_lora else 12,
            factors=json.dumps(factor_map or next_round_matrix, sort_keys=True),
            aggregation_weight_sum=float(sum(fed_avg_freqs)),
            uplink_parameters=int(param_upload_per_round),
            cumulative_uplink_bytes_fp32=int(sum(param_upload) * 4),
            server_reward_macro_f1=float(server_metrics['macro_f1']),
            server_reward_loss=float(test_loss),
            rank_change_reward_baseline=jl_rank_reward_baseline,
            rank_policy=args.jl_rank_policy,
            proposed_next_rank=jl_rank_controller.rank if jl_rank_controller is not None else rank,
            controller_epsilon=(jl_study.epsilon if jl_study is not None else
                                float(agent.epsilon) if agent is not None else None),
            training_aggregation_reward_seconds=time.perf_counter() - round_wall_start,
        )

        if(len(sever_test_acc_history) >= 2 and abs(sever_test_acc_history[-1]-sever_test_acc_history[-2])<0.01):
            convergence_round.append(round)
        # 开始强化学习 / 贡献分数更新（论文 Step 7）
        if factor_selection_mode in {
            "adaptive_score",
            "product_delta",
            "global_adaptive",
            "arc_aslora",
            "outcome_alignment",
            "cmfv",
            "gi_cmfv",
            "qgs",
            "cfvg",
        } and factor_map:
            update_factor_select_counts(factor_select_counts, factor_map, aggregated_params, model)
            selected_layer_ids = sorted(int(layer_idx) for layer_idx in factor_map.keys())
            q, adaptive_layer_delta_norms = compute_selected_factor_delta_norms(
                prev_global_lora_state,
                global_model,
                factor_map,
                model,
            )
        else:
            selected_layer_ids = sorted({layer_idx for layer_idx, _ in (G_A | G_B)})
            q = sum(list(update_A.values()) + list(update_B.values()))
            adaptive_layer_delta_norms = {}
        selected_layer_count = max(1, len(selected_layer_ids))
        param_caculate_per_round += len(update_A.values()) + len(update_B.values())
        reward_t = (
            server_macro_f1_history[-1] - server_macro_f1_history[-2]
            if len(server_macro_f1_history) >= 2
            else 0.0
        )
        if jl_rank_reward_baseline is not None:
            reward_t = float(server_metrics['macro_f1']) - jl_rank_reward_baseline
        # Paper: c_{t,l} = r_t / G_t if l in L_t; else retain previous score.
        if len(server_macro_f1_history) >= 2 and selected_layer_ids:
            current_layer_contribution = reward_t / selected_layer_count
            for layer_idx in selected_layer_ids:
                layer_contribution_scores[layer_idx] = current_layer_contribution
        factor_credit_result = None
        if factor_selection_mode == "adaptive_score" and factor_map:
            if factor_credit_mode == "rms":
                global_lora_state_after = clone_lora_factor_state(global_model)
                factor_credit_result = assign_rms_weighted_factor_credit(
                    reward=reward_t,
                    selected_layers=selected_layer_ids,
                    factor_map=factor_map,
                    global_state_before=prev_global_lora_state,
                    global_state_after=global_lora_state_after,
                    model_name=model,
                    eps=1e-12,
                )
            else:
                factor_credit_result = assign_equal_factor_credit(
                    reward=reward_t,
                    selected_layers=selected_layer_ids,
                )
            for layer_idx in selected_layer_ids:
                selected_factor = factor_map[int(layer_idx)]
                old_score = float(factor_scores[int(layer_idx)][selected_factor])
                factor_scores[int(layer_idx)][selected_factor] = (
                    (1.0 - factor_score_beta) * old_score
                    + factor_score_beta * factor_credit_result["credits"][int(layer_idx)]
                )
        elif factor_selection_mode == "product_delta" and factor_map:
            global_lora_state_after = clone_lora_factor_state(global_model)
            factor_credit_result = assign_product_delta_weighted_credit(
                reward=reward_t,
                selected_layers=selected_layer_ids,
                global_state_before=prev_global_lora_state,
                global_state_after=global_lora_state_after,
                eps=1e-12,
            )
            for layer_idx in selected_layer_ids:
                layer_idx = int(layer_idx)
                selected_factor = factor_map[layer_idx]
                assigned_credit = factor_credit_result["credits"][layer_idx]
                layer_contribution_scores[layer_idx] = float(assigned_credit)
                old_score = float(factor_scores[layer_idx][selected_factor])
                factor_scores[layer_idx][selected_factor] = (
                    (1.0 - factor_score_beta) * old_score
                    + factor_score_beta * assigned_credit
                )
            logger.info(
                "ARC product-delta reward round=%d, reward=%.6f, "
                "effective_product_deltas=%s, credit_weights=%s, "
                "assigned_credits=%s, fallback_equal=%s",
                round + 1,
                reward_t,
                factor_credit_result["product_deltas"],
                factor_credit_result["weights"],
                factor_credit_result["credits"],
                factor_credit_result["fallback_used"],
            )
        elif factor_selection_mode == "global_adaptive" and factor_map:
            factor_selector_update = factor_selector.update(
                factor=active_factor,
                reward=reward_t,
                num_selected_layers=selected_layer_count,
                round_idx=round,
            )
        elif factor_selection_mode == "outcome_alignment" and factor_map:
            if outcome_alignment_validation_loader is None:
                raise RuntimeError(
                    "Outcome-alignment validation loader was not initialized"
                )
            _gpu_enter_compute(args, logger=logger)
            outcome_alignment_diagnostics = compute_server_validation_alignment(
                model=global_model,
                validation_loader=outcome_alignment_validation_loader,
                state_before=prev_global_lora_state,
                factor_map=factor_map,
                device=args.device,
                finetune=args.finetune,
                eps=1e-12,
            )
            outcome_alignment_update = outcome_alignment_selector.update(
                factor_map=factor_map,
                alignment_diagnostics=outcome_alignment_diagnostics["per_layer"],
                validation_reward=reward_t,
            )
            for layer_idx, layer_update in outcome_alignment_update["layers"].items():
                layer_contribution_scores[int(layer_idx)] = float(
                    layer_update["utility_after"]
                )
            for layer_idx in range(total_layers):
                factor_scores[layer_idx] = {
                    factor: float(
                        outcome_alignment_selector.utilities[layer_idx][factor]
                    )
                    for factor in ("A", "B")
                }
            logger.info(
                "Outcome-alignment feedback round=%d, reward=%.6f, "
                "normalized_reward=%.6f, gradient_validation_loss=%.6f, "
                "factor_map=%s, per_layer=%s",
                round + 1,
                reward_t,
                outcome_alignment_update["normalized_reward"],
                outcome_alignment_diagnostics["validation_loss"],
                factor_map,
                outcome_alignment_diagnostics["per_layer"],
            )
        elif factor_selection_mode == "cmfv" and factor_map:
            if cmfv_coherence_observations is None or cmfv_task_fit is None:
                raise RuntimeError(
                    "CMFV requires task-fit diagnostics and client coherence observations"
                )
            cmfv_coherence_update = cmfv_selector.update_coherence(
                factor_map=factor_map,
                observations=cmfv_coherence_observations,
                round_one_based=round + 1,
            )
            _, _, cmfv_post_metadata = cmfv_selector.select_coordinates(
                task_fit_by_layer=cmfv_task_fit["per_layer"],
                budget=total_layers,
                round_one_based=round + 1,
            )
            for layer_idx in range(total_layers):
                post_decision = cmfv_post_metadata[layer_idx]
                layer_contribution_scores[layer_idx] = float(
                    post_decision["layer_value"]
                )
                factor_scores[layer_idx] = {
                    "A": float(post_decision["factor_value_A"]),
                    "B": float(post_decision["factor_value_B"]),
                }
            logger.info(
                "CMFV feedback round=%d, validation_reward=%.6f, "
                "coherence_updates=%s",
                round + 1,
                reward_t,
                cmfv_coherence_update,
            )
        elif factor_selection_mode == "gi_cmfv" and factor_map:
            if (
                gi_cmfv_client_update is None
                or gi_cmfv_server_scores is None
            ):
                raise RuntimeError(
                    "GI-CMFV requires server task fit and dual-factor "
                    "client validation evidence"
                )
            logger.info(
                "GI-CMFV feedback round=%d, validation_reward=%.6f, "
                "client_counterfactual_update=%s",
                round + 1,
                reward_t,
                gi_cmfv_client_update,
            )
        imprv_per_layer = np.asarray(layer_contribution_scores, dtype=float)
        logger.info(f'imprv_per_layer:{imprv_per_layer}')
        if factor_selection_mode == "arc_aslora" and factor_map:
            for layer_idx in range(total_layers):
                selected = int(layer_idx) in factor_map
                decision = factor_decision_metadata.get(int(layer_idx), {})
                raw_scores = (
                    arc_aslora_aggregated_scores.get(int(layer_idx), {})
                    if arc_aslora_aggregated_scores is not None else {}
                )
                ema_scores = arc_aslora_selector.ema_scores[int(layer_idx)]
                arc_aslora_selection_log.append({
                    "round": int(round + 1),
                    "layer_id": int(layer_idx),
                    "selected": bool(selected),
                    "selected_factor": factor_map.get(int(layer_idx), ""),
                    "decision_reason": decision.get("reason", "inactive"),
                    "temperature": float(decision.get("temperature", arc_aslora_selector.last_temperature)),
                    "probability_a": decision.get("probability_a", ""),
                    "random_draw": decision.get("random_draw", ""),
                    "raw_score_a": raw_scores.get("A", ""),
                    "raw_score_b": raw_scores.get("B", ""),
                    "ema_score_a": ema_scores["A"] if ema_scores["A"] is not None else "",
                    "ema_score_b": ema_scores["B"] if ema_scores["B"] is not None else "",
                })
            logger.info(
                "ARC+AS-LoRA factor feedback round=%d uses client gradient-curvature scores only; "
                "ARC validation reward remains reserved for layer feedback.",
                round + 1,
            )
        if factor_selection_mode == "outcome_alignment" and factor_map:
            for layer_idx in range(total_layers):
                selected = int(layer_idx) in factor_map
                decision = factor_decision_metadata.get(int(layer_idx), {})
                alignment_row = (
                    outcome_alignment_diagnostics["per_layer"].get(int(layer_idx), {})
                    if outcome_alignment_diagnostics is not None else {}
                )
                update_row = (
                    outcome_alignment_update["layers"].get(int(layer_idx), {})
                    if outcome_alignment_update is not None else {}
                )
                outcome_alignment_selection_log.append({
                    "round": int(round + 1),
                    "layer_id": int(layer_idx),
                    "selected": bool(selected),
                    "selected_factor": factor_map.get(int(layer_idx), ""),
                    "decision_reason": decision.get("reason", "inactive"),
                    "utility_A_before": decision.get("utility_a", ""),
                    "utility_B_before": decision.get("utility_b", ""),
                    "count_A_before": decision.get("count_a", ""),
                    "count_B_before": decision.get("count_b", ""),
                    "ucb_A": decision.get("ucb_a", ""),
                    "ucb_B": decision.get("ucb_b", ""),
                    "alignment": alignment_row.get("alignment", ""),
                    "gradient_update_dot": alignment_row.get(
                        "gradient_update_dot", ""
                    ),
                    "gradient_norm": alignment_row.get("gradient_norm", ""),
                    "delta_norm": alignment_row.get("delta_norm", ""),
                    "validation_reward": float(reward_t) if selected else "",
                    "normalized_reward": (
                        outcome_alignment_update["normalized_reward"]
                        if selected and outcome_alignment_update is not None else ""
                    ),
                    "attribution_weight": update_row.get(
                        "attribution_weight", ""
                    ),
                    "global_component": update_row.get("global_component", ""),
                    "credit": update_row.get("credit", ""),
                    "selected_utility_before": update_row.get(
                        "utility_before", ""
                    ),
                    "selected_utility_after": update_row.get(
                        "utility_after", ""
                    ),
                    "utility_A_after": float(
                        outcome_alignment_selector.utilities[layer_idx]["A"]
                    ),
                    "utility_B_after": float(
                        outcome_alignment_selector.utilities[layer_idx]["B"]
                    ),
                    "count_A_after": int(
                        outcome_alignment_selector.counts[layer_idx]["A"]
                    ),
                    "count_B_after": int(
                        outcome_alignment_selector.counts[layer_idx]["B"]
                    ),
                    "gradient_validation_loss": (
                        outcome_alignment_diagnostics["validation_loss"]
                        if selected and outcome_alignment_diagnostics is not None else ""
                    ),
                    "gradient_validation_examples": (
                        outcome_alignment_diagnostics["num_examples"]
                        if selected and outcome_alignment_diagnostics is not None else ""
                    ),
                })
            _write_rows_csv_atomic(
                os.path.join(log_dir, "outcome_alignment_selection_log.csv"),
                outcome_alignment_selection_log,
            )
        if factor_selection_mode == "cmfv" and factor_map:
            for layer_idx in range(total_layers):
                decision = factor_decision_metadata[layer_idx]
                post_decision = (
                    cmfv_post_metadata[layer_idx]
                    if cmfv_post_metadata is not None else decision
                )
                selected = bool(decision["selected"])
                selected_factor = factor_map.get(layer_idx, "")
                observation = (
                    cmfv_coherence_observations.get(layer_idx, {})
                    if selected and cmfv_coherence_observations is not None
                    else {}
                )
                update_row = (
                    cmfv_coherence_update.get(layer_idx, {})
                    if selected and cmfv_coherence_update is not None
                    else {}
                )
                task_row = (
                    cmfv_task_fit["per_layer"].get(layer_idx, {})
                    if cmfv_task_fit is not None else {}
                )
                cmfv_selection_log.append({
                    "round": int(round + 1),
                    "layer_id": int(layer_idx),
                    "selected": selected,
                    "selected_factor": selected_factor,
                    "decision_reason": decision["decision_reason"],
                    "exploration": bool(decision["exploration"]),
                    "probability_A": decision["probability_A"],
                    "factor_random_draw": (
                        decision["factor_random_draw"]
                        if decision["factor_random_draw"] is not None else ""
                    ),
                    "posterior_z": decision["posterior_z"],
                    "greedy_factor": decision["greedy_factor"],
                    "Q_A": decision["A"]["task_fit"],
                    "Q_B": decision["B"]["task_fit"],
                    "coherence_A_before": decision["A"]["coherence"],
                    "coherence_B_before": decision["B"]["coherence"],
                    "uncertainty_A": decision["A"]["uncertainty"],
                    "uncertainty_B": decision["B"]["uncertainty"],
                    "value_uncertainty_A": decision["A"]["value_uncertainty"],
                    "value_uncertainty_B": decision["B"]["value_uncertainty"],
                    "exploration_bonus_A": decision["A"]["exploration_bonus"],
                    "exploration_bonus_B": decision["B"]["exploration_bonus"],
                    "decision_score_A": decision["A"]["decision_score"],
                    "decision_score_B": decision["B"]["decision_score"],
                    "factor_value_A_before": decision["factor_value_A"],
                    "factor_value_B_before": decision["factor_value_B"],
                    "lower_bound_A": decision["A"]["lower_bound"],
                    "lower_bound_B": decision["B"]["lower_bound"],
                    "upper_bound_A": decision["A"]["upper_bound"],
                    "upper_bound_B": decision["B"]["upper_bound"],
                    "layer_value_before": decision["layer_value"],
                    "layer_decision_score": decision["layer_decision_score"],
                    "count_A_before": decision["A"]["count"],
                    "count_B_before": decision["B"]["count"],
                    "observed_coherence": observation.get("coherence", ""),
                    "aggregate_update_norm": observation.get(
                        "aggregate_norm", ""
                    ),
                    "weighted_client_energy": observation.get(
                        "weighted_client_energy", ""
                    ),
                    "coherence_selected_after": update_row.get(
                        "coherence_after", ""
                    ),
                    "coherence_A_after": post_decision["A"]["coherence"],
                    "coherence_B_after": post_decision["B"]["coherence"],
                    "factor_value_A_after": post_decision["factor_value_A"],
                    "factor_value_B_after": post_decision["factor_value_B"],
                    "layer_value_after": post_decision["layer_value"],
                    "count_A_after": int(cmfv_selector.counts[layer_idx]["A"]),
                    "count_B_after": int(cmfv_selector.counts[layer_idx]["B"]),
                    "gradient_energy": task_row.get("gradient_energy", ""),
                    "factor_gradient_energy_A": task_row.get(
                        "factor_gradient_energy_A", ""
                    ),
                    "factor_gradient_energy_B": task_row.get(
                        "factor_gradient_energy_B", ""
                    ),
                    "directional_curvature_A": task_row.get(
                        "directional_curvature_A", ""
                    ),
                    "directional_curvature_B": task_row.get(
                        "directional_curvature_B", ""
                    ),
                    "stability_multiplier_A": task_row.get(
                        "stability_multiplier_A", ""
                    ),
                    "stability_multiplier_B": task_row.get(
                        "stability_multiplier_B", ""
                    ),
                    "marginal_gain_A": task_row.get("marginal_gain_A", ""),
                    "marginal_gain_B": task_row.get("marginal_gain_B", ""),
                    "validation_gradient_loss": (
                        cmfv_task_fit["validation_loss"]
                        if cmfv_task_fit is not None else ""
                    ),
                    "validation_reward": float(reward_t),
                })
            _write_rows_csv_atomic(
                os.path.join(log_dir, "cmfv_selection_log.csv"),
                cmfv_selection_log,
            )
        if factor_selection_mode == "gi_cmfv" and factor_map:
            for layer_idx in next_round_layer:
                layer_idx = int(layer_idx)
                decision = factor_decision_metadata[layer_idx]
                client_update = gi_cmfv_client_update[layer_idx]
                server_row = gi_cmfv_server_scores[layer_idx]
                gi_cmfv_selection_log.append({
                    "round": int(round + 1),
                    "layer_id": layer_idx,
                    "selected_factor": factor_map[layer_idx],
                    "greedy_factor": decision["greedy_factor"],
                    "exploration": bool(decision["exploration"]),
                    "decision_reason": decision["decision_reason"],
                    "factor_warmup_rounds": factor_warmup_rounds,
                    "warmup_active": bool(decision["warmup_active"]),
                    "schedule_override": bool(
                        decision["schedule_override"]
                    ),
                    "probability_A": decision["probability_A"],
                    "relative_difference": decision["relative_difference"],
                    "selection_logit": decision["selection_logit"],
                    "selection_temperature": decision["temperature"],
                    "factor_random_draw": decision["random_draw"],
                    "server_A_revision": server_row["A"],
                    "server_B_exploitation": server_row["B"],
                    "server_A_full": server_row["A_full"],
                    "server_A_overlap": server_row["A_overlap"],
                    "server_gradient_energy": server_row[
                        "gradient_energy"
                    ],
                    "rank_A_mean": server_row["rank_A_mean"],
                    "rank_B_mean": server_row["rank_B_mean"],
                    "server_share_A": decision["A"]["server_share"],
                    "server_share_B": decision["B"]["server_share"],
                    "factor_prior": "none",
                    "client_evidence_available_A": decision["A"][
                        "evidence_available"
                    ],
                    "client_evidence_available_B": decision["B"][
                        "evidence_available"
                    ],
                    "evidence_mean_A_before": decision["A"][
                        "evidence_mean"
                    ],
                    "evidence_mean_B_before": decision["B"][
                        "evidence_mean"
                    ],
                    "effective_evidence_count_A": decision["A"][
                        "effective_count"
                    ],
                    "effective_evidence_count_B": decision["B"][
                        "effective_count"
                    ],
                    "server_weight_applied_A": decision["A"][
                        "server_weight_applied"
                    ],
                    "server_weight_applied_B": decision["B"][
                        "server_weight_applied"
                    ],
                    "client_weight_applied_A": decision["A"][
                        "client_weight_applied"
                    ],
                    "client_weight_applied_B": decision["B"][
                        "client_weight_applied"
                    ],
                    "client_disagreement_A_before": decision["A"][
                        "client_disagreement"
                    ],
                    "client_disagreement_B_before": decision["B"][
                        "client_disagreement"
                    ],
                    "uncertainty_A": decision["A"]["uncertainty"],
                    "uncertainty_B": decision["B"]["uncertainty"],
                    "A_revision_disagreement_penalty": decision["A"][
                        "disagreement_penalty_applied"
                    ],
                    "risk_adjusted_value_A": decision["A"][
                        "risk_adjusted_value"
                    ],
                    "risk_adjusted_value_B": decision["B"][
                        "risk_adjusted_value"
                    ],
                    "client_round_mean_A": client_update["A"][
                        "round_mean"
                    ],
                    "client_round_mean_B": client_update["B"][
                        "round_mean"
                    ],
                    "client_round_variance_A": client_update["A"][
                        "round_variance"
                    ],
                    "client_round_variance_B": client_update["B"][
                        "round_variance"
                    ],
                    "client_mean_A_after": client_update["A"][
                        "mean_after"
                    ],
                    "client_mean_B_after": client_update["B"][
                        "mean_after"
                    ],
                    "client_variance_A_after": client_update["A"][
                        "variance_after"
                    ],
                    "client_variance_B_after": client_update["B"][
                        "variance_after"
                    ],
                    "count_A_after": client_update["A"]["count_after"],
                    "count_B_after": client_update["B"]["count_after"],
                    "server_validation_loss": gi_cmfv_server_diagnostics[
                        "validation_loss"
                    ],
                    "validation_reward": float(reward_t),
                })
            _write_rows_csv_atomic(
                os.path.join(log_dir, "gi_cmfv_selection_log.csv"),
                gi_cmfv_selection_log,
            )
        if factor_selection_mode == "qgs" and factor_map:
            if qgs_scores is None or qgs_diagnostics is None:
                raise RuntimeError(
                    "QGS requires participating-client validation scores"
                )
            for layer_idx in next_round_layer:
                layer_idx = int(layer_idx)
                score_row = qgs_scores[layer_idx]
                decision = factor_decision_metadata[layer_idx]
                qgs_selection_log.append({
                    "round": int(round + 1),
                    "layer_id": layer_idx,
                    "selected_factor": factor_map[layer_idx],
                    "decision_reason": decision["decision_reason"],
                    "factor_prior": "none",
                    "warmup": False,
                    "exploration": False,
                    "score_A": score_row["A"],
                    "score_B": score_row["B"],
                    "score_margin_A_minus_B": decision[
                        "score_margin_A_minus_B"
                    ],
                    "guaranteed_descent_numerator_A": decision[
                        "guaranteed_descent_numerator_A"
                    ],
                    "guaranteed_descent_numerator_B": decision[
                        "guaranteed_descent_numerator_B"
                    ],
                    "A_projected_gradient_fraction": score_row[
                        "A_fraction"
                    ],
                    "B_projected_gradient_fraction": score_row[
                        "B_fraction"
                    ],
                    "mean_gradient_energy": score_row[
                        "gradient_energy"
                    ],
                    "rank_A_mean": score_row["rank_A_mean"],
                    "rank_B_mean": score_row["rank_B_mean"],
                    "participating_client_ids": ",".join(
                        str(client_id)
                        for client_id in qgs_diagnostics["client_ids"]
                    ),
                    "participating_client_count": qgs_diagnostics[
                        "num_clients"
                    ],
                    "client_mean_validation_loss": qgs_diagnostics[
                        "validation_loss"
                    ],
                    "per_client_validation_losses": json.dumps(
                        qgs_diagnostics["client_losses"],
                        sort_keys=True,
                    ),
                    "objective": qgs_diagnostics["objective"],
                    "smoothness_model": qgs_diagnostics[
                        "smoothness_model"
                    ],
                    "validation_reward": float(reward_t),
                })
            _write_rows_csv_atomic(
                os.path.join(log_dir, "qgs_selection_log.csv"),
                qgs_selection_log,
            )
        if factor_selection_mode == "cfvg" and factor_map:
            if cfvg_scores is None or cfvg_diagnostics is None:
                raise RuntimeError(
                    "CFVG requires dual counterfactual probe results"
                )
            for layer_idx in next_round_layer:
                layer_idx = int(layer_idx)
                score_row = cfvg_scores[layer_idx]
                cfvg_selection_log.append({
                    "round": int(round + 1),
                    "layer_id": layer_idx,
                    "selected_factor": factor_map[layer_idx],
                    "decision_reason": score_row["decision_reason"],
                    "factor_prior": "none",
                    "warmup": False,
                    "exploration": False,
                    "gain_A": score_row["gain_A"],
                    "gain_B": score_row["gain_B"],
                    "gain_margin_A_minus_B": (
                        score_row["gain_A"] - score_row["gain_B"]
                    ),
                    "candidate_loss_A": score_row["candidate_loss_A"],
                    "candidate_loss_B": score_row["candidate_loss_B"],
                    "conditional_baseline_loss": score_row[
                        "conditional_baseline_loss"
                    ],
                    "selected_candidate_loss": score_row[
                        "selected_candidate_loss"
                    ],
                    "update_norm_A": score_row["update_norm_A"],
                    "update_norm_B": score_row["update_norm_B"],
                    "paired_advantage_A": score_row[
                        "paired_advantage_A"
                    ],
                    "paired_standard_error": score_row[
                        "paired_standard_error"
                    ],
                    "confidence_z": score_row["confidence_z"],
                    "confidence_threshold": score_row[
                        "confidence_threshold"
                    ],
                    "confidence_decisive": score_row[
                        "confidence_decisive"
                    ],
                    "raw_winner": score_row["raw_winner"],
                    "incumbent_factor": score_row["incumbent_factor"],
                    "actionable_A": score_row["actionable_A"],
                    "actionable_B": score_row["actionable_B"],
                    "actionability_decisive": score_row[
                        "actionability_decisive"
                    ],
                    "actionability_override": score_row[
                        "actionability_override"
                    ],
                    "candidate_client_losses_A": json.dumps(
                        score_row["candidate_client_losses_A"],
                        sort_keys=True,
                    ),
                    "candidate_client_losses_B": json.dumps(
                        score_row["candidate_client_losses_B"],
                        sort_keys=True,
                    ),
                    "base_map_validation_loss": cfvg_diagnostics[
                        "base_validation_loss"
                    ],
                    "selected_map_validation_loss": cfvg_diagnostics[
                        "selected_map_validation_loss"
                    ],
                    "selected_map_validation_gain": cfvg_diagnostics[
                        "selected_map_validation_gain"
                    ],
                    "participating_client_ids": ",".join(
                        str(client_id)
                        for client_id in cfvg_diagnostics["client_ids"]
                    ),
                    "participating_client_count": cfvg_diagnostics[
                        "num_clients"
                    ],
                    "aggregation_weights": json.dumps(
                        cfvg_diagnostics["aggregation_weights"],
                        sort_keys=True,
                    ),
                    "aggregation_weight_sum": cfvg_diagnostics[
                        "aggregation_weight_sum"
                    ],
                    "base_client_validation_losses": json.dumps(
                        cfvg_diagnostics[
                            "base_client_validation_losses"
                        ],
                        sort_keys=True,
                    ),
                    "probe_client_train_losses": json.dumps(
                        cfvg_diagnostics["probe_client_train_losses"],
                        sort_keys=True,
                    ),
                    "probe_optimizer": cfvg_diagnostics["optimizer"],
                    "probe_learning_rate": cfvg_diagnostics[
                        "learning_rate"
                    ],
                    "probe_train_batch_size": (
                        cfvg_probe_train_batch_size
                    ),
                    "probe_validation_batch_size": (
                        cfvg_probe_validation_batch_size
                    ),
                    "probe_steps": cfvg_diagnostics["probe_steps"],
                    "probe_validation_batches": (
                        cfvg_probe_validation_batches
                    ),
                    "objective": cfvg_diagnostics["objective"],
                    "selection": cfvg_diagnostics["selection"],
                    "aggregation": cfvg_diagnostics["aggregation"],
                    "validation_reward": float(reward_t),
                })
            _write_rows_csv_atomic(
                os.path.join(log_dir, "cfvg_selection_log.csv"),
                cfvg_selection_log,
            )
        if factor_selection_mode == "global_adaptive" and factor_map:
            logger.info(
                "Global adaptive factor metrics round=%d, active_factor=%s, factor_selection_reason=%s, "
                "factor_score_a_before=%.6f, factor_score_b_before=%.6f, "
                "factor_score_a_after=%.6f, factor_score_b_after=%.6f, "
                "factor_count_a=%d, factor_count_b=%d, factor_last_selected_a=%s, "
                "factor_last_selected_b=%s, factor_utility=%.6f, factor_a_ratio_so_far=%.6f, "
                "factor_b_ratio_so_far=%.6f, server_macro_f1=%.6f, server_validation_loss=%.6f, "
                "reward=%.6f, active_factor_update_norm=%.6f",
                round + 1,
                active_factor,
                factor_selection_reason,
                factor_selector_before["score_a"] if factor_selector_before is not None else factor_selector.score_a,
                factor_selector_before["score_b"] if factor_selector_before is not None else factor_selector.score_b,
                factor_selector_update["score_a_after"] if factor_selector_update is not None else factor_selector.score_a,
                factor_selector_update["score_b_after"] if factor_selector_update is not None else factor_selector.score_b,
                factor_selector_update["count_a"] if factor_selector_update is not None else factor_selector.count_a,
                factor_selector_update["count_b"] if factor_selector_update is not None else factor_selector.count_b,
                factor_selector_update["last_selected_a"] if factor_selector_update is not None else factor_selector.last_selected_a,
                factor_selector_update["last_selected_b"] if factor_selector_update is not None else factor_selector.last_selected_b,
                factor_selector_update["factor_utility"] if factor_selector_update is not None else 0.0,
                factor_selector_update["factor_a_ratio_so_far"] if factor_selector_update is not None else 0.0,
                factor_selector_update["factor_b_ratio_so_far"] if factor_selector_update is not None else 0.0,
                server_metrics["macro_f1"],
                test_loss,
                reward_t,
                q,
            )
            global_factor_selection_log.append({
                "round": round + 1,
                "factor_selection_mode": factor_selection_mode,
                "active_factor": active_factor,
                "factor_selection_reason": factor_selection_reason,
                "factor_score_a_before": factor_selector_before["score_a"] if factor_selector_before is not None else factor_selector.score_a,
                "factor_score_b_before": factor_selector_before["score_b"] if factor_selector_before is not None else factor_selector.score_b,
                "factor_score_a_after": factor_selector_update["score_a_after"] if factor_selector_update is not None else factor_selector.score_a,
                "factor_score_b_after": factor_selector_update["score_b_after"] if factor_selector_update is not None else factor_selector.score_b,
                "factor_count_a": factor_selector_update["count_a"] if factor_selector_update is not None else factor_selector.count_a,
                "factor_count_b": factor_selector_update["count_b"] if factor_selector_update is not None else factor_selector.count_b,
                "factor_last_selected_a": factor_selector_update["last_selected_a"] if factor_selector_update is not None else factor_selector.last_selected_a,
                "factor_last_selected_b": factor_selector_update["last_selected_b"] if factor_selector_update is not None else factor_selector.last_selected_b,
                "factor_epsilon": factor_selector.epsilon,
                "factor_score_beta": factor_selector.beta,
                "factor_utility": factor_selector_update["factor_utility"] if factor_selector_update is not None else 0.0,
                "num_selected_layers": selected_layer_count,
                "selected_layer_ids": ",".join(str(x) for x in selected_layer_ids),
                "server_macro_f1": float(server_metrics["macro_f1"]),
                "server_validation_loss": float(test_loss),
                "reward": float(reward_t),
                "active_factor_update_norm": float(q),
                "factor_a_ratio_so_far": factor_selector_update["factor_a_ratio_so_far"] if factor_selector_update is not None else 0.0,
                "factor_b_ratio_so_far": factor_selector_update["factor_b_ratio_so_far"] if factor_selector_update is not None else 0.0,
            })
        if factor_selection_mode in SCORE_HISTORY_FACTOR_MODES and factor_map:
            logger.info(
                "Score-history factor metrics mode=%s, round=%d, reward=%.6f, "
                "server_macro_f1=%.6f, server_loss=%.6f, q_t=%.6f, factor_map=%s",
                factor_selection_mode,
                round + 1,
                reward_t,
                server_metrics["macro_f1"],
                test_loss,
                q,
                factor_map,
            )
            for layer_idx in range(total_layers):
                before_scores = (
                    factor_scores_before_round[int(layer_idx)]
                    if factor_scores_before_round is not None else factor_scores[int(layer_idx)]
                )
                before_counts = (
                    factor_counts_before_round[int(layer_idx)]
                    if factor_counts_before_round is not None else factor_select_counts[int(layer_idx)]
                )
                decision = factor_decision_metadata.get(int(layer_idx), {})
                selected = int(layer_idx) in factor_map
                assigned_credit = (
                    factor_credit_result["credits"][int(layer_idx)]
                    if selected and factor_credit_result is not None else 0.0
                )
                rms_delta = (
                    factor_credit_result.get("rms_deltas", {}).get(int(layer_idx), 0.0)
                    if selected and factor_credit_result is not None else 0.0
                )
                rms_weight = (
                    factor_credit_result["weights"][int(layer_idx)]
                    if (
                        selected
                        and factor_credit_result is not None
                        and factor_credit_result["credit_mode"] == "rms"
                    )
                    else 0.0
                )
                product_delta = (
                    factor_credit_result.get("product_deltas", {}).get(
                        int(layer_idx), 0.0
                    )
                    if selected and factor_credit_result is not None else 0.0
                )
                product_weight = (
                    factor_credit_result["weights"][int(layer_idx)]
                    if (
                        selected
                        and factor_credit_result is not None
                        and factor_credit_result["credit_mode"] == "product_delta"
                    )
                    else 0.0
                )
                adaptive_factor_selection_log.append({
                    "round": round + 1,
                    "factor_selection_mode": factor_selection_mode,
                    "layer_id": int(layer_idx),
                    "selected": selected,
                    "selected_factor": factor_map.get(int(layer_idx), ""),
                    "decision_reason": decision.get("reason", ""),
                    "random_draw": decision.get("random_draw", ""),
                    "score_A_before": float(before_scores["A"]),
                    "score_B_before": float(before_scores["B"]),
                    "count_A_before": int(before_counts["A"]),
                    "count_B_before": int(before_counts["B"]),
                    "credit": float(assigned_credit),
                    "credit_mode": (
                        factor_credit_result["credit_mode"]
                        if factor_credit_result is not None else ""
                    ),
                    "rms_delta": float(rms_delta),
                    "rms_credit_weight": float(rms_weight),
                    "assigned_factor_credit": float(assigned_credit),
                    "rms_credit_fallback": bool(
                        factor_credit_result["fallback_used"] if factor_credit_result is not None else False
                    ),
                    "total_round_rms_delta": float(
                        factor_credit_result.get("total_rms_delta", 0.0)
                        if factor_credit_result is not None else 0.0
                    ),
                    "effective_product_delta": float(product_delta),
                    "effective_product_credit_weight": float(product_weight),
                    "total_round_effective_product_delta": float(
                        factor_credit_result.get("total_product_delta", 0.0)
                        if factor_credit_result is not None else 0.0
                    ),
                    "credit_fallback": bool(
                        factor_credit_result["fallback_used"]
                        if factor_credit_result is not None else False
                    ),
                    "score_A_after": float(factor_scores[int(layer_idx)]["A"]),
                    "score_B_after": float(factor_scores[int(layer_idx)]["B"]),
                    "count_A_after": int(factor_select_counts[int(layer_idx)]["A"]),
                    "count_B_after": int(factor_select_counts[int(layer_idx)]["B"]),
                    "delta_norm": float(adaptive_layer_delta_norms.get(int(layer_idx), 0.0)),
                })
        if (rl):
            if layer_select_method == "rl" and jl_study is not None:
                controller_metrics = dict(server_metrics)
                if jl_study.spec['reward_threshold'] is not None:
                    controller_metrics.update(macro_f1=server_metrics['fixed_macro_f1'], threshold=0.5)
                next_g, event = jl_study.finish_round(
                    round + 1, (round + 1, q, test_loss), controller_metrics, len(next_round_layer))
                logger.info('ControllerStudy round=%s executed_G=%s reward=%s next_G=%s learning=%s',
                            round + 1, event['executed_G'], event['reward'], next_g, event['learning'])
                if next_g is not None:
                    next_round_budget = next_g
                    next_round_layer, next_round_layer_reason, next_round_layer_random_draw = (
                        select_layers_fixed_20_80(
                            all_layer_ids=list(range(total_layers)), layer_scores=imprv_per_layer,
                            budget=next_g, rng=server_rng, layer_random_prob=layer_random_prob))
                    next_round_layer, next_round_layer_reason, next_round_layer_random_draw = jl_component.layers(
                        (next_round_layer, next_round_layer_reason, next_round_layer_random_draw),
                        list(range(total_layers)), next_g, round + 2)
                elif round + 1 < factor_warmup_rounds:
                    next_round_budget = total_layers
                    next_round_layer = list(range(total_layers))
                    next_round_layer_reason = 'warmup_all_layers'
                    next_round_layer_random_draw = None
            elif layer_select_method == "rl":
                param_caculate_per_round += agent.get_model_parameters_count()

                state = (round + 1, q, test_loss)
                logger.info(
                    f"RL state components: round={round + 1}, q={q}, "
                    f"state_loss={test_loss}, reward={reward_t}"
                )
                skip_dqn_transition = bool(
                    factor_selection_mode == "global_adaptive" and round < factor_warmup_rounds
                )
                if len(trajectory_state) != 0 and not skip_dqn_transition:
                    # Warm-up can override the planned budget; replay the executed action.
                    agent.remember(trajectory_state[-1], len(next_round_layer) - 1, reward_t, state)
                    dqn_loss = agent.replay()
                    dqn_loss_history.append(dqn_loss)
                    logger.info(f"double_dqn_loss:{dqn_loss}")
                elif skip_dqn_transition:
                    logger.info(
                        "Skip DQN transition/action for warm-up round %d in global_adaptive mode; "
                        "selected_budget is forced to all layers.",
                        round + 1,
                    )

                should_plan_next_budget = not (
                    factor_selection_mode == "global_adaptive" and (round + 1) < factor_warmup_rounds
                )
                if should_plan_next_budget:
                    action = agent.act(state)
                    trajectory_state.append(state)
                    trajectory_action.append(action)
                    num_next_round_layer = action + 1
                    if round == n_comm_rounds - 1:
                        if factor_selection_mode in {
                            "product_delta",
                            "gi_cmfv",
                            "qgs",
                            "cfvg",
                        }:
                            agent.save_models(
                                os.path.join(log_dir, "q_network.pth"),
                                os.path.join(log_dir, "target_network.pth"),
                                os.path.join(log_dir, "memory.pkl"),
                            )
                        else:
                            agent.save_models(
                                os.path.join(log_dir, "q_network.pth"),
                                os.path.join(log_dir, "target_network.pth"),
                                os.path.join(log_dir, "memory.pkl"),
                            )
                    logger.info(f"num_next_round_layer:{num_next_round_layer}")
                    next_round_budget = int(num_next_round_layer)
                else:
                    next_round_budget = total_layers
                    next_round_layer = list(range(total_layers))
                    next_round_layer_reason = "warmup_all_layers"
                    next_round_layer_random_draw = None
                    logger.info(
                        "Keep next round at full layer set because global_adaptive warm-up continues "
                        "(next_round_budget=%d).",
                        next_round_budget,
                    )

                if should_plan_next_budget and factor_selection_mode == "cmfv":
                    next_round_layer = list(range(total_layers))
                    next_round_layer_reason = "cmfv_pending_current_task_fit"
                    next_round_layer_random_draw = None
                    logger.info(
                        "CMFV next-round budget=%d; concrete layer-factor coordinates "
                        "will be selected from next-round task fit and coherence.",
                        next_round_budget,
                    )
                elif should_plan_next_budget and factor_selection_mode in {
                    "adaptive_score",
                    "product_delta",
                    "global_adaptive",
                    "arc_aslora",
                    "outcome_alignment",
                    "gi_cmfv",
                    "qgs",
                    "cfvg",
                }:
                    next_round_layer, next_round_layer_reason, next_round_layer_random_draw = (
                        select_layers_fixed_20_80(
                            all_layer_ids=list(range(total_layers)),
                            layer_scores=imprv_per_layer,
                            budget=next_round_budget,
                            rng=server_rng,
                            layer_random_prob=layer_random_prob,
                        )
                    )
                    logger.info(
                        "Adaptive next_round_layer: budget=%d, layers=%s, reason=%s, random_draw=%s",
                        next_round_budget,
                        next_round_layer,
                        next_round_layer_reason,
                        next_round_layer_random_draw,
                    )
                elif should_plan_next_budget and len(server_macro_f1_history) < 2:
                    next_round_layer = random.sample(range(total_layers), num_next_round_layer)
                    next_round_layer_reason = "legacy_random_warm_start"
                    next_round_layer_random_draw = None
                elif should_plan_next_budget:
                    next_round_layer = select_layers_epsilon_greedy(
                        imprv_per_layer,
                        num_next_round_layer,
                        epsilon_layer=epsilon_layer,
                        total_layers=total_layers,
                        layer_select_mode=layer_select_mode,
                    )
                    next_round_layer_reason = "legacy_epsilon_greedy"
                    next_round_layer_random_draw = None
                logger.info(
                    f"next_round_layer (largest-contribution first):{next_round_layer}"
                )
            elif layer_select_method == "random_g":
                if round + 1 < n_comm_rounds:
                    next_round_budget = random_g_schedule[round + 1]
                    if round + 1 < factor_warmup_rounds:
                        next_round_layer = list(range(total_layers))
                        next_round_layer_reason = "warmup_all_layers"
                        next_round_layer_random_draw = None
                    else:
                        next_round_layer, next_round_layer_reason, next_round_layer_random_draw = (
                            select_layers_fixed_20_80(
                                all_layer_ids=list(range(total_layers)),
                                layer_scores=imprv_per_layer, budget=next_round_budget,
                                rng=server_rng, layer_random_prob=layer_random_prob,
                            )
                        )
                    logger.info("Random-G plan: round=%d G=%d layers=%s reason=%s",
                                round + 2, next_round_budget, next_round_layer, next_round_layer_reason)
            elif layer_select_method == "fedfmsl" and layer_selector is not None:
                next_round_layer = layer_selector.select_next_layers(
                    round, float(server_macro_f1_history[-1])
                )
            elif layer_select_method == "spdcfl" and layer_selector is not None:
                next_round_layer = layer_selector.select_next_layers(round, imprv_per_layer)
            elif layer_select_method == "all":
                all_layer_state = resolve_all_layer_selection(total_layers)
                next_round_layer = all_layer_state["selected_layers"]
                next_round_budget = all_layer_state["selected_budget"]
                next_round_layer_reason = all_layer_state["layer_selection_reason"]
                next_round_layer_random_draw = all_layer_state["layer_random_draw"]
                logger.info(
                    "All-layer baseline keeps next_round_budget=%d and next_round_layer=%s.",
                    next_round_budget,
                    next_round_layer,
                )
            else:
                next_round_layer = list(range(12))
            if prepare_rl_lora_round is not None:
                _, _, _, preview_next_round_matrix = prepare_rl_lora_round(
                    round + 1, next_round_layer, model, logger
                )
                logger.info(
                    f"下一轮 LoRA baseline ({LORA_AGG_MODE}): {preview_next_round_matrix}"
                )
            else:
                preview_next_round_matrix = None
                if factor_selection_mode in {
                    "adaptive_score",
                    "product_delta",
                    "global_adaptive",
                    "arc_aslora",
                    "outcome_alignment",
                    "cmfv",
                    "gi_cmfv",
                    "qgs",
                    "cfvg",
                } and factor_map:
                    logger.info(
                        "下一轮 adaptive factor selection pending server decision; "
                        "selected layers will be %s",
                        next_round_layer,
                    )
                elif _trend_slope_switching_active:
                    next_round_factor = factor_name_to_coordinate_value(
                        trend_slope_state["current_factor"]
                    )
                else:
                    next_round_factor = get_coordinate_factor(round + 1)
                if preview_next_round_matrix is not None:
                    preview_next_round_matrix = build_round_matrix(next_round_layer, next_round_factor, n_layers=12)
                    logger.info(
                        f"下一轮统一更新因子：{factor_name(next_round_factor)} "
                        f"(schedule={coordinate_schedule_label}); "
                        f"next_round_matrix:{preview_next_round_matrix}"
                    )

            should_save_adaptive_state = (
                factor_selection_mode in SCORE_HISTORY_FACTOR_MODES
                and layer_select_method in ADAPTIVE_CD_LAYER_METHODS
            ) or (
                factor_selection_mode == "global_adaptive" and layer_select_method == "rl"
            ) or (
                factor_selection_mode == "arc_aslora" and layer_select_method == "rl"
            ) or (
                factor_selection_mode == "outcome_alignment"
                and layer_select_method == "rl"
            ) or (
                factor_selection_mode == "cmfv"
                and layer_select_method == "rl"
            ) or (
                factor_selection_mode == "gi_cmfv"
                and layer_select_method == "rl"
            ) or (
                factor_selection_mode == "qgs"
                and layer_select_method == "rl"
            ) or (
                factor_selection_mode == "cfvg"
                and layer_select_method == "rl"
            )
            if should_save_adaptive_state:
                save_adaptive_factor_checkpoint(
                    log_dir,
                    {
                        "factor_selection_mode": factor_selection_mode,
                        "warmup_rounds": factor_warmup_rounds,
                        "layer_random_prob": layer_random_prob,
                        "factor_random_prob": factor_random_prob,
                        "factor_score_beta": factor_score_beta,
                        "factor_lr": factor_lr,
                        "factor_scores": factor_scores,
                        "factor_select_counts": factor_select_counts,
                        "factor_selector_state": factor_selector.state_dict() if factor_selection_mode == "global_adaptive" else None,
                        "arc_aslora_selector_state": arc_aslora_selector.state_dict() if factor_selection_mode == "arc_aslora" else None,
                        "outcome_alignment_selector_state": (
                            outcome_alignment_selector.state_dict()
                            if factor_selection_mode == "outcome_alignment"
                            else None
                        ),
                        "cmfv_selector_state": (
                            cmfv_selector.state_dict()
                            if factor_selection_mode == "cmfv"
                            else None
                        ),
                        "gi_cmfv_selector_state": (
                            gi_cmfv_selector.state_dict()
                            if factor_selection_mode == "gi_cmfv"
                            else None
                        ),
                        "cfvg_incumbent_factors": (
                            dict(cfvg_incumbent_factors)
                            if factor_selection_mode == "cfvg"
                            else None
                        ),
                        "server_rng_state": server_rng.getstate(),
                    },
                )

         #如果使用聚类块坐标下降选择模型层
            if (cd == "cluster" or cd =='prob'):
                weights_per_layer = [[] for _ in range(12)]
                for name, param in global_model.named_parameters():
                    if name.split('.')[-1] == 'lora_a' or name.split('.')[-1] == 'lora_b':
                        # print(f'layer:{name}:')
                        # print(param.data)
                        layer_number = int(name.split('.')[-5])
                        weights_per_layer[layer_number].append(param.data.clone().cpu().numpy().flatten())
                for layer in range(12):
                    weights_per_layer[layer] = np.concatenate(weights_per_layer[layer])
                x = np.array(weights_per_layer)
                kmeans = KMeans(n_clusters=3, random_state=0).fit(x)
                labels = kmeans.labels_
                clusters = {i: [] for i in range(3)}
                for layer, cluster in enumerate(labels):
                    clusters[cluster].append(layer)
                logger.info(f"当前轮的划分簇：{clusters}")
                if cd =='cluster':
                    # 初始化一个字典来保存每个簇的总体提升量
                    cluster_improvements = {}

                    # 对于每个簇，计算其总体提升量
                    for cluster_id, layers in clusters.items():
                        improvement = sum(imprv_per_layer[layer] for layer in layers)
                        cluster_improvements[cluster_id] = improvement
                    logger.info("每个簇的总体提升量:", cluster_improvements)
                    best_cluster = max(cluster_improvements, key=cluster_improvements.get)
                    logger.info(f"提升量最大的簇是 Cluster {best_cluster}, 提升量为 {cluster_improvements[best_cluster]}")
                    # 获取该簇中的所有层编号
                    best_layers = clusters[best_cluster]

                    logger.info(f"Cluster {best_cluster} 包含以下层: {best_layers}")
                    next_round_layer = best_layers
                else:
                    sparsity_cluster = {}
                    for cluster_id, layers in clusters.items():
                        sparsity_sum = sum(all_sparsities[layer] for layer in layers)
                        sparsity_cluster[cluster_id] = sparsity_sum
                    logger.info("每个簇的sparsity总和:", sparsity_cluster)
                    best_cluster = max(sparsity_cluster, key=sparsity_cluster.get)
                    logger.info(f"sparsity最大的簇是 Cluster {best_cluster}, sparsity总和为 {sparsity_cluster[best_cluster]}")
                    best_layers = clusters[best_cluster]
                    logger.info(f"Cluster {best_cluster} 包含以下层: {best_layers}")
                    next_round_layer = best_layers
            elif cd=='sequence':
                layer_1 = [0,1,2]
                layer_2 = [3,4,5]
                layer_3 = [6,7,8]
                layer_4 = [9,10,11]
                if round%4==0:
                    next_round_layer = layer_2
                elif round%4==3:
                    next_round_layer = layer_3
                elif round%4==2:
                    next_round_layer = layer_4
                else:
                    next_round_layer = layer_1
            elif cd =='random':
                next_round_layer = random.sample(range(0, 12), 4)

        param_caculate.append(param_caculate_per_round)
        _gpu_enter_idle(args, logger=logger)
        _log_training_eta(
            logger,
            round,
            n_comm_rounds,
            time.perf_counter() - round_wall_start,
            time.perf_counter() - training_wall_start,
            method=args.jl_method,
        )
        if args.jl_validation_target is not None and jl_record['validation_macro_f1'] >= args.jl_validation_target:
            write_json(Path(log_dir) / 'jl_validation_target.json', {
                'target': args.jl_validation_target, 'round': round + 1,
                'validation_macro_f1': jl_record['validation_macro_f1'],
                'uplink_bytes_fp32': int(sum(param_upload) * 4),
            })
            break
        # 停止记录线程
        # stop_event.set()
        # recording_thread.join()
        #
        # # 模拟计算后
        # final_cpu_usage = process.cpu_percent(interval=None)
        # final_memory_usage = process.memory_info().rss
        # max_cpu_usage = max(cpu_usages)
        # avg_cpu_usage = sum(cpu_usages) / len(cpu_usages)
        # print(gpu_usages)
        # max_gpu_usage = max(gpu_usages)
        # logger.info(f"max_gpu_usage:{max_gpu_usage}")
        # avg_gpu_usage = sum(gpu_usages) / len(gpu_usages)
        # max_memory_usage = max(memory_usages)
        # avg_memory_usage = sum(memory_usages) / len(memory_usages)
        # logger.info(f"max_cpu:{max_cpu_usage:.2f}MB")
        # logger.info(f"avg_cpu:{avg_cpu_usage:.2f}MB")
        # logger.info(f"max_gpu:{max_gpu_usage:.2f}MB")
        # logger.info(f"avg_gpu:{avg_gpu_usage:.2f}MB")
        # logger.info(f"max_memory:{max_memory_usage:.2f}MB")
        # logger.info(f"avg_memory:{avg_memory_usage:.2f}MB")
        # max_cpu_usages_history.append(max_cpu_usage)
        # max_memory_usages_history.append(max_memory_usage)
        # max_gpu_usages_history.append(max_gpu_usage)
        # avg_gpu_usages_history.append(avg_gpu_usage)
        # avg_cpu_usages_history.append(avg_cpu_usage)
        # avg_memory_usages_history.append(avg_memory_usage)
    logger.debug(f"param_caculate:{param_caculate}")
    logger.debug(f"param_caculate:{sum(param_caculate)}")
    logger.debug(f"param_upload:{param_upload}")
    logger.debug(f"param_upload:{sum(param_upload)}")
    # JL outputs replace the legacy server-selected/local-test reporting below.
    jl_summary = jl_protocol.finalize(global_model)
    write_csv(Path(log_dir) / 'jl_factor_history.csv', adaptive_factor_selection_log)
    write_csv(Path(log_dir) / 'jl_aslora_history.csv', arc_aslora_selection_log)
    if agent is not None:
        torch.save({'q': agent.q_network.state_dict(), 'target': agent.target_network.state_dict(),
                    'optimizer': agent.optimizer.state_dict(), 'epsilon': agent.epsilon,
                    'memory': list(agent.memory), 'factor_scores': factor_scores,
                    'factor_counts': factor_select_counts, 'layer_scores': layer_contribution_scores},
                   os.path.join(log_dir, 'jl_controller_last.pt'))
    logger.info('JL final validation-selected result: %s', jl_summary)
    raise SystemExit(0)
