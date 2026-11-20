import argparse
import copy
import datetime
import json
import os
import pickle
from pathlib import Path
import random
import threading
import concurrent.futures
import gc
import math
from contextlib import nullcontext
import time
import torch
import numpy as np
from datasets import concatenate_datasets
from transformers import RobertaForSequenceClassification
import logging
from lora.lora_finetune import add_lora, freeze_layer, unfrozen
from rl.Agent import DQNAgent
from jl_controller_study import ControllerStudy
from jl_component_ablation import ComponentAblation, METHOD_MODES
from jl_controller_variants import VARIANTS, client_shuffle_seed
from datetime import datetime
from arc_aslora import ASLoRAComponentSelector, OFFICIAL_EMA_BETA, OFFICIAL_FD_EPS, OFFICIAL_GAMMA, OFFICIAL_SCORE_EPS, OFFICIAL_T0, OFFICIAL_TMIN
from jl_aggregation import aggregate_uploaded_deltas
from jl_trainer import configure_dropout, fit, test
from jl_protocol import EvaluationProtocol, no_local_evaluation, write_json, write_csv
from jl_rank import ValidationFeedbackRank, lora_layout, match_checkpoint_layout, resize_common_rank
from jl_progress import emit_progress, report_progress, FINAL_TEST_NOTE
from utils.gpu_memory_guard import get_gpu_memory_guard, init_gpu_memory_guard
model_file = os.environ.get('JL_MODEL_PATH', 'FacebookAI/roberta-base')
import sys
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

def str2bool(v):
    """Parse CLI booleans correctly (argparse type=bool breaks on 'False' -> True)."""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if s in {'0', 'false', 'no', 'n', 'off', ''}:
        return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got {v!r}')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training. Default is 64.')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate for the optimizer. Default is 0.001.')
    parser.add_argument('--lr_a_scale', type=float, default=4.0, help='LR multiplier for lora_a on pure-A coordinate rounds; compensates the small ||B|| that otherwise makes A-phase updates nearly a no-op. Set 1.0 to disable.')
    parser.add_argument('--lora_weight_decay', type=float, default=0.0, help="Weight decay applied to LoRA factors. The old 0.01 lets Adam's decay term dominate the weak A-phase gradient and shrink lora_a toward 0.")
    parser.add_argument('--lr_head', type=float, default=None, help='Classifier head LR for full fine-tuning (default: 5x --lr when finetune=full).')
    parser.add_argument('--epochs', type=int, default=3, help='Number of local epochs to run. Default is 3.')
    parser.add_argument('--n_parties', type=int, default=10, help='Number of workers in a distributed cluster. Default is 5.')
    parser.add_argument('--comm_round', type=int, default=200, help='Communication rounds per run (default: 200).')
    parser.add_argument('--early_stop_macro_f1', type=float, default=None, help='Stop after a round when client-mean test macro-F1 >= this threshold (None = disabled).')
    parser.add_argument('--init_seed', type=int, default=0, help='Seed for random number generation. Default is 0.')
    parser.add_argument('--datadir', type=str, required=False, default='./data/', help='Data directory')
    parser.add_argument('--logdir', type=str, required=False, default='./logs/', help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default='./models/', help='Model directory path')
    parser.add_argument('--sample_fraction', type=float, default=0.5, help='Fraction of clients to be sampled in each round. Default is 1.0.')
    parser.add_argument('--device', type=str, default='cuda:1', help='The device to run the program')
    parser.add_argument('--optimizer', type=str, default='adam', help="Optimizer to use for training. Default is 'adam'.")
    parser.add_argument('--initial_rank', type=int, default=10, help='initial rank for lora')
    parser.add_argument('--initial_num_layers', type=int, default=12, help='Initial layer budget G1 for round 1 (sampled uniformly without replacement).')
    parser.add_argument('--if_rl', type=str2bool, default=False, help='Whether to use RL or not. Default is False.')
    parser.add_argument('--cd', type=str, default=None, help='Whether to use lora or not. Default is False.')
    parser.add_argument('--lora', type=str2bool, choices=[True], default=True, help='Use LoRA adaptation.')
    parser.add_argument('--if_ffa', type=str2bool, default=False)
    parser.add_argument('--acc_threshold', type=int, default=0, help='threshold for rank')
    parser.add_argument('--max_rank', type=int, default=10, help='threshold for rank')
    parser.add_argument('--min_rank', type=int, default=10, help='threshold for rank')
    parser.add_argument('--model', type=str, help='model', choices=['roberta'], default='roberta')
    parser.add_argument('--out_dim', type=int, default=2)
    parser.add_argument('--data', type=str, choices=['mimic_icu10_tok512'], default='mimic_icu10_tok512', help='Prepared MIMIC-III client splits.')
    parser.add_argument('--client_partition', type=str, default='icu10', choices=['icu10', 'icu5_large'], help='ICU subset: icu10 (all 10 clients) or icu5_large (5 larger *_1500 clients).')
    parser.add_argument('--max_length', type=int, default=128, help='Tokenizer max_length for raw-text datasets.')
    parser.add_argument('--change', type=int, default=100)
    parser.add_argument('--reset_memory', type=str2bool, default=True, help='Whether to reset RL memory.')
    parser.add_argument('--regenerate_layers', type=str2bool, default=True, help='Whether to regenerate layers_output.json for new dataset.')
    parser.add_argument('--max_workers', type=int, default=0, help='Thread pool size (0 = all clients this round).')
    parser.add_argument('--gpu_devices', type=str, default='cuda:0,cuda:1', help='Comma-separated GPUs for client training, e.g. cuda:0,cuda:1')
    parser.add_argument('--per_gpu_max_parallel', type=str, default='3', help='每张 GPU 上最多同时跑几个客户端。可写单个整数（各卡相同），或按 --gpu_devices 顺序逗号分隔，如 2,4 表示 cuda:0 最多 2 路、cuda:1 最多 4 路。设为 0 表示不限制；显存不足时可改为 1 或 2,4。')
    parser.add_argument('--gpu_memory_reserve', type=str, default='', help='每张 GPU 为本进程保留的显存上限（GB），防止他人挤占。启动即占卡；训练/验证时仅缩到保底占位，轮间占满；后台线程持续补占。格式: cuda:0:75 或 cuda:0:75,cuda:1:75。')
    parser.add_argument('--gpu_memory_training_headroom', type=float, default=72.0, help='训练/验证阶段留给计算的显存（GB）。保底占位 = reserve - headroom（如 75-72=3GB）。')
    parser.add_argument('--gpu_memory_persistent_reserve', action='store_true', help='Keep the reserved CUDA allocator pool owned by this process during compute so training reuses it without exposing it to other jobs.')
    parser.add_argument('--no_client_tqdm', action='store_true', help='关闭各客户端 fit/test 的 tqdm（默认开启；多线程时多行进度可能交错，不需要展示时再加本项）。')
    parser.add_argument('--client_tqdm', action='store_true', help='兼容旧命令行：各客户端 tqdm 已默认开启，无需再传。')
    parser.add_argument('--evaluate_client_test_every_round', action='store_true', help='Evaluate each selected client on its test split after local training. Metrics are reporting-only and never enter aggregation or RL/factor feedback.')
    parser.add_argument('--epsilon_layer', type=float, default=0.2, help='Epsilon for epsilon-greedy layer selection. Default 0.2.')
    parser.add_argument('--epsilon_matrix', type=float, default=0.2, help='Epsilon for matrix-related exploration (logged in run name). Default 0.2.')
    parser.add_argument('--layer_select_mode', type=str, default='max', choices=['min', 'max'], help="Greedy layer pick: 'min' = smallest contribution first, 'max' = largest first.")
    parser.add_argument('--layer_select_method', type=str, default='rl', choices=['rl', 'all', 'fedfmsl', 'spdcfl', 'random_g'], help='Layer selection rule for the paper experiments.')
    parser.add_argument('--fedfmsl_queue_size', type=int, default=5, help='FedFMSL capability queue length Q (default 5).')
    parser.add_argument('--fedfmsl_delta', type=float, default=0.005, help='FedFMSL bottleneck threshold delta on server macro-F1 increment.')
    parser.add_argument('--spdcfl_dropout_step', type=int, default=1, help='SPD-CFL: layers to drop when SGC stagnates (default 1).')
    parser.add_argument('--spdcfl_ema_decay', type=float, default=0.9, help='SPD-CFL: EMA decay theta for SGC (default 0.9).')
    parser.add_argument('--use_mk_aggregation', action='store_true', help='Use (M/K)*|D_k|/|D| aggregation for partial client participation (Eq.3).')
    parser.add_argument('--coordinate_schedule', type=str, choices=['baba', 'none'], default='baba', help='Shared B/A update schedule.')
    parser.add_argument('--factor_selection_mode', type=str, choices=['adaptive_score', 'arc_aslora', 'fixed_baba', 'joint'], default='adaptive_score', help='LoRA factor update rule for the paper experiments.')
    parser.add_argument('--factor_warmup_rounds', type=int, default=2, help='Global adaptive factor-selection warm-up rounds.')
    parser.add_argument('--layer_random_prob', type=float, default=0.2, help='Adaptive layer-selection random branch probability (fixed 20%%).')
    parser.add_argument('--layer_score_prob', type=float, default=0.8, help='Adaptive layer-selection score branch probability (fixed 80%%).')
    parser.add_argument('--factor_epsilon', type=float, default=0.2, help='Global adaptive factor-selection epsilon-greedy exploration probability.')
    parser.add_argument('--factor_max_staleness', type=int, default=5, help='Maximum allowed rounds since a factor was last selected before forcing it.')
    parser.add_argument('--factor_score_beta', type=float, default=0.2, help='EMA beta for adaptive factor contribution scores.')
    parser.add_argument('--factor_reward_normalization', type=str, default='per_layer', choices=['per_layer'], help='Reward normalization mode for global adaptive factor score update.')
    parser.add_argument('--factor_score_clip', type=float, default=None, help='Optional symmetric clip for factor utility before EMA update.')
    parser.add_argument('--factor_credit_mode', type=str, default='equal', choices=['equal', 'rms'], help='Credit assignment for adaptive_score factor scores: equal split or RMS-weighted.')
    parser.add_argument('--aslora_ema_beta', type=float, default=OFFICIAL_EMA_BETA)
    parser.add_argument('--aslora_fd_eps', type=float, default=OFFICIAL_FD_EPS)
    parser.add_argument('--aslora_norm_eps', type=float, default=OFFICIAL_SCORE_EPS)
    parser.add_argument('--aslora_temperature_0', type=float, default=OFFICIAL_T0)
    parser.add_argument('--aslora_temperature_min', type=float, default=OFFICIAL_TMIN)
    parser.add_argument('--aslora_temperature_gamma', type=float, default=OFFICIAL_GAMMA)
    parser.add_argument('--aslora_projection_seed_offset', type=int, default=104729, help='Independent seed offset for cached Gaussian projection matrices.')
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
    parser.add_argument('--jl_rank_min_delta', type=float, default=0.0001)
    args = parser.parse_args()
    component_active = args.jl_method in METHOD_MODES
    if args.jl_component_ablation != METHOD_MODES.get(args.jl_method, 'none') or (component_active and args.jl_controller_variant != 'ddqn_s01'):
        parser.error('Component ablations require their matching method and frozen D01 controller')
    if args.jl_controller_variant and (not component_active and args.jl_method != args.jl_controller_variant or args.layer_select_method != 'rl' or args.factor_selection_mode != 'adaptive_score' or (args.factor_warmup_rounds != 2) or (not args.if_rl)):
        parser.error('Controller study requires matching method, adaptive factors and two warm-up rounds')
    if args.early_stop_macro_f1 is not None:
        parser.error('Test-based early stopping is disabled; use --jl_validation_target')
    if args.evaluate_client_test_every_round:
        parser.error('JL uses the audited full-model client runtime without local test evaluation/rank annealing')
    if args.jl_method == 'flexlora_cd':
        if args.jl_rank_policy != 'server_validation_feedback' or not args.if_rl or args.layer_select_method != 'all' or (args.factor_selection_mode != 'adaptive_score') or (args.initial_rank != 10):
            parser.error('FlexLoRA+CD requires common-rank adjustment, initial r=10, G=12 and our adaptive factor policy')
        ValidationFeedbackRank(args.initial_rank, args.min_rank, args.max_rank, args.jl_rank_step, args.jl_rank_patience, args.jl_rank_decrease_patience, args.jl_rank_min_delta, args.factor_warmup_rounds)
    elif args.jl_rank_policy != 'fixed' or (args.initial_rank, args.min_rank, args.max_rank) != (10, 10, 10):
        parser.error('All other JL LoRA methods retain fixed rank 10')
    if args.jl_method == 'fixed_g12' and (args.layer_select_method != 'all' or args.factor_selection_mode != 'adaptive_score'):
        parser.error('fixed_g12 retains our adaptive factor policy at G=12')
    if args.jl_method == 'random_g' and (args.layer_select_method != 'random_g' or args.factor_selection_mode != 'adaptive_score' or (not args.if_rl) or (args.factor_warmup_rounds != 2)):
        parser.error('random_g requires adaptive factors, fixed rank 10 and two all-layer warm-up rounds')
    if args.jl_method in ('centralized', 'local_only'):
        if args.if_rl or args.layer_select_method != 'all' or args.factor_selection_mode != 'joint' or (args.coordinate_schedule != 'none'):
            parser.error('Centralized/local-only controls jointly train all LoRA A/B factors without layer or factor selection')
    elif args.factor_selection_mode == 'joint':
        parser.error('The joint factor mode is only used by centralized/local-only controls')
    return args

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

def _log_training_eta(logger, round_idx, total_rounds, round_elapsed_sec, training_elapsed_sec, method=''):
    report_progress(round_idx + 1, total_rounds, round_elapsed_sec, training_elapsed_sec, label=method, note=FINAL_TEST_NOTE, logger=logger)

def _fed_progress_prefix(round_idx, total_rounds, client_id=None, tag=None):
    """供 tqdm / 打印使用的轮次与客户端标签。"""
    bits = [f'第{round_idx + 1}/{total_rounds}轮']
    if client_id is not None:
        bits.append(f'客户端{client_id}')
    if tag:
        bits.append(tag)
    return '[' + '|'.join(bits) + '] '

def _resolve_devices(args, n_clients):
    if args.gpu_devices:
        return [d.strip() for d in args.gpu_devices.split(',') if d.strip()]
    if torch.cuda.is_available():
        return [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    return [args.device]

def _parse_per_gpu_limits(spec, devices):
    """Parse per_gpu_max_parallel: int/str '3' or '2,4' aligned with unique devices order."""
    if spec is None:
        return None
    text = str(spec).strip()
    if not text:
        return None
    parts = [p.strip() for p in text.split(',')]
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
        raise ValueError(f'--per_gpu_max_parallel has {len(parts)} value(s) but {len(unique_devices)} GPU(s) in --gpu_devices: {unique_devices}')
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

def _execute_client_training_on_device(net, args, trainset, valiset, testset, frozen_layer, not_com_layer, device_id, prefix, round_idx, total_rounds, net_id, no_client_tqdm, tqdm_pos, as_lora_active_layers=None):
    arc_aslora_mode = str(getattr(args, 'factor_selection_mode', '')) == 'arc_aslora'
    arc_aslora_config = None
    if arc_aslora_mode:
        arc_aslora_config = {'active_layers': list(as_lora_active_layers or []), 'eta': float(args.lr), 'fd_eps': float(args.aslora_fd_eps), 'norm_eps': float(args.aslora_norm_eps), 'projection_seed': int(args.init_seed) + int(args.aslora_projection_seed_offset)}
    fit_result = fit(net, trainset, args.epochs, args.lr, args.optimizer, device=device_id, finetune=None, model='roberta', progress_prefix=prefix, disable_tqdm=no_client_tqdm, tqdm_position=tqdm_pos, max_length=args.max_length, batch_size=args.batch_size, lr_head=args.lr_head, lr_a_scale=getattr(args, 'lr_a_scale', 1.0), lora_weight_decay=getattr(args, 'lora_weight_decay', None), arc_aslora_config=arc_aslora_config, precision=getattr(args, 'jl_precision', 'fp32'), shuffle_seed=client_shuffle_seed(args.init_seed, round_idx, net_id) if args.jl_controller_variant else None)
    if arc_aslora_mode:
        train_acc, train_loss, arc_aslora_scores, arc_aslora_diagnostics = fit_result
    else:
        train_acc, train_loss = fit_result
        arc_aslora_scores, arc_aslora_diagnostics = (None, None)
    val_metrics = no_local_evaluation()
    test_metrics = dict(val_metrics)
    test_acc = test_loss = float('nan')
    layers_params = {}
    for name, param in net.named_parameters():
        if name.split('.')[-1] in ['lora_a', 'lora_b'] and name not in frozen_layer and (name not in not_com_layer):
            t = param.detach().cpu().clone()
            if param.grad is not None:
                t.grad = param.grad.detach().cpu().clone()
            layers_params[name] = t
    net.to('cpu')
    return (train_acc, train_loss, test_acc, test_loss, val_metrics, test_metrics, layers_params, arc_aslora_scores, arc_aslora_diagnostics)

def train_single_client(net_id, net, args, trainset, valiset, testset, frozen_layer, not_com_layer, device_id, log_lock, round_idx, total_rounds, gpu_semaphores, no_client_tqdm, as_lora_active_layers=None):
    prefix = _fed_progress_prefix(round_idx, total_rounds, client_id=net_id)
    tqdm_pos = min(int(net_id), 40)
    with log_lock:
        logger.info('%sTraining network %s on %s. n_training: %d' % (prefix, str(net_id), device_id, len(trainset)))
        logger.info(f'{prefix}------------第{net_id}个客户端训练-------------')
    net = freeze_layer(net, frozen_layer)
    lock_ctx = gpu_semaphores[device_id] if gpu_semaphores is not None else nullcontext()
    with lock_ctx:
        emit_progress(f'[训练阶段] {args.jl_method} | {prefix.strip()} | device={device_id}', logger=logger)
        train_acc, train_loss, test_acc, test_loss, val_metrics, test_metrics, layers_params, arc_aslora_scores, arc_aslora_diagnostics = _execute_client_training_on_device(net, args, trainset, valiset, testset, frozen_layer, not_com_layer, device_id, prefix, round_idx, total_rounds, net_id, no_client_tqdm, tqdm_pos, as_lora_active_layers=as_lora_active_layers)
    if get_gpu_memory_guard() is None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    gc.collect()
    return (net_id, layers_params, train_loss, test_loss, train_acc, test_acc, val_metrics, test_metrics, arc_aslora_scores, arc_aslora_diagnostics)

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
        d_total = float(d_total_all if d_total_all is not None else sum((len(cli_train[i]) for i in range(n_parties))))
        if d_total <= 0 or k_selected == 0:
            raise ValueError('Invalid D_total or empty client selection for M/K aggregation.')
        return [m_clients / k_selected * (len(cli_train[cid]) / d_total) for cid in client_ids]
    total_selected = sum((len(cli_train[cid]) for cid in client_ids))
    return [len(cli_train[cid]) / total_selected for cid in client_ids]

def local_train_net(nets, args, cli_train, cli_vali, cli_test, frozen_layer, not_com_layer, round_idx, total_rounds, as_lora_active_layers=None):
    devices = _resolve_devices(args, len(nets))
    max_workers = args.max_workers or len(nets)
    log_lock = threading.Lock()
    results = {}
    gpu_semaphores = _gpu_semaphores(devices, args.per_gpu_max_parallel)
    gpu_guard = get_gpu_memory_guard()
    unique_devices = list(dict.fromkeys(devices))
    if gpu_guard is not None:
        gpu_guard.enter_compute(unique_devices, logger=logger)
    no_client_tqdm = bool(args.no_client_tqdm)
    if max_workers > 1:
        limits = _parse_per_gpu_limits(args.per_gpu_max_parallel, devices)
        if limits is None:
            lim_desc = '不限制（0）'
        else:
            lim_desc = ', '.join((f'{d}={limits[d]}' for d in limits))
        logger.info('并行训练: max_workers=%s；同卡并发上限=%s。说明: gpustat/nvidia-smi 按进程汇总显存，同一 Python 进程只占一行，与是否多线程并行无关；请看 GPU-Util 百分比或显存总占用是否随训练上升。 分词: 多线程共用同一 HuggingFace Tokenizer 时批量分词已在 trainer 内加锁串行，各客户端会先后出现「分词编码…完成」属正常。 若分词完成后仍卡在训练步，可把 --per_gpu_max_parallel 设为 1 对同卡 GPU 计算串行。 各客户端 tqdm=%s（多行可能交错，不需要时加 --no_client_tqdm）。', max_workers, lim_desc, '关' if no_client_tqdm else '开(默认)')
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, (net_id, net) in enumerate(nets.items()):
            dev = devices[i % len(devices)]
            futures.append(executor.submit(train_single_client, net_id, net, args, cli_train[net_id], cli_vali[net_id], cli_test[net_id], frozen_layer, not_com_layer, dev, log_lock, round_idx, total_rounds, gpu_semaphores, no_client_tqdm, as_lora_active_layers))
        for future in concurrent.futures.as_completed(futures):
            net_id, layers_params, train_loss, test_loss, train_acc, test_acc, val_metrics, test_metrics, arc_aslora_scores, arc_aslora_diagnostics = future.result()
            results[net_id] = (layers_params, train_loss, test_loss, train_acc, test_acc, val_metrics, test_metrics, arc_aslora_scores, arc_aslora_diagnostics)
            emit_progress(f'[客户端进度] {args.jl_method} | 轮次 {round_idx + 1}/{total_rounds} | 已完成 {len(results)}/{len(nets)} 个训练客户端 | 刚完成 client={net_id}', logger=logger)
    gc.collect()
    if torch.cuda.is_available():
        for device_id in unique_devices:
            if not str(device_id).startswith('cuda'):
                continue
            with torch.cuda.device(device_id):
                persistent_reservation = bool(gpu_guard is not None and gpu_guard.persistent_reservation)
                if not persistent_reservation:
                    torch.cuda.empty_cache()
                allocated_gb = torch.cuda.memory_allocated() / 1024 ** 3
                reserved_gb = torch.cuda.memory_reserved() / 1024 ** 3
            logger.info('CUDA cache cleanup after client round: device=%s, allocated=%.2f GB, reserved=%.2f GB, persistent=%s', device_id, allocated_gb, reserved_gb, persistent_reservation)
    net_params = [results[net_id][0] for net_id in nets.keys()]
    cli_train_loss = [results[net_id][1] for net_id in nets.keys()]
    cli_test_loss = [results[net_id][2] for net_id in nets.keys()]
    cli_train_acc = [results[net_id][3] for net_id in nets.keys()]
    cli_test_acc = [results[net_id][4] for net_id in nets.keys()]
    client_metrics = {net_id: {'train_loss': results[net_id][1], 'test_loss': results[net_id][2], 'train_acc': results[net_id][3], 'test_acc': results[net_id][4], 'val_threshold': results[net_id][5]['threshold'], 'val_macro_f1': results[net_id][5]['macro_f1'], 'test_macro_f1': results[net_id][6]['macro_f1'], 'test_balanced_acc': results[net_id][6]['balanced_acc'], 'test_macro_precision': results[net_id][6]['macro_precision'], 'test_macro_recall': results[net_id][6]['macro_recall'], 'test_macro_auc': results[net_id][6]['macro_auc'], 'test_precision_cls1': results[net_id][6]['precision_cls1'], 'test_recall_cls1': results[net_id][6]['recall_cls1'], 'test_f1_cls1': results[net_id][6]['f1_cls1'], 'test_evaluated': False} for net_id in nets.keys()}
    arc_aslora_client_scores = {int(net_id): results[net_id][7] for net_id in nets.keys() if results[net_id][7] is not None}
    arc_aslora_client_diagnostics = {int(net_id): results[net_id][8] for net_id in nets.keys() if results[net_id][8] is not None}
    fed_train_loss = np.mean(cli_train_loss)
    fed_test_loss = np.mean(cli_test_loss)
    fed_train_acc = np.mean(cli_train_acc)
    fed_test_acc = np.mean(cli_test_acc)
    base_result = (net_params, fed_train_loss, fed_test_loss, fed_train_acc, fed_test_acc, client_metrics)
    if str(getattr(args, 'factor_selection_mode', '')) in {'arc_aslora'}:
        return base_result + (arc_aslora_client_scores, arc_aslora_client_diagnostics)
    return base_result

def init_nets(out_dim, n_parties, rank, model, dropout_policy='model_default', attention='eager'):
    nets = {net_i: None for net_i in range(n_parties)}
    for net_i in range(n_parties):
        net, loading_info = RobertaForSequenceClassification.from_pretrained(model_file, num_labels=out_dim, output_loading_info=True, attn_implementation=attention)
        net._jl_loading_info = loading_info
        if attention == 'sdpa' and any((type(layer.attention.self).__name__ != 'RobertaSdpaSelfAttention' for layer in net.roberta.encoder.layer)):
            raise RuntimeError('Requested SDPA, but the loaded RoBERTa uses a different attention backend')
        filter_names = [i for i in range(0, 12)]
        add_lora(net, filter_names, rank, model)
        nets[net_i] = net
        count = configure_dropout(nets[net_i], dropout_policy)
        emit_progress(f'[Model] attention={attention} | dropout={dropout_policy} | modules={count} | model_index={net_i}')
    return nets

def generate_frozen_layer(layer, matrix, model):
    all_layer_num = [i for i in range(12)]
    frozen_layer_num = [i for i in all_layer_num if i not in layer]
    frozen_layer = []
    not_com_layer = []
    not_com_layer_mark = []
    if len(frozen_layer_num) != 0:
        frozen_layer = ['roberta.encoder.layer.' + str(i) + '.attention.self.' + postfix + factor for i in frozen_layer_num for postfix in ['key.', 'value.'] for factor in ['lora_a', 'lora_b']]
        not_com_layer = frozen_layer.copy()
        not_com_layer_mark = [str(i) + postfix + factor for postfix in ['.key.', '.value.'] for i in frozen_layer_num for factor in ['0', '1']]
    if 0 in matrix.values() or 1 in matrix.values():
        for i in layer:
            if matrix[i] == 0:
                not_com_layer.append('roberta.encoder.layer.' + str(i) + '.attention.self.key.lora_b')
                not_com_layer.append('roberta.encoder.layer.' + str(i) + '.attention.self.value.lora_b')
                not_com_layer_mark.append(str(i) + '.key.1')
                not_com_layer_mark.append(str(i) + '.value.1')
            elif matrix[i] == 1:
                not_com_layer.append('roberta.encoder.layer.' + str(i) + '.attention.self.key.lora_a')
                not_com_layer.append('roberta.encoder.layer.' + str(i) + '.attention.self.value.lora_a')
                not_com_layer_mark.append(str(i) + '.key.0')
                not_com_layer_mark.append(str(i) + '.value.0')
            else:
                continue
    return (frozen_layer, not_com_layer_mark, not_com_layer)

def compute_sparsity(grad, epsilon=1e-05):
    """
    计算梯度的稀疏性
    :param grad: 梯度张量
    :param epsilon: 判断梯度接近零的阈值
    :return: 稀疏性比例
    """
    close_to_zero = (torch.abs(grad) < epsilon).float()
    sparsity = torch.mean(close_to_zero).item()
    return sparsity
COORDINATE_SCHEDULES = {'baba': [1, 0, 1, 0], 'none': []}
COORDINATE_SCHEDULE_LABELS = {'baba': 'B,A,B,A', 'none': 'N/A (no LoRA CD)'}
TREND_SLOPE_W = 5
TREND_SLOPE_P = 2
TREND_SLOPE_EPS = 0.001
TREND_SLOPE_M_MIN = 5
TREND_SLOPE_M_MAX = 20
_coordinate_schedule_list = COORDINATE_SCHEDULES['bba']
B10_WARMUP_TAIL_CYCLES = {'b10_bba': (10, [1, 1, 0]), 'b10_b2a2': (10, [1, 1, 0, 0]), 'b20_b10a10': (20, [1] * 10 + [0] * 10)}

def resolve_coordinate_schedule(name):
    key = str(name).lower().strip()
    if key not in COORDINATE_SCHEDULES:
        raise ValueError(f'Unknown coordinate_schedule={name!r}; choose from {list(COORDINATE_SCHEDULES.keys())}')
    return (COORDINATE_SCHEDULES[key], COORDINATE_SCHEDULE_LABELS[key])

def set_active_coordinate_schedule(name):
    global _coordinate_schedule_list, _trend_slope_switching_active, _b10_warmup_schedule
    key = str(name).lower().strip()
    _coordinate_schedule_list, label = resolve_coordinate_schedule(name)
    return label

def get_coordinate_factor(round_idx, schedule=None):
    """
    RLCD-FedLoRA coordinate schedule: 1 = update/aggregate B, 0 = A.
    Cycles over the list set by --coordinate_schedule (default B,B,A).
    """
    sched = schedule if schedule is not None else _coordinate_schedule_list
    return sched[round_idx % len(sched)]

def factor_name(factor_value):
    return 'B' if factor_value == 1 else 'A'

def resolve_server_validation_dataset(sst_dataset, cli_vali, logger):
    """Use dedicated server_vali when present; otherwise pooled client validation."""
    if 'server_vali' in sst_dataset:
        cen_vali = sst_dataset['server_vali']
        logger.info(f'Using dedicated server_vali (n={len(cen_vali)}) for global validation / RL reward.')
        if 'server_test' in sst_dataset:
            logger.info(f"Held-out server_test available (n={len(sst_dataset['server_test'])}).")
        return cen_vali
    cen_vali = concatenate_datasets(cli_vali)
    logger.info(f'Using pooled client vali D_val for server eval / RL (n={len(cen_vali)}).')
    return cen_vali

def factor_name_to_coordinate_value(factor_name):
    return 1 if str(factor_name).upper() == 'B' else 0

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
    suffix = 'lora_b' if str(factor_name).upper() == 'B' else 'lora_a'
    return [f'roberta.encoder.layer.{layer_idx}.attention.self.key.{suffix}', f'roberta.encoder.layer.{layer_idx}.attention.self.value.{suffix}']
    raise ValueError(f'Unsupported model for LoRA factor names: {model_name}')

def expected_upload_param_names_from_factor_map(factor_map, model_name):
    expected = set()
    for layer_idx, factor in factor_map.items():
        expected.update(lora_factor_param_names_for_layer(int(layer_idx), factor, model_name))
    return expected

def prepare_round_factor_masking(selected_layers, round_matrix, model, rl_active=False):
    frozen_layer, not_com_layer_mark, not_com_layer = generate_frozen_layer(selected_layers, round_matrix, model)
    if rl_active:
        for layer_idx in selected_layers:
            factor_value = round_matrix.get(int(layer_idx))
            if factor_value is None:
                continue
            if factor_value == 1:
                frozen_layer.extend(lora_factor_param_names_for_layer(int(layer_idx), 'A', model))
            else:
                frozen_layer.extend(lora_factor_param_names_for_layer(int(layer_idx), 'B', model))
    return (frozen_layer, not_com_layer_mark, not_com_layer)

def _layer_score_value(layer_scores, layer_id):
    if isinstance(layer_scores, dict):
        return float(layer_scores[int(layer_id)])
    return float(layer_scores[int(layer_id)])

def top_k_layers(layer_scores, k, all_layer_ids=None):
    if all_layer_ids is None:
        if isinstance(layer_scores, dict):
            all_layer_ids = sorted((int(x) for x in layer_scores.keys()))
        else:
            all_layer_ids = list(range(len(layer_scores)))
    ordered = sorted([int(layer_id) for layer_id in all_layer_ids], key=lambda layer_id: (-_layer_score_value(layer_scores, layer_id), int(layer_id)))
    return ordered[:int(k)]

def select_layers_fixed_20_80(all_layer_ids, layer_scores, budget, rng, layer_random_prob=0.2):
    all_layer_ids = [int(layer_id) for layer_id in all_layer_ids]
    budget = int(budget)
    if not 1 <= budget <= len(all_layer_ids):
        raise ValueError(f'Invalid layer budget {budget}; expected 1 <= budget <= {len(all_layer_ids)}')
    draw = rng.random()
    if draw < float(layer_random_prob):
        selected_layers = rng.sample(list(all_layer_ids), budget)
        reason = 'random_20pct'
    else:
        selected_layers = top_k_layers(layer_scores=layer_scores, k=budget, all_layer_ids=all_layer_ids)
        reason = 'score_greedy' if float(layer_random_prob) == 0 else 'score_80pct'
    return ([int(layer_id) for layer_id in selected_layers], reason, draw)

def init_factor_scores_and_counts(total_layers):
    factor_scores = {int(layer_id): {'A': 0.0, 'B': 0.0} for layer_id in range(int(total_layers))}
    factor_select_counts = {int(layer_id): {'A': 0, 'B': 0} for layer_id in range(int(total_layers))}
    return (factor_scores, factor_select_counts)

def resolve_warmup_layer_selection(round_idx, warmup_rounds, total_layers):
    if int(round_idx) < int(warmup_rounds):
        selected_layers = list(range(int(total_layers)))
        return {'warmup_active': True, 'selected_budget': len(selected_layers), 'selected_layers': selected_layers, 'layer_selection_reason': 'warmup_all_layers', 'layer_random_draw': None}
    return {'warmup_active': False, 'selected_budget': None, 'selected_layers': None, 'layer_selection_reason': None, 'layer_random_draw': None}

def resolve_all_layer_selection(total_layers):
    selected_layers = list(range(int(total_layers)))
    return {'selected_budget': len(selected_layers), 'selected_layers': selected_layers, 'layer_selection_reason': 'all_layers_baseline', 'layer_random_draw': None}
FACTOR_SELECTOR_SEED_OFFSET = 7919
ADAPTIVE_CD_LAYER_METHODS = frozenset({'rl', 'all', 'fedfmsl', 'spdcfl', 'random_g'})
SCORE_HISTORY_FACTOR_MODES = frozenset({'adaptive_score'})

def select_factor_for_layer(layer_id, factor_scores, factor_select_counts, rng, factor_random_prob=0.2):
    layer_id = int(layer_id)
    count_a = int(factor_select_counts[layer_id]['A'])
    count_b = int(factor_select_counts[layer_id]['B'])
    if count_a == 0 and count_b == 0:
        return {'factor': 'B', 'reason': 'forced_B_both_unseen', 'random_draw': None}
    if count_a == 0:
        return {'factor': 'A', 'reason': 'forced_A_unseen', 'random_draw': None}
    if count_b == 0:
        return {'factor': 'B', 'reason': 'forced_B_unseen', 'random_draw': None}
    draw = rng.random()
    if draw < float(factor_random_prob):
        factor = rng.choice(['A', 'B'])
        return {'factor': factor, 'reason': 'random_20pct', 'random_draw': draw}
    score_a = float(factor_scores[layer_id]['A'])
    score_b = float(factor_scores[layer_id]['B'])
    if score_a > score_b:
        factor = 'A'
        reason = 'score_80pct'
    elif score_b > score_a:
        factor = 'B'
        reason = 'score_80pct'
    else:
        factor = 'B' if float(factor_random_prob) == 0 else rng.choice(['A', 'B'])
        reason = 'score_tie_B' if float(factor_random_prob) == 0 else 'score_tie_random'
    return {'factor': factor, 'reason': reason, 'random_draw': draw}

def build_factor_map_for_selected_layers(selected_layers, factor_scores, factor_select_counts, rng, factor_random_prob=0.2):
    factor_map = {}
    factor_decision_metadata = {}
    for layer_id in selected_layers:
        result = select_factor_for_layer(layer_id=layer_id, factor_scores=factor_scores, factor_select_counts=factor_select_counts, rng=rng, factor_random_prob=factor_random_prob)
        layer_id = int(layer_id)
        factor_map[layer_id] = result['factor']
        factor_decision_metadata[layer_id] = result
    return (factor_map, factor_decision_metadata)

def validate_factor_upload_consistency(net_params, factor_map, model_name):
    if not factor_map:
        return
    expected_names = expected_upload_param_names_from_factor_map(factor_map, model_name)
    for client_idx, client_params in enumerate(net_params):
        client_names = set(client_params.keys())
        unexpected = sorted(client_names - expected_names)
        missing = sorted(expected_names - client_names)
        if unexpected or missing:
            raise ValueError(f'Inconsistent factor uploads for client {client_idx}: unexpected={unexpected}, missing={missing}')

def update_factor_select_counts(factor_select_counts, factor_map, aggregated_params, model_name):
    aggregated_keys = set(aggregated_params.keys())
    for layer_id, factor in factor_map.items():
        expected = set(lora_factor_param_names_for_layer(int(layer_id), factor, model_name))
        missing = sorted(expected - aggregated_keys)
        if missing:
            raise ValueError(f'Aggregation missing expected active factor parameters for layer {layer_id} factor {factor}: {missing}')
    for layer_id, factor in factor_map.items():
        factor_select_counts[int(layer_id)][str(factor).upper()] += 1

def compute_selected_factor_delta_norms(prev_lora_state, curr_model, factor_map, model_name):
    if prev_lora_state is None or not factor_map:
        return (0.0, {})
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
    return (float(total_q), per_layer_delta)

def compute_layer_factor_rms_delta(layer_id, selected_factor, global_state_before, global_state_after, model_name):
    squared_delta_sum = 0.0
    total_numel = 0
    parameter_names = lora_factor_param_names_for_layer(layer_id, selected_factor, model_name)
    if not parameter_names:
        raise RuntimeError(f'No LoRA parameters found for layer={layer_id}, factor={selected_factor}')
    for parameter_name in parameter_names:
        if parameter_name not in global_state_before or parameter_name not in global_state_after:
            raise RuntimeError(f'Missing LoRA parameter snapshot for {parameter_name} (layer={layer_id}, factor={selected_factor})')
        before = global_state_before[parameter_name]
        after = global_state_after[parameter_name]
        if tuple(before.shape) != tuple(after.shape):
            raise RuntimeError(f'Shape mismatch for {parameter_name}: before={tuple(before.shape)}, after={tuple(after.shape)}')
        delta = after.detach().to(dtype=torch.float64, device='cpu') - before.detach().to(dtype=torch.float64, device='cpu')
        squared_delta_sum += torch.sum(delta * delta).item()
        total_numel += delta.numel()
    if total_numel <= 0:
        raise RuntimeError(f'Invalid total_numel for layer={layer_id}, factor={selected_factor}')
    return math.sqrt(squared_delta_sum / total_numel)

def assign_rms_weighted_factor_credit(reward, selected_layers, factor_map, global_state_before, global_state_after, model_name, eps=1e-12):
    if not selected_layers:
        raise ValueError('selected_layers must not be empty when assigning credit')
    rms_deltas = {}
    for layer_id in selected_layers:
        factor = factor_map[layer_id]
        rms_deltas[layer_id] = compute_layer_factor_rms_delta(layer_id=layer_id, selected_factor=factor, global_state_before=global_state_before, global_state_after=global_state_after, model_name=model_name)
    total_rms_delta = float(sum(rms_deltas.values()))
    if total_rms_delta <= eps:
        equal_credit = reward / len(selected_layers)
        credits = {layer_id: float(equal_credit) for layer_id in selected_layers}
        weights = {layer_id: float(1.0 / len(selected_layers)) for layer_id in selected_layers}
        fallback_used = True
    else:
        weights = {layer_id: float(rms_deltas[layer_id] / total_rms_delta) for layer_id in selected_layers}
        credits = {layer_id: float(reward * weights[layer_id]) for layer_id in selected_layers}
        fallback_used = False
    return {'credits': credits, 'weights': weights, 'rms_deltas': rms_deltas, 'total_rms_delta': total_rms_delta, 'fallback_used': fallback_used, 'credit_mode': 'rms'}

def assign_equal_factor_credit(reward, selected_layers):
    if not selected_layers:
        raise ValueError('selected_layers must not be empty when assigning credit')
    equal_credit = float(reward) / len(selected_layers)
    credits = {layer_id: equal_credit for layer_id in selected_layers}
    weights = {layer_id: 1.0 / len(selected_layers) for layer_id in selected_layers}
    return {'credits': credits, 'weights': weights, 'rms_deltas': {layer_id: 0.0 for layer_id in selected_layers}, 'total_rms_delta': 0.0, 'fallback_used': False, 'credit_mode': 'equal'}

def _json_ready_factor_state(state_dict):
    out = {}
    for layer_id, payload in state_dict.items():
        out[str(layer_id)] = {key: float(value) if isinstance(value, (int, float, np.floating)) else value for key, value in payload.items()}
    return out

def save_adaptive_factor_checkpoint(log_dir, payload):
    if not payload:
        return
    pkl_path = os.path.join(log_dir, 'adaptive_factor_state.pkl')
    state_path = os.path.join(log_dir, 'adaptive_factor_state.json')
    rng_path = os.path.join(log_dir, 'server_rng_state.pkl')
    arc_rng_path = os.path.join(log_dir, 'arc_aslora_rng_state.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(payload, f)
    serializable = {'factor_selection_mode': payload['factor_selection_mode'], 'warmup_rounds': int(payload['warmup_rounds']), 'layer_random_prob': float(payload['layer_random_prob']), 'factor_random_prob': float(payload['factor_random_prob']), 'factor_score_beta': float(payload['factor_score_beta']), 'factor_lr': float(payload['factor_lr']), 'factor_scores': _json_ready_factor_state(payload['factor_scores']) if payload.get('factor_scores') is not None else None, 'factor_select_counts': _json_ready_factor_state(payload['factor_select_counts']) if payload.get('factor_select_counts') is not None else None, 'arc_aslora_selector_state': None}
    if payload.get('arc_aslora_selector_state') is not None:
        arc_state = payload['arc_aslora_selector_state']
        serializable['arc_aslora_selector_state'] = {'total_layers': int(arc_state['total_layers']), 'warmup_rounds': int(arc_state['warmup_rounds']), 'ema_beta': float(arc_state['ema_beta']), 'temperature_0': float(arc_state['temperature_0']), 'temperature_min': float(arc_state['temperature_min']), 'temperature_gamma': float(arc_state['temperature_gamma']), 'score_eps': float(arc_state['score_eps']), 'ema_scores': _json_ready_factor_state(arc_state['ema_scores']), 'last_temperature': float(arc_state['last_temperature'])}
    with open(state_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    with open(rng_path, 'wb') as f:
        pickle.dump(payload['server_rng_state'], f)
    if payload.get('arc_aslora_selector_state') is not None:
        with open(arc_rng_path, 'wb') as f:
            pickle.dump(payload['arc_aslora_selector_state']['random_state'], f)

def clone_lora_factor_state(model):
    state = {}
    for name, param in model.named_parameters():
        if name.split('.')[-1] in {'lora_a', 'lora_b'}:
            state[name] = param.detach().cpu().clone()
    return state

def select_layers_epsilon_greedy(layer_scores, num_layers, epsilon_layer, total_layers=12, layer_select_mode='max'):
    """
    Paper epsilon-greedy on contribution dictionary O_t.
    layer_select_mode='max': TopK by largest c_{t,l}; 'min': smallest first.
    """
    num_layers = max(1, min(int(num_layers), int(total_layers)))
    layer_scores = np.asarray(layer_scores, dtype=float)
    if random.random() < 1 - epsilon_layer:
        order = np.argsort(layer_scores)
        if str(layer_select_mode).lower() == 'max':
            return [int(i) for i in order[::-1][:num_layers].tolist()]
        return [int(i) for i in order[:num_layers].tolist()]
    return [int(i) for i in random.sample(range(total_layers), num_layers)]
LORA_AGG_MODE = None
prepare_rl_lora_round = None
aggregate_lora_server = None
LAYER_SELECT_METHOD = 'rl'
layer_selector = None

def _init_lora_baseline_hooks():
    import os
    import sys
    mode = os.environ.get('LORA_BASELINE_MODE', '').strip()
    if not mode:
        return
    from jl_baseline_hooks import _baseline_callables
    global LORA_AGG_MODE, prepare_rl_lora_round, aggregate_lora_server
    prepare_fn, aggregate_fn = _baseline_callables(mode)
    LORA_AGG_MODE = mode
    prepare_rl_lora_round = prepare_fn
    aggregate_lora_server = aggregate_fn
    main_mod = sys.modules.get('__main__')
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
    mode = os.environ.get('LAYER_SELECT_METHOD', '').strip().lower()
    if mode not in ('fedfmsl', 'spdcfl'):
        return
    global LAYER_SELECT_METHOD
    LAYER_SELECT_METHOD = mode
    targets = []
    main_mod = sys.modules.get('__main__')
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
if __name__ == '__main__':

    def load_layers_from_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            layers_list = json.load(f)
        return layers_list
    loaded_layers_list = None
    args = get_args()
    set_seed(args.init_seed)
    if args.gpu_memory_reserve:
        init_gpu_memory_guard(args.gpu_memory_reserve, logger=None, early_claim=True, training_headroom_gb=float(args.gpu_memory_training_headroom), start_watchdog=False, persistent_reservation=bool(args.gpu_memory_persistent_reserve))
    factor_selection_mode = str(getattr(args, 'factor_selection_mode', 'global_adaptive')).lower().strip()
    factor_warmup_rounds = int(getattr(args, 'factor_warmup_rounds', 2))
    layer_random_prob = float(getattr(args, 'layer_random_prob', 0.2))
    layer_score_prob = float(getattr(args, 'layer_score_prob', 0.8))
    factor_epsilon = float(getattr(args, 'factor_epsilon', 0.2))
    factor_score_beta = float(getattr(args, 'factor_score_beta', 0.2))
    factor_max_staleness = int(getattr(args, 'factor_max_staleness', 5))
    factor_reward_normalization = str(getattr(args, 'factor_reward_normalization', 'per_layer'))
    factor_score_clip = getattr(args, 'factor_score_clip', None)
    factor_credit_mode = str(getattr(args, 'factor_credit_mode', 'equal')).lower().strip()
    if factor_credit_mode not in {'equal', 'rms'}:
        raise ValueError(f'Unsupported factor_credit_mode={factor_credit_mode!r}')
    factor_random_prob = factor_epsilon
    factor_score_prob = 1.0 - factor_epsilon
    factor_lr = float(args.lr)
    aslora_ema_beta = float(args.aslora_ema_beta)
    aslora_fd_eps = float(args.aslora_fd_eps)
    aslora_norm_eps = float(args.aslora_norm_eps)
    aslora_temperature_0 = float(args.aslora_temperature_0)
    aslora_temperature_min = float(args.aslora_temperature_min)
    aslora_temperature_gamma = float(args.aslora_temperature_gamma)
    assert abs(layer_random_prob + layer_score_prob - 1.0) < 1e-08
    assert abs(factor_random_prob + factor_score_prob - 1.0) < 1e-08
    if factor_selection_mode not in {'arc_aslora'}:
        assert factor_warmup_rounds == 2
    coordinate_schedule_label = set_active_coordinate_schedule(args.coordinate_schedule)
    epsilon_layer = float(args.epsilon_layer)
    epsilon_matrix = float(args.epsilon_matrix)
    layer_select_mode = str(args.layer_select_mode).lower()
    layer_select_method = str(True).lower().strip()
    if layer_select_method not in ('rl', 'all', 'fedfmsl', 'spdcfl', 'random_g', 'none'):
        layer_select_method = 'rl'
    if layer_select_method == 'rl' and layer_select_mode != 'max':
        print(f"Overriding layer_select_mode={layer_select_mode} to 'max' for RLCD-FedLoRA.", flush=True)
        layer_select_mode = 'max'
        args.layer_select_mode = 'max'
    if factor_selection_mode == 'adaptive_score':
        if layer_select_method not in ADAPTIVE_CD_LAYER_METHODS or not args.if_rl or LORA_AGG_MODE:
            raise ValueError('adaptive_score mode is only supported for RLCD-FedLoRA with layer_select_method in {rl, all, fedfmsl, spdcfl, random_g} (if_rl=True, lora=True, no LoRA baseline hooks).')
        if factor_credit_mode not in {'equal', 'rms'}:
            raise ValueError(f'Unsupported factor_credit_mode={factor_credit_mode!r} for adaptive_score')
        effective_alr = factor_lr * float(args.lr_a_scale)
        effective_blr = factor_lr
        if abs(effective_alr - effective_blr) > 1e-12:
            raise ValueError(f'adaptive_score mode requires equal A/B learning rates: alr={effective_alr}, blr={effective_blr}. Set --lr_a_scale 1.0 or use fixed_baba mode.')
        args.lr = factor_lr
        args.lr_a_scale = 1.0
    if factor_selection_mode == 'arc_aslora':
        if layer_select_method != 'rl' or not args.if_rl or LORA_AGG_MODE:
            raise ValueError('arc_aslora is a controlled ARC-FedLoRA baseline and requires if_rl=True, lora=True, layer_select_method=rl, and no baseline hooks.')
        if abs(factor_lr * float(args.lr_a_scale) - factor_lr) > 1e-12:
            raise ValueError('arc_aslora requires equal A/B learning rates; set --lr_a_scale 1.0')
        if factor_warmup_rounds < 0:
            raise ValueError('arc_aslora factor_warmup_rounds must be non-negative')
        if not 0.0 <= aslora_ema_beta < 1.0:
            raise ValueError('aslora_ema_beta must be in [0, 1)')
        if aslora_fd_eps <= 0.0 or aslora_norm_eps <= 0.0:
            raise ValueError('AS-LoRA finite-difference epsilons must be positive')
        if aslora_temperature_0 <= 0.0 or aslora_temperature_min <= 0.0:
            raise ValueError('AS-LoRA temperatures must be positive')
        if not 0.0 < aslora_temperature_gamma <= 1.0:
            raise ValueError('aslora_temperature_gamma must be in (0, 1]')
        args.lr = factor_lr
        args.lr_a_scale = 1.0
    LAYER_SELECT_METHOD = layer_select_method
    if layer_select_method == 'fedfmsl':
        from jl_layer_selection import FedFMSLLayerSelector
        layer_selector = FedFMSLLayerSelector(queue_size=args.fedfmsl_queue_size, delta_threshold=args.fedfmsl_delta)
    elif layer_select_method == 'spdcfl':
        from jl_layer_selection import SPDCFLLayerSelector
        layer_selector = SPDCFLLayerSelector(dropout_step=args.spdcfl_dropout_step, ema_decay=args.spdcfl_ema_decay)
    else:
        layer_selector = None
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)
    now = datetime.now()
    change = int(args.change)
    _baseline_mode = LORA_AGG_MODE or os.environ.get('LORA_BASELINE_MODE', '').strip()
    baseline_tag = f'{_baseline_mode}_lora_' if _baseline_mode else ''
    if layer_select_method == 'rl':
        if factor_selection_mode in {'adaptive_score', 'arc_aslora'}:
            where_tag = f'layers_{layer_select_mode}_{factor_selection_mode}_'
        else:
            where_tag = f'layers_{layer_select_mode}_'
    elif layer_select_method == 'all':
        where_tag = f'all_layers_{factor_selection_mode}_'
    elif layer_select_method == 'none':
        where_tag = 'no_rl_'
    elif layer_select_method in {'fedfmsl', 'spdcfl'} and factor_selection_mode == 'adaptive_score':
        where_tag = f'where_{layer_select_method}_{factor_selection_mode}_'
    else:
        where_tag = f'where_{layer_select_method}_'
    sched_name = args.coordinate_schedule
    log_dir = 'log/' + now.strftime('log_%Y%m%d_%H%M%S') + '_' + baseline_tag + f'sched_{sched_name}_' + f'{args.if_rl}_' + f'{epsilon_layer}_' + f'{epsilon_matrix}_' + where_tag + f'{args.sample_fraction}_' + f'{None}_' + f'{None}_' + str(args.n_parties) + '_' + str(args.epochs) + '_dqn' + str('mimic_icu10_tok512') + '_' + str(args.initial_rank) + '_' + str(change) + '轮后训练A'
    log_dir = args.jl_output
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, 'record.log')
    fh = logging.FileHandler(log_filename)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    if args.if_rl:
        if layer_select_method == 'rl':
            logger.info('RL layer selection active; legacy cd layer-selection override disabled.')
        else:
            logger.info('Adaptive CD workflow active with layer_select_method=%s; legacy cd layer-selection override disabled.', layer_select_method)
    logger.info(f'Layer selection method: {layer_select_method}')
    if layer_select_method == 'rl':
        if args.jl_controller_variant:
            logger.info('Controller study: %s | %s', args.jl_controller_variant, VARIANTS[args.jl_controller_variant])
        else:
            logger.info(f"  Double DQN + epsilon_layer={epsilon_layer}, mode={layer_select_mode} ({('largest' if layer_select_mode == 'max' else 'smallest')} contribution first)")
    elif layer_select_method == 'fedfmsl':
        logger.info(f'  FedFMSL SAL heuristic (queue={args.fedfmsl_queue_size}, delta={args.fedfmsl_delta})')
    elif layer_select_method == 'spdcfl':
        logger.info(f'  SPD-CFL SGC budget (dropout_step={args.spdcfl_dropout_step}, ema={args.spdcfl_ema_decay})')
    elif layer_select_method == 'all':
        logger.info('  All-layer controlled baseline: all 12 transformer layers are selected every round; no DQN layer action or layer exploration is used.')
    elif layer_select_method == 'random_g':
        logger.info('Random-G ablation: two all-layer warm-up rounds, then uniform G in 1..12; private count RNG; unchanged 20/80 layer selection and adaptive factors; no DDQN.')
    if LORA_AGG_MODE:
        logger.info(f'LoRA aggregation baseline mode: {LORA_AGG_MODE}')
        print(f'LoRA aggregation baseline mode: {LORA_AGG_MODE}')
    if factor_selection_mode == 'adaptive_score':
        logger.info('Adaptive factor selection active: warmup_rounds=%d, layer_random_prob=%.2f, factor_random_prob=%.2f, factor_score_beta=%.2f, factor_credit_mode=%s, factor_lr=%.6f (legacy epsilon_layer=%.3f ignored for layer-set exploration in this mode)', factor_warmup_rounds, layer_random_prob, factor_random_prob, factor_score_beta, factor_credit_mode, factor_lr, epsilon_layer)
        print(f'Adaptive factor selection active: warmup_rounds={factor_warmup_rounds}, layer_random_prob={layer_random_prob}, factor_random_prob={factor_random_prob}, factor_score_beta={factor_score_beta}, factor_credit_mode={factor_credit_mode}, factor_lr={factor_lr}', flush=True)
    elif factor_selection_mode == 'arc_aslora':
        logger.info('ARC+AS-LoRA controlled baseline: warmup_rounds=%d, ema_beta=%.3f, fd_eps=%.3g, norm_eps=%.3g, T0=%.3f, Tmin=%.3f, gamma=%.3f, score_aggregation=uniform, parameter_aggregation=data_size_weighted, softmax_rule=official_relative_difference', factor_warmup_rounds, aslora_ema_beta, aslora_fd_eps, aslora_norm_eps, aslora_temperature_0, aslora_temperature_min, aslora_temperature_gamma)
        logger.info('AS-LoRA release/paper difference: release defaults T0=1.0,Tmin=0.1 and normalizes (SA-SB) by |SA|+|SB|; paper text reports T0=2.0,Tmin=0.2 and raw-score softmax. Release behavior is used per baseline protocol.')
        if args.evaluate_client_test_every_round:
            logger.info('Client-test reporting enabled: each client test split is evaluated every round; these metrics are excluded from aggregation, ARC layer feedback, AS-LoRA factor feedback, and server-validation checkpoint selection.')
        print(f'ARC+AS-LoRA active: warmup={factor_warmup_rounds}, ema_beta={aslora_ema_beta}, fd_eps={aslora_fd_eps}, T0={aslora_temperature_0}, Tmin={aslora_temperature_min}, gamma={aslora_temperature_gamma}', flush=True)
    rl = args.if_rl
    finetune = None
    cd = None
    if args.if_rl and (not LORA_AGG_MODE):
        logger.info(f'RLCD coordinate schedule: {coordinate_schedule_label} ({args.coordinate_schedule})')
        print(f'RLCD coordinate schedule: {coordinate_schedule_label}')
    gpu_guard = get_gpu_memory_guard()
    if gpu_guard is not None:
        gpu_guard.start_watchdog()
        logger.info('GPU memory reserve active on %s (training headroom %.0f GB, baseline hold %.0f GB); persistent=%s; watchdog keeps idle footprint, compute shrinks holds only.', ', '.join((f'{d}={gb:.0f}GB' for d, gb in gpu_guard.targets.items())), gpu_guard.training_headroom_gb, max(0.0, next(iter(gpu_guard.targets.values())) - gpu_guard.training_headroom_gb) if gpu_guard.targets else 0.0, gpu_guard.persistent_reservation)
    out_dim = args.out_dim
    model = 'roberta'
    logger.info(f'Training mode: lora={True}, if_rl={rl}, finetune={finetune}, early_stop_macro_f1={args.early_stop_macro_f1}')
    print(f'Training mode: lora={True}, if_rl={rl}, finetune={finetune}, early_stop_macro_f1={args.early_stop_macro_f1}', flush=True)
    rank = args.initial_rank
    argument_path = f'{args.n_parties}_arguments-%s.json' % datetime.now().strftime('%Y-%m-%d-%H%M-%S')
    argument_output_path = os.path.join(args.logdir, argument_path)
    os.makedirs(args.logdir, exist_ok=True)
    print(os.getcwd())
    data_file = ''
    if args.jl_data_file:
        data_file = args.jl_data_file
    else:
        data_file = 'data/mimic_icu10_server500_tokenized_512.pkl'
        split_bak = Path(data_file + '.bak')
        if split_bak.exists():
            with split_bak.open('rb') as bak_f:
                bak_dataset = pickle.load(bak_f)
            if 'server_test' in bak_dataset:
                data_file = str(split_bak)
                logger.info(f'Using split server data from {data_file}')
    with open(data_file, 'rb') as file:
        sst_dataset = pickle.load(file)
    print(sst_dataset.keys())
    if args.client_partition == 'icu5_large':
        from data.icu_client_partition import subset_icu10_to_large5
        sst_dataset = subset_icu10_to_large5(sst_dataset)
        logger.info('client_partition=icu5_large: selected clients %s', sst_dataset.get('client_names', []))
    cli_train = sst_dataset['train']
    cli_test = sst_dataset['test']
    cli_vali = sst_dataset['vali']
    client_names = sst_dataset.get('client_names', [f'client_{i}' for i in range(len(cli_train))])
    if args.n_parties != len(cli_train):
        print(f"Reset n_parties from {args.n_parties} to {len(cli_train)} for {'mimic_icu10_tok512'} clients.")
        args.n_parties = len(cli_train)
    client_names = client_names[:args.n_parties]
    use_mk_aggregation = True
    d_total_all_clients = sst_dataset.get('total_client_train_size')
    if d_total_all_clients is None:
        d_total_all_clients = sum((len(cli_train[i]) for i in range(len(cli_train))))
    cen_vali = resolve_server_validation_dataset(sst_dataset, cli_vali, logger)
    if 'server_vali' not in sst_dataset:
        raise ValueError('JL requires a dedicated server_vali split for controller feedback')
    cen_test = concatenate_datasets(cli_test)
    jl_protocol = EvaluationProtocol(args, log_dir, cli_vali, cli_test, client_names)
    write_json(Path(log_dir) / 'jl_arguments.json', vars(args))
    if use_mk_aggregation:
        logger.info(f'M/K aggregation enabled: M={args.n_parties}, |D|={d_total_all_clients}, sample_fraction={args.sample_fraction}')
    with open(argument_output_path, 'w') as f:
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
    logger.info('Initializing nets')
    global_models = init_nets(out_dim, 1, rank, model, dropout_policy=args.jl_dropout, attention=args.jl_attention)
    global_model = global_models[0]
    jl_rank_controller, jl_rank_history, jl_rank_changes = (None, [], [])
    if args.jl_rank_policy == 'server_validation_feedback':
        jl_rank_controller = ValidationFeedbackRank(rank, args.min_rank, args.max_rank, args.jl_rank_step, args.jl_rank_patience, args.jl_rank_decrease_patience, args.jl_rank_min_delta, args.factor_warmup_rounds)
        for module in global_model.modules():
            if hasattr(module, 'lora_a') and args.max_rank > min(module.lora_a.shape[0], module.lora_b.shape[1]):
                raise ValueError('Flex rank cap exceeds model adapter dimensions')
    from jl_protocol import record_initialization
    record_initialization(global_model, args, log_dir, 'fresh ' + VARIANTS[args.jl_controller_variant]['kind'] + ' controller study' if args.jl_controller_variant else 'fresh DDQN, no offline checkpoint' if layer_select_method == 'rl' and rl else None)
    nets = init_nets(out_dim, args.n_parties, rank, model, dropout_policy=args.jl_dropout, attention=args.jl_attention)
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
    arc_aslora_selector = ASLoRAComponentSelector(total_layers=total_layers, warmup_rounds=factor_warmup_rounds, ema_beta=aslora_ema_beta, temperature_0=aslora_temperature_0, temperature_min=aslora_temperature_min, temperature_gamma=aslora_temperature_gamma, score_eps=aslora_norm_eps, seed=args.init_seed + FACTOR_SELECTOR_SEED_OFFSET + 1)
    arc_aslora_best = {'round': None, 'validation_macro_f1': float('-inf'), 'validation_threshold': None, 'model_state': None}
    if factor_selection_mode == 'arc_aslora' and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(torch.device(args.device))
        except (RuntimeError, ValueError) as exc:
            logger.warning('Could not reset CUDA peak-memory stats: %s', exc)
    sparsity_A = {i: torch.tensor(0) for i in range(12)}
    sparsity_B = {i: torch.tensor(0) for i in range(12)}
    imprv_per_layer = None
    initial_num_layers = max(1, min(int(args.initial_num_layers), total_layers))
    next_round_layer = list(range(total_layers))
    next_round_budget = len(next_round_layer)
    next_round_layer_reason = 'warmup_all_layers'
    next_round_layer_random_draw = None
    random_g_schedule = None
    if layer_select_method == 'random_g':
        from jl_random_g import make_random_g_schedule
        random_g_schedule = make_random_g_schedule(args.init_seed, n_comm_rounds, total_layers, factor_warmup_rounds)
        write_json(Path(log_dir) / 'jl_random_g_schedule.json', {'seed': args.init_seed, 'policy': 'uniform integer inclusive 1..12 after warm-up', 'warmup_rounds': factor_warmup_rounds, 'budgets': random_g_schedule, 'count_rng': 'private random.Random(seed), precomputed before training'})
    cli_test_acc = {i: None for i in range(12)}
    param_upload = []
    param_caculate = []
    action_size = 12
    agent = None
    jl_study = ControllerStudy(args.jl_controller_variant, args.init_seed, n_comm_rounds, output=args.jl_output, actions=action_size) if args.jl_controller_variant else None
    if rl and layer_select_method == 'rl' and (jl_study is None):
        agent = DQNAgent(3, action_size)
        print('Resetting RL memory and starting agent from scratch.')
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
    trend_slope_state = None
    max_cpu_usages_history = []
    max_memory_usages_history = []
    avg_cpu_usages_history = []
    avg_memory_usages_history = []
    max_gpu_usages_history = []
    avg_gpu_usages_history = []
    interval = 5
    dqn_loss_history = []
    flag = 0
    all_sparsities = {i: 0 for i in range(12)}
    next_round_matrix = {i: None for i in range(12)}
    training_wall_start = time.perf_counter()
    report_progress(0, n_comm_rounds, 0, 0, label=args.jl_method, note=FINAL_TEST_NOTE, logger=logger)
    logger.info(f'RLCD setup: initial_num_layers={initial_num_layers}, coordinate_schedule={coordinate_schedule_label}, initial_layers={next_round_layer}')
    for round in range(n_comm_rounds):
        round_wall_start = time.perf_counter()
        emit_progress(f'[训练阶段] {args.jl_method} | 轮次 {round + 1}/{n_comm_rounds} | 准备本轮更新', logger=logger)
        jl_rank_reward_baseline = None
        if jl_rank_controller is not None and jl_rank_controller.rank != rank:
            emit_progress(f'[验证阶段] {args.jl_method} | 变秩后的服务器奖励基线', logger=logger)
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.manual_seed(args.init_seed + 100000 + round)
                jl_resize_details = resize_common_rank(global_model, jl_rank_controller.rank)
            logger.info('JL common rank changes before round %d: %d -> %d', round + 1, rank, jl_rank_controller.rank)
            _gpu_enter_compute(args, logger=logger)
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                _, jl_baseline_loss, jl_baseline_metrics = test(global_model, cen_vali, args, verbose=False, device=args.device, finetune=None, logger=logger, return_metrics=True, max_length=args.max_length, progress_prefix='[JL rank baseline] ', disable_tqdm=True)
            jl_rank_controller.rebase(jl_baseline_loss)
            jl_rank_reward_baseline = float(jl_baseline_metrics['macro_f1'])
            jl_rank_changes.append(dict(round=round + 1, old_rank=rank, new_rank=jl_rank_controller.rank, baseline_server_loss=float(jl_baseline_loss), baseline_server_macro_f1=jl_rank_reward_baseline, adapters=jl_resize_details))
            write_json(Path(log_dir) / 'jl_rank_changes.json', jl_rank_changes)
            rank = jl_rank_controller.rank
        next_round_matrix = {i: None for i in range(12)}
        factor_map = {}
        factor_decision_metadata = {}
        selected_budget = 0
        layer_selection_reason = ''
        layer_random_draw = None
        warmup_active = False
        factor_scores_before_round = None
        factor_counts_before_round = None
        arc_aslora_aggregated_scores = None
        update_sparsities = {}
        frozen_layer, not_com_layer_mark, not_com_layer = ([], [], [])
        param_upload_per_round = 0
        param_caculate_per_round = 0
        logger.info('in comm round:' + str(round))
        party_list_this_round = party_list_rounds[round]
        global_w = global_model.state_dict()
        nets_this_round = {k: nets[k] for k in party_list_this_round}
        for net in nets_this_round.values():
            if jl_rank_controller is not None:
                match_checkpoint_layout(net, global_w, lora_layout(global_model))
            net.load_state_dict(global_w)
        logger.info('冷冻除lora外的所有层')
        global_model = unfrozen(global_model)
        nets_this_round = unfrozen(nets_this_round)
        if rl:
            adaptive_factor_active = layer_select_method in ADAPTIVE_CD_LAYER_METHODS and factor_selection_mode == 'adaptive_score' and (prepare_rl_lora_round is None)
            arc_aslora_active = rl and layer_select_method == 'rl' and (factor_selection_mode == 'arc_aslora') and (prepare_rl_lora_round is None)
            factor_map = {}
            factor_decision_metadata = {}
            current_round_factor = None
            active_factor = None
            factor_selection_reason = ''
            factor_selector_before = None
            factor_selector_update = None
            warmup_state = resolve_warmup_layer_selection(round, factor_warmup_rounds, total_layers)
            warmup_active = bool(round < factor_warmup_rounds)
            if warmup_active and (not arc_aslora_active):
                next_round_layer = warmup_state['selected_layers']
                selected_budget = warmup_state['selected_budget']
                layer_selection_reason = warmup_state['layer_selection_reason']
                layer_random_draw = warmup_state['layer_random_draw']
            elif rl:
                next_round_layer = list(next_round_layer)
                selected_budget = int(next_round_budget)
                layer_selection_reason = next_round_layer_reason
                layer_random_draw = next_round_layer_random_draw
            else:
                next_round_layer = loaded_layers_list[round - 2]
                selected_budget = len(next_round_layer)
                layer_selection_reason = 'loaded_layers_list'
                layer_random_draw = None
            if rl and prepare_rl_lora_round is not None:
                frozen_layer, not_com_layer_mark, not_com_layer, next_round_matrix = prepare_rl_lora_round(round, next_round_layer, model, logger)
            elif rl:
                if adaptive_factor_active:
                    factor_scores_before_round = copy.deepcopy(factor_scores)
                    factor_counts_before_round = copy.deepcopy(factor_select_counts)
                    factor_map, factor_decision_metadata = build_factor_map_for_selected_layers(selected_layers=next_round_layer, factor_scores=factor_scores, factor_select_counts=factor_select_counts, rng=server_rng, factor_random_prob=factor_random_prob)
                    factor_map, factor_decision_metadata = jl_component.factors((factor_map, factor_decision_metadata), next_round_layer, round + 1)
                    jl_component.record(round + 1, next_round_layer, factor_map, layer_selection_reason, factor_decision_metadata)
                    next_round_matrix = build_round_matrix_from_factor_map(next_round_layer, factor_map, n_layers=12)
                elif arc_aslora_active:
                    factor_map, factor_decision_metadata = arc_aslora_selector.select_factor_map(active_layers=next_round_layer, round_one_based=round + 1)
                    next_round_matrix = build_round_matrix_from_factor_map(next_round_layer, factor_map, n_layers=12)
                else:
                    current_round_factor = get_coordinate_factor(round)
                    next_round_matrix = build_round_matrix(next_round_layer, current_round_factor, n_layers=12)
            elif round + 1 > change:
                next_round_matrix = {i: 0 for i in range(12)}
            else:
                next_round_matrix = {i: 1 for i in range(12)}
            logger.info(f'当前轮训练层：{next_round_layer}')
            if rl and prepare_rl_lora_round is not None:
                logger.info(f'LoRA baseline ({LORA_AGG_MODE}): {next_round_matrix}')
            elif rl:
                if adaptive_factor_active:
                    logger.info('Adaptive factor selection round=%d, warmup_active=%s, selected_budget=%d, selected_layers=%s, layer_selection_reason=%s, layer_random_draw=%s, factor_map=%s, num_A_layers=%d, num_B_layers=%d, factor_random_prob=%.2f', round + 1, warmup_active, selected_budget, next_round_layer, layer_selection_reason, layer_random_draw, factor_map, sum((1 for factor in factor_map.values() if factor == 'A')), sum((1 for factor in factor_map.values() if factor == 'B')), factor_random_prob)
                elif arc_aslora_active:
                    logger.info('ARC+AS-LoRA selection round=%d, warmup_active=%s, selected_budget=%d, selected_layers=%s, layer_selection_reason=%s, factor_map=%s, temperature=%.6f', round + 1, warmup_active, selected_budget, next_round_layer, layer_selection_reason, factor_map, arc_aslora_selector.last_temperature)
                else:
                    logger.info(f'当前轮统一更新因子：{factor_name(current_round_factor)} (schedule={coordinate_schedule_label})')
                logger.info(f'当前轮训练矩阵：{next_round_matrix}')
            else:
                logger.info(f'当前轮训练矩阵：{next_round_matrix}')
            if not (rl and prepare_rl_lora_round is not None):
                frozen_layer, not_com_layer_mark, not_com_layer = prepare_round_factor_masking(next_round_layer, next_round_matrix, model, rl_active=rl)
        elif args.if_ffa == True:
            next_round_layer = [i for i in range(12)]
            next_round_matrix = {i: 1 for i in range(12)}
            frozen_layer, not_com_layer_mark, not_com_layer = generate_frozen_layer(next_round_layer, next_round_matrix, model)
            frozen_layer = []
            for i in range(12):
                frozen_layer.append('roberta.encoder.layer.' + str(i) + '.attention.self.key.lora_a')
                frozen_layer.append('roberta.encoder.layer.' + str(i) + '.attention.self.value.lora_a')
        else:
            next_round_layer = [i for i in range(12)]
            frozen_layer, not_com_layer_mark, not_com_layer = generate_frozen_layer(next_round_layer, next_round_matrix, model)
            flag = 1 - flag
        logger.info(f'当前轮不交换的层标记：{not_com_layer_mark}')
        logger.info(f'当前轮不交换的层：{not_com_layer}')
        logger.info(f'当前轮冻结层：{frozen_layer}')
        _gpu_enter_compute(args, logger=logger)
        local_result = local_train_net(nets_this_round, args, cli_train, cli_vali, cli_test, frozen_layer, not_com_layer, round_idx=round, total_rounds=n_comm_rounds, as_lora_active_layers=next_round_layer if factor_selection_mode == 'arc_aslora' else None)
        if factor_selection_mode in {'arc_aslora'}:
            net_params, fed_train_loss, fed_test_loss, fed_train_acc, fed_test_acc, client_metrics, arc_aslora_client_scores, arc_aslora_client_diagnostics = local_result
            arc_aslora_aggregated_scores = arc_aslora_selector.update_from_clients(arc_aslora_client_scores, active_layers=next_round_layer)
            logger.info('ARC+AS-LoRA uniform score aggregation round=%d, clients=%s, raw_scores=%s, ema_scores=%s, diagnostics=%s', round + 1, sorted(arc_aslora_client_scores), arc_aslora_aggregated_scores, {layer_id: arc_aslora_selector.ema_scores[layer_id] for layer_id in next_round_layer}, arc_aslora_client_diagnostics)
        else:
            net_params, fed_train_loss, fed_test_loss, fed_train_acc, fed_test_acc, client_metrics = local_result
        cli_train_loss_history.append(fed_train_loss)
        cli_test_loss_history.append(fed_test_loss)
        cli_test_acc_history.append(fed_test_acc)
        cli_train_acc_history.append(fed_train_acc)
        train_loss_row = np.full(args.n_parties, np.nan)
        test_loss_row = np.full(args.n_parties, np.nan)
        train_acc_row = np.full(args.n_parties, np.nan)
        test_acc_row = np.full(args.n_parties, np.nan)
        for client_id, metrics in client_metrics.items():
            train_loss_row[client_id] = metrics['train_loss']
            test_loss_row[client_id] = metrics['test_loss']
            train_acc_row[client_id] = metrics['train_acc']
            test_acc_row[client_id] = metrics['test_acc']
            client_round_metrics_history.append({'round': round + 1, 'client_id': int(client_id), 'client_name': client_names[client_id] if client_id < len(client_names) else f'client_{client_id}', 'train_loss': float(metrics['train_loss']), 'train_acc': float(metrics['train_acc']), 'val_threshold': float(metrics['val_threshold']), 'val_macro_f1': float(metrics['val_macro_f1']), 'test_loss': float(metrics['test_loss']), 'test_acc': float(metrics['test_acc']), 'test_macro_f1': float(metrics['test_macro_f1']), 'test_balanced_acc': float(metrics['test_balanced_acc']), 'test_macro_precision': float(metrics['test_macro_precision']), 'test_macro_recall': float(metrics['test_macro_recall']), 'test_macro_auc': float(metrics['test_macro_auc']), 'test_precision_cls1': float(metrics['test_precision_cls1']), 'test_recall_cls1': float(metrics['test_recall_cls1']), 'test_f1_cls1': float(metrics['test_f1_cls1']), 'test_evaluated': bool(metrics.get('test_evaluated', True))})
        client_train_loss_history.append(train_loss_row)
        client_test_loss_history.append(test_loss_row)
        client_train_acc_history.append(train_acc_row)
        client_test_acc_history.append(test_acc_row)
        write_csv(Path(log_dir) / 'jl_local_train_metrics.csv', [{k: r[k] for k in ('round', 'client_id', 'client_name', 'train_loss', 'train_acc')} for r in client_round_metrics_history])
        global_model = freeze_layer(global_model, frozen_layer)
        emit_progress(f'[聚合阶段] {args.jl_method} | 轮次 {round + 1}/{n_comm_rounds}', logger=logger)
        prev_global_lora_state = clone_lora_factor_state(global_model)
        total_data_points = sum([len(cli_train[net_id]) for net_id, net in nets_this_round.items()])
        fed_avg_freqs = compute_fed_avg_weights(nets_this_round, cli_train, args.n_parties, use_mk_correction=use_mk_aggregation, d_total_all=d_total_all_clients)
        logger.info(f"Aggregation weights (selected clients {list(nets_this_round.keys())}): {[f'{w:.6f}' for w in fed_avg_freqs]} (sum={sum(fed_avg_freqs):.6f})")
        aggregated_params = {}
        aggregated_grad = {}
        if aggregate_lora_server is not None:
            aggregated_params, aggregated_grad = aggregate_lora_server(net_params, fed_avg_freqs, global_model=global_model, next_round_layer=next_round_layer, model_name=model, rank=rank, lora_alpha=8)
            n_upload_clients = len(net_params)
            for name, param in aggregated_params.items():
                param_caculate_per_round += param.numel()
                param_upload_per_round += param.numel() * n_upload_clients
        else:
            if factor_selection_mode in {'adaptive_score', 'arc_aslora'} and factor_map:
                validate_factor_upload_consistency(net_params, factor_map, model)
            logger.info('Parameter aggregation round=%d: global_base_plus_weighted_client_deltas', round + 1)
            aggregated_params, aggregated_grad, agg_param_calculate, agg_param_upload = aggregate_uploaded_deltas(net_params, fed_avg_freqs, prev_global_lora_state)
            param_caculate_per_round += agg_param_calculate
            param_upload_per_round += agg_param_upload
        if factor_selection_mode in {'arc_aslora'}:
            param_upload_per_round += 2 * len(next_round_layer) * len(nets_this_round)
        param_upload.append(param_upload_per_round)
        del net_params
        update_A = {}
        update_B = {}
        G_A = set()
        G_B = set()
        sparsities = []
        for name, param in global_model.named_parameters():
            if name in aggregated_params:
                if rl:
                    layer_number = int(name.split('.')[-5])
                    if name.split('.')[-1] == 'lora_a':
                        G_A.add((layer_number, 0))
                        if layer_number in update_A.keys():
                            update_A[layer_number] += torch.norm(param.data - aggregated_params[name], 2)
                            grad_a = aggregated_grad.get(name)
                            if grad_a is not None:
                                sparsity_A[layer_number] += grad_a
                            param_caculate_per_round += param.numel()
                        else:
                            update_A[layer_number] = torch.norm(param.data - aggregated_params[name], 2)
                            grad_a = aggregated_grad.get(name)
                            if grad_a is not None:
                                sparsity_A[layer_number] = torch.zeros_like(grad_a)
                                sparsity_A[layer_number] += grad_a
                            else:
                                sparsity_A[layer_number] = torch.tensor(0.0)
                            param_caculate_per_round += param.numel()
                    if name.split('.')[-1] == 'lora_b':
                        G_B.add((layer_number, 1))
                        grad_b = aggregated_grad.get(name)
                        if layer_number in update_B.keys():
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
                    if name.split('.')[-1] == 'lora_a':
                        G_A.add((layer_number, 0))
                    else:
                        G_B.add((layer_number, 1))
                param.data = aggregated_params[name].data.clone()
            elif name not in aggregated_params.keys() and (name.split('.')[-2] == 'key' or name.split('.')[-2] == 'value'):
                layer_number = int(name.split('.')[-5])
                if name.split('.')[-1] == 'lora_a':
                    update_A[layer_number] = 0
                else:
                    update_B[layer_number] = 0
        if rl:
            for name, grad in sparsity_A.items():
                if isinstance(grad, torch.Tensor):
                    sparsity_A[name] = torch.sum(torch.abs(grad)).item()
                else:
                    sparsity_A[name] = grad
            for name, grad in sparsity_B.items():
                if isinstance(grad, torch.Tensor):
                    sparsity_B[name] = torch.sum(torch.abs(grad)).item()
                else:
                    sparsity_B[name] = grad
            print(f'sparsity_A:{sparsity_A}')
            print(f'sparsity_B:{sparsity_B}')
            for name, grad in aggregated_grad.items():
                if grad is not None:
                    layer_number = int(name.split('.')[-5])
                    sparsity = compute_sparsity(grad)
                    sparsities.append(sparsity)
                    if layer_number not in update_sparsities.keys():
                        update_sparsities[layer_number] = abs(grad.sum().item())
                    else:
                        update_sparsities[layer_number] += abs(grad.sum().item())
            sparsities = torch.tensor(sparsities)
            all_sparsities.update(update_sparsities)
            logger.info(f'sparsities:{sparsities}')
            logger.info(f'all_sparsities:{all_sparsities}')
            mean = torch.mean(sparsities)
            var = torch.var(sparsities)
            max_val = torch.max(sparsities)
            min_val = torch.min(sparsities)
        logger.info(f'当前需要更新的层：{G_A | G_B}')
        _gpu_enter_compute(args, logger=logger)
        srv_prefix = _fed_progress_prefix(round, n_comm_rounds, tag='全局验证集')
        emit_progress(f'[验证阶段] {args.jl_method} | {srv_prefix.strip()} | server_vali 奖励评估', logger=logger)
        test_acc, test_loss, server_metrics = test(global_model, cen_vali, args, verbose=True, device=args.device, finetune=None, logger=logger, progress_prefix=srv_prefix, disable_tqdm=True, return_metrics=True, max_length=args.max_length)
        logger.info(f'{srv_prefix.strip()}本轮结束后全局验证集测试精度（server vali）：{test_acc}')
        logger.info(f"{srv_prefix.strip()}本轮结束后全局验证集 macro-F1：{server_metrics['macro_f1']}，阈值：{server_metrics['threshold']}")
        sever_test_acc_history.append(test_acc)
        server_test_loss_history.append(test_loss)
        server_macro_f1_history.append(server_metrics['macro_f1'])
        server_threshold_history.append(server_metrics['threshold'])
        if jl_rank_controller is not None:
            jl_rank_event = jl_rank_controller.observe(round + 1, test_loss)
            jl_rank_history.append(jl_rank_event)
            write_csv(Path(log_dir) / 'jl_rank_history.csv', jl_rank_history)
        jl_record = jl_protocol.validate(global_model, round + 1, selected_clients=json.dumps(party_list_this_round), selected_layers=json.dumps(next_round_layer), G=len(next_round_layer), factors=json.dumps(factor_map or next_round_matrix, sort_keys=True), aggregation_weight_sum=float(sum(fed_avg_freqs)), uplink_parameters=int(param_upload_per_round), cumulative_uplink_bytes_fp32=int(sum(param_upload) * 4), server_reward_macro_f1=float(server_metrics['macro_f1']), server_reward_loss=float(test_loss), rank_change_reward_baseline=jl_rank_reward_baseline, rank_policy=args.jl_rank_policy, proposed_next_rank=jl_rank_controller.rank if jl_rank_controller is not None else rank, controller_epsilon=jl_study.epsilon if jl_study is not None else float(agent.epsilon) if agent is not None else None, training_aggregation_reward_seconds=time.perf_counter() - round_wall_start)
        if len(sever_test_acc_history) >= 2 and abs(sever_test_acc_history[-1] - sever_test_acc_history[-2]) < 0.01:
            convergence_round.append(round)
        if factor_selection_mode in {'adaptive_score', 'arc_aslora'} and factor_map:
            update_factor_select_counts(factor_select_counts, factor_map, aggregated_params, model)
            selected_layer_ids = sorted((int(layer_idx) for layer_idx in factor_map.keys()))
            q, adaptive_layer_delta_norms = compute_selected_factor_delta_norms(prev_global_lora_state, global_model, factor_map, model)
        else:
            selected_layer_ids = sorted({layer_idx for layer_idx, _ in G_A | G_B})
            q = sum(list(update_A.values()) + list(update_B.values()))
            adaptive_layer_delta_norms = {}
        selected_layer_count = max(1, len(selected_layer_ids))
        param_caculate_per_round += len(update_A.values()) + len(update_B.values())
        reward_t = server_macro_f1_history[-1] - server_macro_f1_history[-2] if len(server_macro_f1_history) >= 2 else 0.0
        if jl_rank_reward_baseline is not None:
            reward_t = float(server_metrics['macro_f1']) - jl_rank_reward_baseline
        if len(server_macro_f1_history) >= 2 and selected_layer_ids:
            current_layer_contribution = reward_t / selected_layer_count
            for layer_idx in selected_layer_ids:
                layer_contribution_scores[layer_idx] = current_layer_contribution
        factor_credit_result = None
        if factor_selection_mode == 'adaptive_score' and factor_map:
            if factor_credit_mode == 'rms':
                global_lora_state_after = clone_lora_factor_state(global_model)
                factor_credit_result = assign_rms_weighted_factor_credit(reward=reward_t, selected_layers=selected_layer_ids, factor_map=factor_map, global_state_before=prev_global_lora_state, global_state_after=global_lora_state_after, model_name=model, eps=1e-12)
            else:
                factor_credit_result = assign_equal_factor_credit(reward=reward_t, selected_layers=selected_layer_ids)
            for layer_idx in selected_layer_ids:
                selected_factor = factor_map[int(layer_idx)]
                old_score = float(factor_scores[int(layer_idx)][selected_factor])
                factor_scores[int(layer_idx)][selected_factor] = (1.0 - factor_score_beta) * old_score + factor_score_beta * factor_credit_result['credits'][int(layer_idx)]
        imprv_per_layer = np.asarray(layer_contribution_scores, dtype=float)
        logger.info(f'imprv_per_layer:{imprv_per_layer}')
        if factor_selection_mode == 'arc_aslora' and factor_map:
            for layer_idx in range(total_layers):
                selected = int(layer_idx) in factor_map
                decision = factor_decision_metadata.get(int(layer_idx), {})
                raw_scores = arc_aslora_aggregated_scores.get(int(layer_idx), {}) if arc_aslora_aggregated_scores is not None else {}
                ema_scores = arc_aslora_selector.ema_scores[int(layer_idx)]
                arc_aslora_selection_log.append({'round': int(round + 1), 'layer_id': int(layer_idx), 'selected': bool(selected), 'selected_factor': factor_map.get(int(layer_idx), ''), 'decision_reason': decision.get('reason', 'inactive'), 'temperature': float(decision.get('temperature', arc_aslora_selector.last_temperature)), 'probability_a': decision.get('probability_a', ''), 'random_draw': decision.get('random_draw', ''), 'raw_score_a': raw_scores.get('A', ''), 'raw_score_b': raw_scores.get('B', ''), 'ema_score_a': ema_scores['A'] if ema_scores['A'] is not None else '', 'ema_score_b': ema_scores['B'] if ema_scores['B'] is not None else ''})
            logger.info('ARC+AS-LoRA factor feedback round=%d uses client gradient-curvature scores only; ARC validation reward remains reserved for layer feedback.', round + 1)
        if factor_selection_mode in SCORE_HISTORY_FACTOR_MODES and factor_map:
            logger.info('Score-history factor metrics mode=%s, round=%d, reward=%.6f, server_macro_f1=%.6f, server_loss=%.6f, q_t=%.6f, factor_map=%s', factor_selection_mode, round + 1, reward_t, server_metrics['macro_f1'], test_loss, q, factor_map)
            for layer_idx in range(total_layers):
                before_scores = factor_scores_before_round[int(layer_idx)] if factor_scores_before_round is not None else factor_scores[int(layer_idx)]
                before_counts = factor_counts_before_round[int(layer_idx)] if factor_counts_before_round is not None else factor_select_counts[int(layer_idx)]
                decision = factor_decision_metadata.get(int(layer_idx), {})
                selected = int(layer_idx) in factor_map
                assigned_credit = factor_credit_result['credits'][int(layer_idx)] if selected and factor_credit_result is not None else 0.0
                rms_delta = factor_credit_result.get('rms_deltas', {}).get(int(layer_idx), 0.0) if selected and factor_credit_result is not None else 0.0
                rms_weight = factor_credit_result['weights'][int(layer_idx)] if selected and factor_credit_result is not None and (factor_credit_result['credit_mode'] == 'rms') else 0.0
                product_delta = factor_credit_result.get('product_deltas', {}).get(int(layer_idx), 0.0) if selected and factor_credit_result is not None else 0.0
                product_weight = factor_credit_result['weights'][int(layer_idx)] if selected and factor_credit_result is not None and (factor_credit_result['credit_mode'] == 'product_delta') else 0.0
                adaptive_factor_selection_log.append({'round': round + 1, 'factor_selection_mode': factor_selection_mode, 'layer_id': int(layer_idx), 'selected': selected, 'selected_factor': factor_map.get(int(layer_idx), ''), 'decision_reason': decision.get('reason', ''), 'random_draw': decision.get('random_draw', ''), 'score_A_before': float(before_scores['A']), 'score_B_before': float(before_scores['B']), 'count_A_before': int(before_counts['A']), 'count_B_before': int(before_counts['B']), 'credit': float(assigned_credit), 'credit_mode': factor_credit_result['credit_mode'] if factor_credit_result is not None else '', 'rms_delta': float(rms_delta), 'rms_credit_weight': float(rms_weight), 'assigned_factor_credit': float(assigned_credit), 'rms_credit_fallback': bool(factor_credit_result['fallback_used'] if factor_credit_result is not None else False), 'total_round_rms_delta': float(factor_credit_result.get('total_rms_delta', 0.0) if factor_credit_result is not None else 0.0), 'effective_product_delta': float(product_delta), 'effective_product_credit_weight': float(product_weight), 'total_round_effective_product_delta': float(factor_credit_result.get('total_product_delta', 0.0) if factor_credit_result is not None else 0.0), 'credit_fallback': bool(factor_credit_result['fallback_used'] if factor_credit_result is not None else False), 'score_A_after': float(factor_scores[int(layer_idx)]['A']), 'score_B_after': float(factor_scores[int(layer_idx)]['B']), 'count_A_after': int(factor_select_counts[int(layer_idx)]['A']), 'count_B_after': int(factor_select_counts[int(layer_idx)]['B']), 'delta_norm': float(adaptive_layer_delta_norms.get(int(layer_idx), 0.0))})
        if rl:
            if layer_select_method == 'rl' and jl_study is not None:
                controller_metrics = dict(server_metrics)
                if jl_study.spec['reward_threshold'] is not None:
                    controller_metrics.update(macro_f1=server_metrics['fixed_macro_f1'], threshold=0.5)
                next_g, event = jl_study.finish_round(round + 1, (round + 1, q, test_loss), controller_metrics, len(next_round_layer))
                logger.info('ControllerStudy round=%s executed_G=%s reward=%s next_G=%s learning=%s', round + 1, event['executed_G'], event['reward'], next_g, event['learning'])
                if next_g is not None:
                    next_round_budget = next_g
                    next_round_layer, next_round_layer_reason, next_round_layer_random_draw = select_layers_fixed_20_80(all_layer_ids=list(range(total_layers)), layer_scores=imprv_per_layer, budget=next_g, rng=server_rng, layer_random_prob=layer_random_prob)
                    next_round_layer, next_round_layer_reason, next_round_layer_random_draw = jl_component.layers((next_round_layer, next_round_layer_reason, next_round_layer_random_draw), list(range(total_layers)), next_g, round + 2)
                elif round + 1 < factor_warmup_rounds:
                    next_round_budget = total_layers
                    next_round_layer = list(range(total_layers))
                    next_round_layer_reason = 'warmup_all_layers'
                    next_round_layer_random_draw = None
            elif layer_select_method == 'rl':
                param_caculate_per_round += agent.get_model_parameters_count()
                state = (round + 1, q, test_loss)
                logger.info(f'RL state components: round={round + 1}, q={q}, state_loss={test_loss}, reward={reward_t}')
                skip_dqn_transition = False
                if len(trajectory_state) != 0 and (not skip_dqn_transition):
                    agent.remember(trajectory_state[-1], len(next_round_layer) - 1, reward_t, state)
                    dqn_loss = agent.replay()
                    dqn_loss_history.append(dqn_loss)
                    logger.info(f'double_dqn_loss:{dqn_loss}')
                elif skip_dqn_transition:
                    logger.info('Skip DQN transition/action for warm-up round %d in global_adaptive mode; selected_budget is forced to all layers.', round + 1)
                should_plan_next_budget = True
                if should_plan_next_budget:
                    action = agent.act(state)
                    trajectory_state.append(state)
                    trajectory_action.append(action)
                    num_next_round_layer = action + 1
                    if round == n_comm_rounds - 1:
                        agent.save_models(os.path.join(log_dir, 'q_network.pth'), os.path.join(log_dir, 'target_network.pth'), os.path.join(log_dir, 'memory.pkl'))
                    logger.info(f'num_next_round_layer:{num_next_round_layer}')
                    next_round_budget = int(num_next_round_layer)
                else:
                    next_round_budget = total_layers
                    next_round_layer = list(range(total_layers))
                    next_round_layer_reason = 'warmup_all_layers'
                    next_round_layer_random_draw = None
                    logger.info('Keep next round at full layer set because global_adaptive warm-up continues (next_round_budget=%d).', next_round_budget)
                if should_plan_next_budget and factor_selection_mode in {'adaptive_score', 'arc_aslora'}:
                    next_round_layer, next_round_layer_reason, next_round_layer_random_draw = select_layers_fixed_20_80(all_layer_ids=list(range(total_layers)), layer_scores=imprv_per_layer, budget=next_round_budget, rng=server_rng, layer_random_prob=layer_random_prob)
                    logger.info('Adaptive next_round_layer: budget=%d, layers=%s, reason=%s, random_draw=%s', next_round_budget, next_round_layer, next_round_layer_reason, next_round_layer_random_draw)
                elif should_plan_next_budget and len(server_macro_f1_history) < 2:
                    next_round_layer = random.sample(range(total_layers), num_next_round_layer)
                    next_round_layer_reason = 'legacy_random_warm_start'
                    next_round_layer_random_draw = None
                elif should_plan_next_budget:
                    next_round_layer = select_layers_epsilon_greedy(imprv_per_layer, num_next_round_layer, epsilon_layer=epsilon_layer, total_layers=total_layers, layer_select_mode=layer_select_mode)
                    next_round_layer_reason = 'legacy_epsilon_greedy'
                    next_round_layer_random_draw = None
                logger.info(f'next_round_layer (largest-contribution first):{next_round_layer}')
            elif layer_select_method == 'random_g':
                if round + 1 < n_comm_rounds:
                    next_round_budget = random_g_schedule[round + 1]
                    if round + 1 < factor_warmup_rounds:
                        next_round_layer = list(range(total_layers))
                        next_round_layer_reason = 'warmup_all_layers'
                        next_round_layer_random_draw = None
                    else:
                        next_round_layer, next_round_layer_reason, next_round_layer_random_draw = select_layers_fixed_20_80(all_layer_ids=list(range(total_layers)), layer_scores=imprv_per_layer, budget=next_round_budget, rng=server_rng, layer_random_prob=layer_random_prob)
                    logger.info('Random-G plan: round=%d G=%d layers=%s reason=%s', round + 2, next_round_budget, next_round_layer, next_round_layer_reason)
            elif layer_select_method == 'fedfmsl' and layer_selector is not None:
                next_round_layer = layer_selector.select_next_layers(round, float(server_macro_f1_history[-1]))
            elif layer_select_method == 'spdcfl' and layer_selector is not None:
                next_round_layer = layer_selector.select_next_layers(round, imprv_per_layer)
            elif layer_select_method == 'all':
                all_layer_state = resolve_all_layer_selection(total_layers)
                next_round_layer = all_layer_state['selected_layers']
                next_round_budget = all_layer_state['selected_budget']
                next_round_layer_reason = all_layer_state['layer_selection_reason']
                next_round_layer_random_draw = all_layer_state['layer_random_draw']
                logger.info('All-layer baseline keeps next_round_budget=%d and next_round_layer=%s.', next_round_budget, next_round_layer)
            else:
                next_round_layer = list(range(12))
            if prepare_rl_lora_round is not None:
                _, _, _, preview_next_round_matrix = prepare_rl_lora_round(round + 1, next_round_layer, model, logger)
                logger.info(f'下一轮 LoRA baseline ({LORA_AGG_MODE}): {preview_next_round_matrix}')
            else:
                preview_next_round_matrix = None
                if factor_selection_mode in {'adaptive_score', 'arc_aslora'} and factor_map:
                    logger.info('下一轮 adaptive factor selection pending server decision; selected layers will be %s', next_round_layer)
                else:
                    next_round_factor = get_coordinate_factor(round + 1)
                if preview_next_round_matrix is not None:
                    preview_next_round_matrix = build_round_matrix(next_round_layer, next_round_factor, n_layers=12)
                    logger.info(f'下一轮统一更新因子：{factor_name(next_round_factor)} (schedule={coordinate_schedule_label}); next_round_matrix:{preview_next_round_matrix}')
            should_save_adaptive_state = factor_selection_mode in SCORE_HISTORY_FACTOR_MODES and layer_select_method in ADAPTIVE_CD_LAYER_METHODS or (factor_selection_mode == 'arc_aslora' and layer_select_method == 'rl')
            if should_save_adaptive_state:
                save_adaptive_factor_checkpoint(log_dir, {'factor_selection_mode': factor_selection_mode, 'warmup_rounds': factor_warmup_rounds, 'layer_random_prob': layer_random_prob, 'factor_random_prob': factor_random_prob, 'factor_score_beta': factor_score_beta, 'factor_lr': factor_lr, 'factor_scores': factor_scores, 'factor_select_counts': factor_select_counts, 'arc_aslora_selector_state': arc_aslora_selector.state_dict() if factor_selection_mode == 'arc_aslora' else None, 'server_rng_state': server_rng.getstate()})
        param_caculate.append(param_caculate_per_round)
        _gpu_enter_idle(args, logger=logger)
        _log_training_eta(logger, round, n_comm_rounds, time.perf_counter() - round_wall_start, time.perf_counter() - training_wall_start, method=args.jl_method)
        if args.jl_validation_target is not None and jl_record['validation_macro_f1'] >= args.jl_validation_target:
            write_json(Path(log_dir) / 'jl_validation_target.json', {'target': args.jl_validation_target, 'round': round + 1, 'validation_macro_f1': jl_record['validation_macro_f1'], 'uplink_bytes_fp32': int(sum(param_upload) * 4)})
            break
    logger.debug(f'param_caculate:{param_caculate}')
    logger.debug(f'param_caculate:{sum(param_caculate)}')
    logger.debug(f'param_upload:{param_upload}')
    logger.debug(f'param_upload:{sum(param_upload)}')
    jl_summary = jl_protocol.finalize(global_model)
    write_csv(Path(log_dir) / 'jl_factor_history.csv', adaptive_factor_selection_log)
    write_csv(Path(log_dir) / 'jl_aslora_history.csv', arc_aslora_selection_log)
    if agent is not None:
        torch.save({'q': agent.q_network.state_dict(), 'target': agent.target_network.state_dict(), 'optimizer': agent.optimizer.state_dict(), 'epsilon': agent.epsilon, 'memory': list(agent.memory), 'factor_scores': factor_scores, 'factor_counts': factor_select_counts, 'layer_scores': layer_contribution_scores}, os.path.join(log_dir, 'jl_controller_last.pt'))
    logger.info('JL final validation-selected result: %s', jl_summary)
    raise SystemExit(0)
