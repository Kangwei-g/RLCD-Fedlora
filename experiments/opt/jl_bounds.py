"""Centralized/local-only controls with validation-selected checkpoints."""

import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time

import numpy as np
from datasets import concatenate_datasets

import jl_engine as base
from jl_data import load_data
from jl_protocol import (EvaluationProtocol, binary_metrics, save_predictions,
                         write_csv, write_json, record_initialization)
from jl_trainer import fit
from jl_progress import emit_progress, report_progress, FINAL_TEST_NOTE


def train_epoch(model, dataset, args):
    # Both factors in every adapter train together; backbone/head stay frozen.
    base.unfrozen(model)
    acc, loss = fit(model, dataset, args.epochs, args.lr, args.optimizer,
                    args.device, args.finetune, args.model, disable_tqdm=True,
                    batch_size=args.batch_size, max_length=args.max_length,
                    lr_a_scale=args.lr_a_scale, lora_weight_decay=args.lora_weight_decay,
                    precision=getattr(args, 'jl_precision', 'fp32'))
    model.to('cpu')
    return acc, loss


def run(mode):
    args = base.get_args()
    base.set_seed(args.init_seed)
    base.if_lora, base.finetune = args.lora, args.finetune
    data = load_data(args.jl_data_file, args.jl_setting)
    output = Path(args.jl_output)
    write_json(output / 'jl_arguments.json', vars(args))
    model = base.init_nets(args.out_dim, 1, args.initial_rank, args.model,
                           dropout_policy=args.jl_dropout, attention=args.jl_attention)[0]
    record_initialization(model, args, output, None)
    training_seconds = 0.0
    if mode == 'centralized':
        protocol = EvaluationProtocol(args, output, data['vali'], data['test'], data['client_names'])
        dataset = concatenate_datasets(data['train'])
        report_progress(0, args.comm_round, 0, 0, label=mode, note=FINAL_TEST_NOTE)
        for r in range(args.comm_round):
            emit_progress(f"[Training] {mode} | round {r + 1}/{args.comm_round} | pooled client training")
            start = time.perf_counter()
            acc, loss = train_epoch(model, dataset, args)
            protocol.validate(model, r + 1, train_accuracy=acc, train_loss=loss,
                              training_seconds=time.perf_counter() - start, G=12)
            round_seconds = time.perf_counter() - start
            training_seconds += round_seconds
            report_progress(r + 1, args.comm_round, round_seconds, training_seconds,
                            label=mode, note=FINAL_TEST_NOTE)
        protocol.finalize(model)
        return
    # All clients start from the same seed/head but train independently; no averaging.
    rows, predictions, chosen_predictions, selections = [], [], [], {}
    total_rounds = len(data['client_names']) * args.comm_round
    completed_rounds = 0
    progress_lock = threading.Lock()
    overall_start = time.perf_counter()
    report_progress(0, total_rounds, 0, 0, label=mode,
                    unit="\u5ba2\u6237\u7aef\u8f6e\u6b21", note=FINAL_TEST_NOTE)

    def run_client(item):
        nonlocal completed_rounds
        cid, name = item
        local_model = copy.deepcopy(model)
        protocol = EvaluationProtocol(args, output / ('client_%02d' % cid),
                                      [data['vali'][cid]], [data['test'][cid]], [name])
        for r in range(args.comm_round):
            context = (f"{mode} | client {cid + 1}/{len(data['client_names'])} ({name})"
                       f" | client round {r + 1}/{args.comm_round}")
            emit_progress(f"[Training] {context}")
            start = time.perf_counter()
            acc, loss = train_epoch(local_model, data['train'][cid], args)
            protocol.validate(local_model, r + 1, train_accuracy=acc, train_loss=loss,
                              training_seconds=time.perf_counter() - start, G=12)
            round_seconds = time.perf_counter() - start
            with progress_lock:
                completed_rounds += 1
                report_progress(completed_rounds, total_rounds,
                                round_seconds, time.perf_counter() - overall_start, label=context,
                                unit="\u5ba2\u6237\u7aef\u8f6e\u6b21", note=FINAL_TEST_NOTE)
        result = protocol.finalize(local_model)
        prediction = protocol.final_predictions[0]
        prediction['client_id'] = cid
        threshold = result['selection']['threshold']
        metrics = binary_metrics(prediction['labels'], prediction['probabilities'], threshold, prediction['loss'])
        return (cid, result['selection'], prediction,
                dict(client_id=cid, client_name=name, round=result['selection']['round'], **metrics))

    gpu_limit = int(str(args.per_gpu_max_parallel).split(',')[0])
    workers = min(args.max_workers, gpu_limit if gpu_limit > 0 else args.max_workers,
                  len(data['client_names']))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_client, enumerate(data['client_names'])))
    for cid, selection, prediction, row in results:
        selections[str(cid)] = selection
        threshold = selection['threshold']
        rows.append(row)
        predictions.append(prediction)
        chosen_predictions.extend((prediction['probabilities'] >= threshold).astype(int).tolist())
    y = np.concatenate([x['labels'] for x in predictions])
    p = np.concatenate([x['probabilities'] for x in predictions])
    pooled = binary_metrics(y, p, 0.5, sum(x['loss'] * len(x['labels']) for x in predictions) / len(y))
    from jl_protocol import confusion_metrics
    pooled.update(confusion_metrics(y, chosen_predictions, 0.5))
    pooled['threshold'] = None
    pooled['decision_loss_per_10000'] = 10000 * ((7 / 3) * pooled['fn'] + pooled['fp']) / len(y)
    keys = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'balanced_accuracy',
            'auroc', 'auprc', 'loss', 'net_benefit_p030', 'decision_loss_per_10000']
    means = {key: float(np.mean([r[key] for r in rows])) if all(r[key] is not None for r in rows) else None
             for key in keys}
    write_csv(output / 'jl_client_test_metrics.csv', rows)
    save_predictions(output / 'jl_test_predictions.npz', predictions)
    write_json(output / 'jl_summary.json', dict(method=args.jl_method, setting=args.jl_setting,
               seed=args.init_seed, protocol='independent_client_validation_selection_v1',
               selections=selections, client_mean=means, pooled=pooled,
               n_clients=len(rows), n_test=len(y)))
