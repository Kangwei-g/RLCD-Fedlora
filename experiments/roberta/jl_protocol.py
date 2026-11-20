"""Validation-only selection and explicit metrics for JL experiment runs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from jl_trainer import prepare_tensor_dataloader
from jl_rank import common_rank, lora_layout, match_checkpoint_layout
from jl_progress import emit_progress, report_progress

THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_predictions(labels, probabilities):
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Expected nonempty equally sized one-dimensional predictions")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all():
        raise ValueError("Only finite binary-class predictions are supported")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be within [0, 1]")
    return y, p


def confusion_metrics(labels, probabilities, threshold):
    y, p = checked_predictions(labels, probabilities)
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be within [0, 1]")
    pred = p >= threshold
    tp = int(np.sum((y == 1) & pred))
    tn = int(np.sum((y == 0) & ~pred))
    fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred))
    div = lambda a, b: a / b if b else 0.0
    p1, p0 = div(tp, tp + fp), div(tn, tn + fn)
    r1, r0 = div(tp, tp + fn), div(tn, tn + fp)
    f1, f0 = div(2 * tp, 2 * tp + fp + fn), div(2 * tn, 2 * tn + fp + fn)
    return {"n": len(y), "positive_n": tp + fn, "tn": tn, "fp": fp,
            "fn": fn, "tp": tp, "accuracy": (tp + tn) / len(y),
            "macro_precision": (p0 + p1) / 2, "macro_recall": (r0 + r1) / 2,
            "macro_f1": (f0 + f1) / 2, "balanced_accuracy": (r0 + r1) / 2,
            "precision_cls1": p1, "recall_cls1": r1, "f1_cls1": f1}


def binary_metrics(labels, probabilities, threshold, loss=None):
    y, p = checked_predictions(labels, probabilities)
    result = confusion_metrics(y, p, threshold)
    both_classes = len(np.unique(y)) == 2
    result.update({"threshold": float(threshold),
                   "auroc": float(roc_auc_score(y, p)) if both_classes else None,
                   "auprc": float(average_precision_score(y, p)) if np.any(y == 1) else None,
                   "loss": float(loss) if loss is not None else None})
    pt = 0.30
    clinical = confusion_metrics(y, p, pt)
    result["net_benefit_p030"] = (clinical["tp"] - clinical["fp"] * pt / (1 - pt)) / len(y)
    result["net_benefit_treat_all_p030"] = float(y.mean() - (1 - y.mean()) * pt / (1 - pt))
    result["net_benefit_treat_none"] = 0.0
    # Existing manuscript scenario: FN/FP cost ratio 7/3, not an observed clinical cost.
    result["decision_loss_per_10000"] = 10000 * ((7 / 3) * result["fn"] + result["fp"]) / len(y)
    return result


def select_shared_threshold(predictions, thresholds=THRESHOLDS):
    if not predictions:
        raise ValueError("At least one client validation split is required")
    thresholds = sorted(set(float(t) for t in thresholds))
    if not thresholds:
        raise ValueError("Threshold grid cannot be empty")
    scores = [np.mean([confusion_metrics(x["labels"], x["probabilities"], t)["macro_f1"]
                       for x in predictions]) for t in thresholds]
    # Ascending grid and first argmax define the threshold tie rule in advance.
    i = int(np.argmax(scores))
    return thresholds[i], float(scores[i])


def summarize_predictions(predictions, threshold):
    rows = [dict(client_id=x["client_id"], client_name=x["client_name"],
                 **binary_metrics(x["labels"], x["probabilities"], threshold, x.get("loss")))
            for x in predictions]
    keys = [k for k in rows[0] if k not in {"client_id", "client_name", "threshold", "n", "positive_n", "tn", "tp", "fn", "fp"}]
    means = {k: float(np.mean([r[k] for r in rows])) if all(r[k] is not None for r in rows) else None
             for k in keys}
    y = np.concatenate([x["labels"] for x in predictions])
    p = np.concatenate([x["probabilities"] for x in predictions])
    losses = [x.get("loss") for x in predictions]
    loss = sum(len(x["labels"]) * x["loss"] for x in predictions) / len(y) if all(v is not None for v in losses) else None
    return rows, {"client_mean": means, "pooled": binary_metrics(y, p, threshold, loss)}


def save_predictions(path, predictions):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{
        "client_%d_%s" % (x["client_id"], field): np.asarray(x[field])
        for x in predictions for field in ("labels", "probabilities")})


def selected_state(model, lora=True):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            if not lora or name.endswith(("lora_a", "lora_b")) or name.startswith("classifier.")}


def record_initialization(model, args, output, controller):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.startswith('classifier.'):
            digest.update(value.detach().cpu().numpy().tobytes())
    missing = getattr(model, '_jl_loading_info', {}).get('missing_keys', [])
    write_json(Path(output) / 'jl_initialization.json', {
        'head_sha256': digest.hexdigest(),
        'head_frozen': all(not p.requires_grad for n, p in model.named_parameters() if n.startswith('classifier.')),
        'head_initialization': 'from_pretrained; missing head weights use seeded transformers initialization',
        'missing_pretrained_keys': missing, 'seed': args.init_seed,
        'controller': controller, 'warmup_rounds': args.factor_warmup_rounds if controller else 0})


def save_checkpoint(path, model, args, **metadata):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(dict(model_state_dict=selected_state(model, args.lora),
                    backbone=args.jl_model_path, lora=args.lora,
                    lora_layout=lora_layout(model), **metadata), temp)
    temp.replace(path)


def load_checkpoint(path, model):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint['lora']:
        match_checkpoint_layout(model, checkpoint['model_state_dict'], checkpoint.get('lora_layout'))
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if unexpected or (missing and not checkpoint["lora"]) or any(n.endswith(("lora_a", "lora_b")) or n.startswith("classifier.") for n in missing):
        raise RuntimeError("Checkpoint does not match the evaluation model")
    return checkpoint


class EvaluationProtocol:
    def __init__(self, args, output, validation, tests, names):
        self.args, self.output = args, Path(output)
        self.validation, self.tests, self.names = validation, tests, names
        if not len(names) == len(validation) == len(tests) or not names:
            raise ValueError("Client validation/test collections must match")
        self.best = None
        self.rounds, self.client_rows = [], []
        self.loaders = {}
        self.started = time.perf_counter()
        self.progress_context = getattr(args, "jl_method", "evaluation")
        self.output.mkdir(parents=True, exist_ok=True)

    def predict(self, model, datasets, split):
        stage = "Validation" if split == "validation" else "Test"
        emit_progress(f"[{stage}] {self.progress_context} | {len(datasets)} client(s)")
        stage_start = time.perf_counter()
        original_device = next(model.parameters()).device
        training = model.training
        device = torch.device(self.args.device)
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        predictions = []
        # Validation must not change dropout/training RNG streams.
        with torch.random.fork_rng(devices=cuda_devices):
            model.to(device).eval()
            try:
                for cid, dataset in enumerate(datasets):
                    client_start = time.perf_counter()
                    context = f"{self.progress_context} | client {cid + 1}/{len(datasets)} ({self.names[cid]})"
                    emit_progress(f"[{stage}] {context} | inference")
                    key = (split, cid)
                    if key not in self.loaders:
                        self.loaders[key] = prepare_tensor_dataloader(
                            dataset, self.args.model, self.args.max_length, "[JL] ", split,
                            self.args.jl_eval_batch_size, False)
                    labels, probabilities, total_loss = [], [], 0.0
                    with torch.no_grad():
                        for ids, mask, y in self.loaders[key]:
                            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
                            total_loss += torch.nn.functional.cross_entropy(logits, y.to(device), reduction="sum").item()
                            labels.extend(y.tolist())
                            probabilities.extend(torch.softmax(logits.float(), dim=1)[:, 1].cpu().tolist())
                    y, p = checked_predictions(labels, probabilities)
                    predictions.append(dict(client_id=cid, client_name=self.names[cid],
                                            labels=y, probabilities=p, loss=total_loss / len(y)))
                    report_progress(cid + 1, len(datasets), time.perf_counter() - client_start,
                                    time.perf_counter() - stage_start, label=context,
                                    heading=stage, unit="\u5ba2\u6237\u7aef", step_name="\u5ba2\u6237\u7aef",
                                    note="\u4ec5\u4f30\u8ba1\u5f53\u524d\u8bc4\u4f30\u9636\u6bb5")
            finally:
                model.to(original_device).train(training)
        return predictions

    def validate(self, model, round_number, **diagnostics):
        self.progress_context = f"{self.args.jl_method} | round {round_number}/{getattr(self.args, 'comm_round', '?')}"
        start = time.perf_counter()
        predictions = self.predict(model, self.validation, "validation")
        threshold, score = select_shared_threshold(predictions)
        rows, summary = summarize_predictions(predictions, threshold)
        save_predictions(self.output / "jl_validation_predictions" / ("round_%04d.npz" % round_number), predictions)
        for row in rows:
            source = self.args.jl_method if self.args.jl_method in ("centralized", "local_only") else "aggregated_global"
            row.update(round=round_number, split="validation", model_source=source)
        self.client_rows.extend(rows)
        record = dict(round=round_number, threshold=threshold,
                      lora_rank=common_rank(model),
                      validation_macro_f1=score, validation_seconds=time.perf_counter() - start,
                      elapsed_seconds=time.perf_counter() - self.started, **diagnostics)
        record.update({"validation_" + k: v for k, v in summary["client_mean"].items()})
        self.rounds.append(record)
        if self.best is None or score > self.best["validation_macro_f1"]:
            self.best = dict(round=round_number, threshold=threshold, validation_macro_f1=score,
                             lora_rank=common_rank(model))
            save_checkpoint(self.output / "jl_best.pt", model, self.args, **self.best)
            write_json(self.output / "jl_selection.json", self.best)
        write_csv(self.output / "jl_round_metrics.csv", self.rounds)
        write_csv(self.output / "jl_client_validation_metrics.csv", self.client_rows)
        if self.args.jl_test_every_round:
            test_predictions = self.predict(model, self.tests, "test")
            save_predictions(self.output / "jl_diagnostic_test_predictions" / ("round_%04d.npz" % round_number), test_predictions)
        return record

    def finalize(self, model):
        if self.best is None:
            raise RuntimeError("Cannot test before validation selection")
        self.progress_context = f"{self.args.jl_method} | selected round {self.best['round']} | final test"
        emit_progress(f"[Finalizing] {self.progress_context} | saving last / restoring best checkpoint")
        save_checkpoint(self.output / "jl_last.pt", model, self.args, round=self.rounds[-1]["round"])
        load_checkpoint(self.output / "jl_best.pt", model)
        predictions = self.predict(model, self.tests, "test")
        self.final_predictions = predictions
        rows, summary = summarize_predictions(predictions, self.best["threshold"])
        for row in rows:
            row.update(round=self.best["round"], split="test", model_source="validation_selected")
        save_predictions(self.output / "jl_test_predictions.npz", predictions)
        write_csv(self.output / "jl_client_test_metrics.csv", rows)
        summary.update(method=self.args.jl_method, seed=self.args.init_seed,
                       setting=self.args.jl_setting, selection=self.best,
                       n_clients=len(rows), n_test=sum(r["n"] for r in rows),
                       protocol="client_mean_validation_shared_threshold_v1",
                       elapsed_seconds=time.perf_counter() - self.started)
        write_json(self.output / "jl_summary.json", summary)
        emit_progress(f"[Finalizing] {self.progress_context} | results saved: {self.output}")
        return summary


def no_local_evaluation():
    keys = ("threshold", "macro_f1", "balanced_acc", "macro_precision", "macro_recall",
            "macro_auc", "precision_cls1", "recall_cls1", "f1_cls1")
    return {key: float("nan") for key in keys}
