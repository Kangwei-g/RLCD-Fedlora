"""Data-agnostic local training and binary evaluation helpers."""

from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import nn

from .masking import set_trainable_factor_map


def _move_batch(batch, device: torch.device):
    if isinstance(batch, Mapping):
        moved = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        labels = moved.pop("labels", moved.pop("label", None))
        if labels is None:
            raise ValueError("A mapping batch must contain `labels` or `label`")
        return moved, labels.long()

    if isinstance(batch, (tuple, list)) and len(batch) == 3:
        input_ids, attention_mask, labels = batch
        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
        }, labels.to(device).long()
    raise TypeError("Batch must be a mapping or (input_ids, attention_mask, labels)")


def _logits(model: nn.Module, features: Mapping[str, torch.Tensor]) -> torch.Tensor:
    output = model(**features)
    if hasattr(output, "logits"):
        return output.logits
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise TypeError("Model output does not expose logits")


def train_selected_factors(
    model: nn.Module,
    train_loader: Iterable,
    factor_map: Mapping[int, str],
    learning_rate: float,
    epochs: int = 1,
    device: str = "cpu",
    model_name: str = "roberta",
    weight_decay: float = 0.0,
    class_weights: Optional[torch.Tensor] = None,
) -> float:
    """Fit only the factors named by `factor_map`; all other parameters freeze."""
    set_trainable_factor_map(model, factor_map, model_name=model_name)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("No trainable LoRA parameters were selected")

    target_device = torch.device(device)
    model.to(target_device)
    optimizer = torch.optim.Adam(
        trainable, lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    loss_weight = class_weights.to(target_device) if class_weights is not None else None

    total_loss = 0.0
    total_examples = 0
    model.train()
    for _ in range(int(epochs)):
        for batch in train_loader:
            features, labels = _move_batch(batch, target_device)
            logits = _logits(model, features)
            loss = nn.functional.cross_entropy(logits, labels, weight=loss_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu().item()) * labels.numel()
            total_examples += labels.numel()
    if total_examples == 0:
        raise ValueError("The local training loader is empty")
    return total_loss / total_examples


def collect_binary_predictions(
    model: nn.Module, loader: Iterable, device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray, float]:
    target_device = torch.device(device)
    model.to(target_device)
    model.eval()
    labels_all = []
    probabilities_all = []
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch in loader:
            features, labels = _move_batch(batch, target_device)
            logits = _logits(model, features)
            loss = nn.functional.cross_entropy(logits, labels)
            probabilities = torch.softmax(logits, dim=-1)[:, 1]
            labels_all.append(labels.detach().cpu().numpy())
            probabilities_all.append(probabilities.detach().cpu().numpy())
            total_loss += float(loss.detach().cpu().item()) * labels.numel()
            total_examples += labels.numel()
    if total_examples == 0:
        raise ValueError("The evaluation loader is empty")
    return (
        np.concatenate(labels_all).astype(np.int64),
        np.concatenate(probabilities_all).astype(np.float64),
        total_loss / total_examples,
    )


def _class_metrics(labels: np.ndarray, predictions: np.ndarray, class_id: int):
    true_positive = int(np.sum((labels == class_id) & (predictions == class_id)))
    false_positive = int(np.sum((labels != class_id) & (predictions == class_id)))
    false_negative = int(np.sum((labels == class_id) & (predictions != class_id)))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


def _binary_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    positive = labels == 1
    negative = labels == 0
    num_positive = int(np.sum(positive))
    num_negative = int(np.sum(negative))
    if num_positive == 0 or num_negative == 0:
        return float("nan")

    order = np.argsort(probabilities, kind="mergesort")
    sorted_probabilities = probabilities[order]
    ranks = np.empty(len(probabilities), dtype=np.float64)
    start = 0
    while start < len(probabilities):
        end = start + 1
        while (
            end < len(probabilities)
            and sorted_probabilities[end] == sorted_probabilities[start]
        ):
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(np.sum(ranks[positive]))
    return (
        rank_sum - num_positive * (num_positive + 1) / 2.0
    ) / (num_positive * num_negative)


def binary_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> Dict[str, float]:
    predictions = (probabilities >= float(threshold)).astype(np.int64)
    per_class = [_class_metrics(labels, predictions, class_id) for class_id in (0, 1)]
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "macro_precision": float(np.mean([row[0] for row in per_class])),
        "macro_recall": float(np.mean([row[1] for row in per_class])),
        "macro_f1": float(np.mean([row[2] for row in per_class])),
        "auroc": float(_binary_auc(labels, probabilities)),
        "threshold": float(threshold),
    }


def select_threshold_by_macro_f1(
    labels: np.ndarray,
    probabilities: np.ndarray,
    minimum: float = 0.05,
    maximum: float = 0.95,
    step: float = 0.01,
) -> Tuple[float, float]:
    best_threshold = 0.5
    best_macro_f1 = -1.0
    for threshold in np.arange(minimum, maximum + 1e-8, step):
        macro_f1 = binary_metrics(labels, probabilities, threshold)["macro_f1"]
        if macro_f1 > best_macro_f1:
            best_threshold = float(threshold)
            best_macro_f1 = float(macro_f1)
    return best_threshold, best_macro_f1


def evaluate_with_validation_threshold(
    model: nn.Module,
    validation_loader: Iterable,
    test_loader: Iterable,
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    validation_labels, validation_probabilities, validation_loss = (
        collect_binary_predictions(model, validation_loader, device=device)
    )
    threshold, _ = select_threshold_by_macro_f1(
        validation_labels, validation_probabilities
    )
    test_labels, test_probabilities, test_loss = collect_binary_predictions(
        model, test_loader, device=device
    )
    validation = binary_metrics(
        validation_labels, validation_probabilities, threshold
    )
    test = binary_metrics(test_labels, test_probabilities, threshold)
    validation["loss"] = float(validation_loss)
    test["loss"] = float(test_loss)
    return {"validation": validation, "test": test}
