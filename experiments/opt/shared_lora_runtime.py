import copy
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from lora.lora_finetune import (
    build_lora_param_groups,
    clone_lora_state_dict,
    get_lora_parameter_names,
)
from utils.trainer import (
    compute_class_weights,
    prepare_tensor_dataloader,
    select_threshold_by_macro_f1,
)
from utils.utils import get_accuracy

try:
    from torch.func import functional_call
except ImportError:  # pragma: no cover
    from torch.nn.utils.stateless import functional_call


def _model_signature(model):
    sig = []
    for name, param in model.named_parameters():
        if name.startswith("classifier.") or name.split(".")[-1] in {"lora_a", "lora_b"}:
            sig.append((name, tuple(param.shape)))
    return tuple(sig)


class SharedBackboneRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._entries = {}

    def clear(self):
        with self._lock:
            self._entries.clear()

    def get_models(self, device, source_model):
        signature = _model_signature(source_model)
        with self._lock:
            entry = self._entries.get(device)
            if entry is None or entry["signature"] != signature:
                train_model = copy.deepcopy(source_model).to(device)
                eval_model = copy.deepcopy(source_model).to(device)
                for model in (train_model, eval_model):
                    for param in model.parameters():
                        param.requires_grad = False
                train_model.train()
                eval_model.eval()
                entry = {
                    "signature": signature,
                    "train_model": train_model,
                    "eval_model": eval_model,
                    "lora_names": tuple(get_lora_parameter_names(source_model)),
                }
                self._entries[device] = entry
        return entry["train_model"], entry["eval_model"], entry["lora_names"]


SHARED_BACKBONE_REGISTRY = SharedBackboneRegistry()


def create_lora_client_states(global_model, n_parties):
    base_state = clone_lora_state_dict(global_model, device="cpu")
    return {
        net_i: {name: tensor.clone() for name, tensor in base_state.items()}
        for net_i in range(n_parties)
    }


def sync_lora_client_state_from_global(client_state, global_model):
    global_lora = clone_lora_state_dict(global_model, device="cpu")
    client_state.clear()
    for name, tensor in global_lora.items():
        client_state[name] = tensor.clone()


def _merged_params_and_buffers(model, client_params):
    merged = {}
    for name, param in model.named_parameters():
        merged[name] = client_params.get(name, param)
    for name, buffer in model.named_buffers():
        merged[name] = buffer
    return merged


def _materialize_client_params(client_state, device, frozen_layer):
    frozen = set(frozen_layer)
    params = {}
    trainable = []
    for name, tensor in client_state.items():
        device_tensor = tensor.detach().to(device=device).clone()
        if name in frozen:
            device_tensor.requires_grad_(False)
        else:
            device_tensor.requires_grad_(True)
            trainable.append(device_tensor)
        params[name] = device_tensor
    return params, trainable


def _make_optimizer(args_optimizer, trainable_params, lr):
    if args_optimizer == "adam":
        return optim.Adam(trainable_params, lr=lr, weight_decay=0.01)
    if args_optimizer == "amsgrad":
        return optim.Adam(trainable_params, lr=lr, weight_decay=1e-5, amsgrad=True)
    if args_optimizer == "sgd":
        return optim.SGD(trainable_params, lr=lr, momentum=0.9, weight_decay=1e-5)
    raise ValueError(f"Unsupported optimizer: {args_optimizer}")


def _forward_logits(model, client_params, input_ids, attention_mask):
    merged = _merged_params_and_buffers(model, client_params)
    outputs = functional_call(
        model,
        merged,
        (),
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
    )
    return outputs.logits


def _evaluate_shared_model(
        model,
        client_params,
        dataset,
        args,
        verbose,
        device,
        logger,
        progress_prefix,
        disable_tqdm,
        tqdm_position,
        return_metrics,
        threshold,
        threshold_name,
        max_length,
):
    criterion = nn.CrossEntropyLoss()
    test_dataloader = prepare_tensor_dataloader(
        dataset,
        model_name=getattr(args, "model", "roberta"),
        max_length=max_length,
        progress_prefix=progress_prefix,
        split_name="test",
        batch_size=32,
        shuffle=True,
    )

    tqdm_test_iterator = tqdm(
        enumerate(test_dataloader),
        desc=f"{progress_prefix}测试进度",
        total=len(test_dataloader),
        ascii=True,
        leave=(tqdm_position == 0),
        colour="green",
        position=min(int(tqdm_position), 40),
        ncols=130,
        disable=disable_tqdm,
        mininterval=0.5,
    )

    test_running_loss = 0
    test_running_acc = 0
    actuals = []
    predictions = []
    probabilities = []

    for batch_idx, batch in tqdm_test_iterator:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        with torch.no_grad():
            logits = _forward_logits(model, client_params, input_ids, attention_mask)
            loss = criterion(logits, labels)

        test_running_loss += loss.item()
        actuals.extend(labels.cpu().numpy())
        probs = torch.softmax(logits, dim=1)
        probabilities.extend(probs.detach().cpu().numpy())
        test_running_acc += get_accuracy(logits.detach().cpu(), labels.cpu()).item()

        test_acc = test_running_acc / (batch_idx + 1)
        test_loss = test_running_loss / (batch_idx + 1)
        tqdm_test_iterator.set_postfix(avg_test_acc=f"{test_acc:0.4f}", avg_test_loss=f"{test_loss:0.4f}")

    actuals_arr = np.array(actuals)
    probabilities_arr = np.array(probabilities)
    if threshold is None:
        best_threshold, threshold_macro_f1 = select_threshold_by_macro_f1(actuals_arr, probabilities_arr[:, 1])
        threshold_source = f"Best Threshold ({threshold_name})"
    else:
        best_threshold = float(threshold)
        threshold_macro_f1 = None
        threshold_source = f"Fixed Threshold ({threshold_name})"

    predictions_arr = (probabilities_arr[:, 1] >= best_threshold).astype(int)
    predictions = predictions_arr.tolist()
    test_acc = float(np.mean(predictions_arr == actuals_arr))
    macro_f1 = f1_score(actuals_arr, predictions_arr, average='macro', zero_division=0)
    micro_f1 = f1_score(actuals_arr, predictions_arr, average='micro', zero_division=0)
    balanced_acc = balanced_accuracy_score(actuals_arr, predictions_arr)
    recall_cls1 = recall_score(actuals, predictions, pos_label=1, zero_division=0)
    precision_cls1 = precision_score(actuals, predictions, pos_label=1, zero_division=0)
    f1_cls1 = f1_score(actuals, predictions, pos_label=1, zero_division=0)

    print(f"{progress_prefix}{threshold_source}: {best_threshold:.2f}")
    print(f"{progress_prefix}Test Loss: {test_loss}")
    print(f"{progress_prefix}Test Accuracy: {test_acc}")

    if verbose and logger is not None:
        print(f"{progress_prefix}Metrics Report:")
        prefix_line = progress_prefix.strip() + " " if progress_prefix else ""
        logger.info(prefix_line + f"{threshold_source}: {best_threshold:.2f}")
        logger.info(prefix_line + "\n" + classification_report(
            actuals, predictions, target_names=['Class 0', 'Class 1'], zero_division=0
        ))
        precision = precision_score(actuals, predictions, average='weighted', zero_division=0)
        recall = recall_score(actuals, predictions, average='weighted', zero_division=0)
        logger.info(prefix_line + "=== Class 1 (Minority) Metrics ===")
        logger.info(prefix_line + f"Precision Class 1: {precision_cls1:.4f}")
        logger.info(prefix_line + f"Recall Class 1: {recall_cls1:.4f}")
        logger.info(prefix_line + f"F1 Class 1: {f1_cls1:.4f}")

        roc_auc = {}
        y_true_micro = []
        y_scores_micro = []
        for i in range(args.out_dim):
            y_true_binary = (actuals_arr == i).astype(int)
            y_scores_binary = probabilities_arr[:, i]
            if np.any(y_true_binary == 1) and np.any(y_true_binary == 0):
                try:
                    y_true_micro.extend(y_true_binary)
                    y_scores_micro.extend(y_scores_binary)
                    roc_auc[i] = roc_auc_score(y_true_binary, y_scores_binary)
                except ValueError:
                    roc_auc[i] = None
            else:
                roc_auc[i] = None

        if len(y_true_micro) > 0 and len(set(y_true_micro)) > 1:
            micro_auc = roc_auc_score(y_true_micro, y_scores_micro)
            valid_roc_auc = [roc_auc[i] for i in range(args.out_dim) if roc_auc[i] is not None]
            macro_avg_auc = np.mean(valid_roc_auc) if valid_roc_auc else 0
            logger.info(prefix_line + "=== Overall Metrics ===")
            logger.info(prefix_line + f"Precision (weighted): {precision:.4f}")
            logger.info(prefix_line + f"Recall (weighted): {recall:.4f}")
            logger.info(prefix_line + f"Balanced Accuracy: {balanced_acc:.4f}")
            logger.info(prefix_line + f"ROC AUC (macro): {macro_avg_auc:.4f}")
            logger.info(prefix_line + f"ROC AUC (micro): {micro_auc:.4f}")
            logger.info(prefix_line + f"F1 (macro): {macro_f1:.4f}")
            logger.info(prefix_line + f"F1 (micro): {micro_f1:.4f}")

    if return_metrics:
        return test_acc, test_loss, {
            "threshold": best_threshold,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "balanced_acc": balanced_acc,
            "precision_cls1": precision_cls1,
            "recall_cls1": recall_cls1,
            "f1_cls1": f1_cls1,
            "threshold_macro_f1": threshold_macro_f1,
        }
    return test_acc, test_loss


def train_shared_lora_client(
        client_state,
        source_model,
        args,
        trainset,
        valiset,
        testset,
        frozen_layer,
        not_com_layer,
        device_id,
        logger,
        train_prefix,
        val_prefix,
        test_prefix,
        no_client_tqdm,
        tqdm_pos,
):
    train_model, eval_model, lora_names = SHARED_BACKBONE_REGISTRY.get_models(device_id, source_model)
    client_params, trainable_params = _materialize_client_params(client_state, device_id, frozen_layer)
    if not trainable_params:
        raise RuntimeError("Shared LoRA runtime found no trainable client parameters.")

    lr_a_scale = float(getattr(args, "lr_a_scale", 1.0))
    lora_wd_arg = getattr(args, "lora_weight_decay", None)
    if lr_a_scale != 1.0 or lora_wd_arg is not None:
        lora_wd = 0.0 if lora_wd_arg is None else float(lora_wd_arg)
        groups, lr_a_eff, a_only, has_lora = build_lora_param_groups(
            client_params.items(), args.lr,
            lr_a_scale=lr_a_scale,
            lora_weight_decay=lora_wd,
            default_weight_decay=0.01,
        )
        optimizer = _make_optimizer(args.optimizer, groups, args.lr)
        print(
            f"{train_prefix}LoRA optimizer groups: lr={args.lr}, lr_a={lr_a_eff} "
            f"(a_only={a_only}, lr_a_scale={lr_a_scale}), lora_weight_decay={lora_wd}",
            flush=True,
        )
    else:
        optimizer = _make_optimizer(args.optimizer, trainable_params, args.lr)
    class_weights = compute_class_weights(trainset['label'], max_positive_weight=3.0)
    print(f"{train_prefix}训练类别权重 class_weights={class_weights.tolist()}", flush=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device_id))

    train_dataloader = prepare_tensor_dataloader(
        trainset,
        model_name=args.model,
        max_length=args.max_length,
        progress_prefix=train_prefix,
        split_name="train",
        batch_size=args.batch_size,
        shuffle=True,
    )
    n_batches = len(train_dataloader)
    train_loss = 0.0
    train_acc = 0.0

    for epoch in range(args.epochs):
        train_running_loss = 0.0
        train_running_acc = 0.0
        all_preds = []
        all_labels = []
        recall_class1 = 0.0
        tqdm_train_iterator = tqdm(
            enumerate(train_dataloader),
            desc=f"{train_prefix}训练进度 epoch {epoch + 1}/{args.epochs}",
            ascii=True,
            leave=(tqdm_pos == 0),
            total=n_batches,
            colour="green",
            position=min(int(tqdm_pos), 40),
            ncols=130,
            disable=no_client_tqdm,
            mininterval=0.5,
        )

        for batch_idx, batch in tqdm_train_iterator:
            input_ids = batch[0].to(device_id)
            attention_mask = batch[1].to(device_id)
            labels = batch[2].to(device_id)

            optimizer.zero_grad()
            logits = _forward_logits(train_model, client_params, input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            train_running_loss += loss.item()
            train_running_acc += get_accuracy(logits.detach().cpu(), labels.cpu()).item()
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            train_acc = train_running_acc / (batch_idx + 1)
            train_loss = train_running_loss / (batch_idx + 1)
            if batch_idx % 50 == 0 or batch_idx == n_batches - 1:
                recall_class1 = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
            tqdm_train_iterator.set_postfix(
                avg_train_acc=f"{train_acc:0.4f}",
                avg_train_loss=f"{train_loss:0.4f}",
                recall_cls1=f"{recall_class1:0.4f}",
            )

    print(f"{train_prefix}** Training complete **")

    _, _, val_metrics = _evaluate_shared_model(
        eval_model,
        client_params,
        valiset,
        args,
        verbose=False,
        device=device_id,
        logger=logger,
        progress_prefix=val_prefix,
        disable_tqdm=no_client_tqdm,
        tqdm_position=tqdm_pos,
        return_metrics=True,
        threshold=None,
        threshold_name="macro-F1",
        max_length=args.max_length,
    )
    test_acc, test_loss, test_metrics = _evaluate_shared_model(
        eval_model,
        client_params,
        testset,
        args,
        verbose=True,
        device=device_id,
        logger=logger,
        progress_prefix=test_prefix,
        disable_tqdm=no_client_tqdm,
        tqdm_position=tqdm_pos,
        return_metrics=True,
        threshold=val_metrics["threshold"],
        threshold_name="client-val macro-F1",
        max_length=args.max_length,
    )

    layers_params = {}
    skip_upload = set(not_com_layer)
    for name in lora_names:
        tensor = client_params[name]
        client_state[name] = tensor.detach().cpu().clone()
        if name in skip_upload or name in frozen_layer:
            continue
        snapshot = tensor.detach().cpu().clone()
        if tensor.grad is not None:
            snapshot.grad = tensor.grad.detach().cpu().clone()
        layers_params[name] = snapshot

    if torch.cuda.is_available():
        with torch.cuda.device(device_id):
            torch.cuda.empty_cache()

    return train_acc, train_loss, test_acc, test_loss, val_metrics, test_metrics, layers_params
