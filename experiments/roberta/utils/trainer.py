# Description: This file contains the training and testing functions for the model.
from itertools import chain
import threading
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from transformers import RobertaTokenizer, GPT2Tokenizer
from sklearn.metrics import recall_score  # 新增导入

from utils.utils import get_accuracy
from lora.lora_finetune import build_lora_param_groups
from arc_aslora import compute_arc_aslora_fd_scores
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, roc_curve, auc, \
    roc_auc_score, f1_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

model_file = 'FacebookAI/roberta-base'

_tok_lock = threading.Lock()
# 同一 Tokenizer 实例上并发 __call__/batch_encode 在 Rust 层不安全，多线程会卡死；与 from_pretrained 的锁分开。
_tokenizer_batch_lock = threading.Lock()
_roberta_tokenizer = None
_gpt2_tokenizer = None


def get_roberta_tokenizer():
    """多线程下只加载一次，避免 10 路同时 from_pretrained 抢锁/卡死。"""
    global _roberta_tokenizer
    if _roberta_tokenizer is not None:
        return _roberta_tokenizer
    with _tok_lock:
        if _roberta_tokenizer is None:
            _roberta_tokenizer = RobertaTokenizer.from_pretrained(model_file)
    return _roberta_tokenizer


def get_gpt2_tokenizer():
    global _gpt2_tokenizer
    if _gpt2_tokenizer is not None:
        return _gpt2_tokenizer
    with _tok_lock:
        if _gpt2_tokenizer is None:
            _gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_file)
    return _gpt2_tokenizer


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', num_classes=2):
        super(FocalLoss, self).__init__()
        # alpha 可以是标量或张量，如果是标量则自动扩展
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha, (list, float, int)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 将 alpha 移到相同设备
        if self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)

        # 使用 alpha 作为类别权重
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)  # Prevents nans when probability 0
        F_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


def compute_class_weights(labels, max_positive_weight=3.0):
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=2)
    neg_count, pos_count = counts[0], counts[1]
    if neg_count == 0 or pos_count == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float32)

    positive_weight = np.sqrt(neg_count / pos_count)
    positive_weight = float(np.clip(positive_weight, 1.0, max_positive_weight))
    return torch.tensor([1.0, positive_weight], dtype=torch.float32)


def select_threshold_by_macro_f1(actuals, positive_probs, min_threshold=0.05, max_threshold=0.95, step=0.01):
    thresholds = np.arange(min_threshold, max_threshold + 1e-8, step)
    best_threshold = 0.5
    best_macro_f1 = -1.0

    for threshold in thresholds:
        preds = (positive_probs >= threshold).astype(int)
        macro_f1 = f1_score(actuals, preds, average='macro', zero_division=0)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_threshold = float(threshold)

    return best_threshold, best_macro_f1


def _dataset_has_tokenized_inputs(dataset):
    try:
        column_names = set(dataset.column_names)
    except AttributeError:
        try:
            column_names = set(dataset.keys())
        except AttributeError:
            return False
    return {"input_ids", "attention_mask", "label"}.issubset(column_names)


def _prepare_split_for_model(dataset, tokenizer, max_length, progress_prefix, split_name):
    if _dataset_has_tokenized_inputs(dataset):
        print(
            f"{progress_prefix}使用预编码 {split_name} 集 "
            f"(n={len(dataset['label'])}, max_length={len(dataset['input_ids'][0]) if len(dataset['input_ids']) else 0}) ...",
            flush=True,
        )
        encoded = {
            "input_ids": dataset["input_ids"],
            "attention_mask": dataset["attention_mask"],
        }
        labels = dataset["label"]
        return encoded, labels

    print(f"{progress_prefix}分词编码 {split_name} 集 (n={len(dataset['text'])}, max_length={max_length}) ...", flush=True)
    with _tokenizer_batch_lock:
        encoded = tokenizer(
            dataset["text"],
            padding='max_length',
            truncation=True,
            max_length=max_length,
        )
    print(f"{progress_prefix}分词编码 {split_name} 集 完成。", flush=True)
    return encoded, dataset["label"]


def get_tokenizer_for_model(model_name):
    if model_name == 'gpt':
        return get_gpt2_tokenizer()
    return get_roberta_tokenizer()


def prepare_tensor_dataloader(
        dataset,
        model_name,
        max_length,
        progress_prefix,
        split_name,
        batch_size,
        shuffle,
):
    tokenizer = get_tokenizer_for_model(model_name)
    encoded, labels = _prepare_split_for_model(
        dataset, tokenizer, max_length=max_length, progress_prefix=progress_prefix, split_name=split_name
    )
    tensor_dataset = convert_to_tensor_dataset(encoded, labels)
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)


def fit(
        net,
        trainset,
        epochs,
        lr,
        args_optimizer,
        device,
        finetune,
        model,
        progress_prefix="",
        disable_tqdm=False,
        tqdm_position=0,
        max_length=128,
        batch_size=32,
        lr_head=None,
        lr_a_scale=1.0,
        lora_weight_decay=None,
        arc_aslora_config=None,
):
    net = net.to(device)
    trainable_params = [p for p in net.parameters() if p.requires_grad]

    # LoRA-aware param groups (separate lr for lora_a on pure-A coordinate
    # rounds, configurable weight decay on LoRA factors). Legacy callers that
    # don't pass these kwargs keep the old single-group behaviour.
    lora_param_groups = None
    if finetune != "full" and (lr_a_scale != 1.0 or lora_weight_decay is not None):
        default_wd = 0.01 if args_optimizer == "adam" else 1e-5
        lora_wd = 0.0 if lora_weight_decay is None else float(lora_weight_decay)
        groups, lr_a_eff, a_only, has_lora = build_lora_param_groups(
            net.named_parameters(), lr,
            lr_a_scale=lr_a_scale,
            lora_weight_decay=lora_wd,
            default_weight_decay=default_wd,
        )
        if has_lora:
            lora_param_groups = groups
            print(
                f"{progress_prefix}LoRA optimizer groups: lr={lr}, lr_a={lr_a_eff} "
                f"(a_only={a_only}, lr_a_scale={lr_a_scale}), lora_weight_decay={lora_wd}",
                flush=True,
            )

    if finetune == "full":
        backbone_params = []
        head_params = []
        for name, param in net.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("classifier."):
                head_params.append(param)
            else:
                backbone_params.append(param)
        backbone_lr = float(lr)
        head_lr = float(lr_head) if lr_head is not None else backbone_lr * 5.0
        param_groups = [
            {"params": backbone_params, "lr": backbone_lr, "weight_decay": 0.01},
            {"params": head_params, "lr": head_lr, "weight_decay": 0.0},
        ]
        print(
            f"{progress_prefix}Full FT optimizer: backbone_lr={backbone_lr}, head_lr={head_lr}, "
            f"backbone_params={sum(p.numel() for p in backbone_params)}, "
            f"head_params={sum(p.numel() for p in head_params)}",
            flush=True,
        )
    else:
        param_groups = trainable_params

    if args_optimizer == "adam":
        if finetune == "full":
            optimizer = optim.Adam(param_groups)
        elif lora_param_groups is not None:
            optimizer = optim.Adam(lora_param_groups, lr=lr)
        else:
            optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=0.01)
    elif args_optimizer == "amsgrad":
        if finetune == "full":
            optimizer = optim.Adam(param_groups, amsgrad=True)
        elif lora_param_groups is not None:
            optimizer = optim.Adam(lora_param_groups, lr=lr, amsgrad=True)
        else:
            optimizer = optim.Adam(
                trainable_params, lr=lr, weight_decay=1e-5, amsgrad=True
            )
    elif args_optimizer == "sgd":
        if finetune == "full":
            optimizer = optim.SGD(param_groups, momentum=0.9)
        elif lora_param_groups is not None:
            optimizer = optim.SGD(lora_param_groups, lr=lr, momentum=0.9)
        else:
            optimizer = optim.SGD(
                trainable_params, lr=lr, momentum=0.9, weight_decay=1e-5
            )

    # Full-model FT needs stronger minority weighting; LoRA keeps the sqrt cap at 3.
    max_positive_weight = 5.0 if finetune == "full" else 3.0
    class_weights = compute_class_weights(trainset['label'], max_positive_weight=max_positive_weight)
    print(f"{progress_prefix}训练类别权重 class_weights={class_weights.tolist()}", flush=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # 或者使用简单的加权交叉熵（如果 Focal Loss 效果不佳）
    # criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    train_dataloader = prepare_tensor_dataloader(
        trainset,
        model_name=model,
        max_length=max_length,
        progress_prefix=progress_prefix,
        split_name="train",
        batch_size=batch_size,
        shuffle=True,
    )
    n_batches = len(train_dataloader)

    train_loss = 0
    train_acc = 0
    last_score_batch = None

    for epoch in range(epochs):
        train_running_loss = 0
        train_running_acc = 0
        all_preds = []  # 用于计算 recall
        all_labels = []  # 用于计算 recall
        recall_class1 = 0.0

        tqdm_train_iterator = tqdm(
            enumerate(train_dataloader),
            desc=f"{progress_prefix}训练进度 epoch {epoch + 1}/{epochs}",
            ascii=True,
            leave=(tqdm_position == 0),
            total=n_batches,
            colour="green",
            position=min(int(tqdm_position), 40),
            ncols=130,
            disable=disable_tqdm,
            mininterval=0.5,
        )

        for batch_idx, batch in tqdm_train_iterator:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            last_score_batch = (input_ids, attention_mask, labels)

            optimizer.zero_grad()

            if finetune is not None and finetune != "full":
                net.backbone_model.to(device)
                outputs = net.backbone_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            else:
                outputs = net(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            logits = outputs.logits
            loss = criterion(logits, labels)

            loss.backward()
            if finetune == "full":
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            # 计算准确率
            preds = torch.argmax(logits, dim=1)
            train_running_loss += loss.item()
            train_running_acc += get_accuracy(logits.detach().cpu(), labels.cpu()).item()

            # 收集预测和标签用于计算 recall
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            train_acc = train_running_acc / (batch_idx + 1)
            train_loss = train_running_loss / (batch_idx + 1)

            # 每 50 step 或最后一步才算 recall，避免多线程下每 batch 全量 sklearn 拖死 CPU
            if batch_idx % 50 == 0 or batch_idx == n_batches - 1:
                recall_class1 = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)

            tqdm_train_iterator.set_postfix(
                avg_train_acc=f"{train_acc:0.4f}",
                avg_train_loss=f"{train_loss:0.4f}",
                recall_cls1=f"{recall_class1:0.4f}"  # 监控少数类召回
            )

    arc_aslora_scores = None
    arc_aslora_diagnostics = None
    if arc_aslora_config is not None:
        if last_score_batch is None:
            raise RuntimeError("ARC+AS-LoRA scoring requires a non-empty local train loader")
        arc_aslora_scores, arc_aslora_diagnostics = compute_arc_aslora_fd_scores(
            model=net,
            batch=last_score_batch,
            criterion=criterion,
            active_layers=arc_aslora_config["active_layers"],
            eta=float(arc_aslora_config["eta"]),
            fd_eps=float(arc_aslora_config["fd_eps"]),
            norm_eps=float(arc_aslora_config["norm_eps"]),
            projection_seed=int(arc_aslora_config["projection_seed"]),
        )
        print(
            f"{progress_prefix}AS-LoRA FD scores complete: "
            f"layers={sorted(arc_aslora_scores)}, "
            f"base_loss={arc_aslora_diagnostics['base_loss']:.6f}, "
            f"restore_error={arc_aslora_diagnostics['max_restore_error']:.1e}",
            flush=True,
        )

    net.eval()
    print(f"{progress_prefix}** Training complete **")
    if arc_aslora_config is not None:
        return train_acc, train_loss, arc_aslora_scores, arc_aslora_diagnostics
    return train_acc, train_loss


def test(
        net,
        test_dl,
        args,
        verbose=True,
        device="gpu",
        finetune=None,
        logger=None,
        progress_prefix="",
        disable_tqdm=False,
        tqdm_position=0,
        return_metrics=False,
        threshold=None,
        threshold_name="macro-F1",
        max_length=128,
):
    net = net.to(device)
    net.eval()

    # 服务器/客户端验证用于模型选择和 Double DQN 状态，使用无偏的标准交叉熵。
    criterion = nn.CrossEntropyLoss()

    test_dataloader = prepare_tensor_dataloader(
        test_dl,
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

        if finetune is not None and finetune != "full":
            net.backbone_model.to(device)
            outputs = net.backbone_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        else:
            outputs = net(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        logits = outputs.logits
        loss = criterion(logits, labels)

        test_running_loss += loss.item()

        actuals.extend(labels.cpu().numpy())
        probs = torch.softmax(logits, dim=1)
        probabilities.extend(probs.detach().cpu().numpy())
        test_running_acc += get_accuracy(logits.detach().cpu(), labels.cpu()).item()

        test_acc = test_running_acc / (batch_idx + 1)
        test_loss = test_running_loss / (batch_idx + 1)

        tqdm_test_iterator.set_postfix(avg_test_acc=f"{test_acc:0.4f}",
                                       avg_test_loss=f"{test_loss:0.4f}")

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
    macro_precision = precision_score(
        actuals_arr, predictions_arr, average="macro", zero_division=0
    )
    macro_recall = recall_score(
        actuals_arr, predictions_arr, average="macro", zero_division=0
    )
    per_class_auc = []
    for class_id in range(args.out_dim):
        y_true_binary = (actuals_arr == class_id).astype(int)
        if np.any(y_true_binary == 1) and np.any(y_true_binary == 0):
            per_class_auc.append(
                roc_auc_score(y_true_binary, probabilities_arr[:, class_id])
            )
    macro_auc = float(np.mean(per_class_auc)) if per_class_auc else float("nan")

    print(f"{progress_prefix}{threshold_source}: {best_threshold:.2f}")
    print(f"{progress_prefix}Test Loss: {test_loss}")
    print(f"{progress_prefix}Test Accuracy: {test_acc}")

    if verbose:
        print(f"{progress_prefix}Metrics Report:")
        prefix_line = progress_prefix.strip() + " " if progress_prefix else ""
        logger.info(prefix_line + f"{threshold_source}: {best_threshold:.2f}")
        logger.info(prefix_line + "\n" + classification_report(
            actuals, predictions, target_names=['Class 0', 'Class 1'], zero_division=0
        ))

        # 详细指标
        precision = precision_score(actuals, predictions, average='weighted', zero_division=0)
        recall = recall_score(actuals, predictions, average='weighted', zero_division=0)

        logger.info(prefix_line + "=== Class 1 (Minority) Metrics ===")
        logger.info(prefix_line + f"Precision Class 1: {precision_cls1:.4f}")
        logger.info(prefix_line + f"Recall Class 1: {recall_cls1:.4f}")
        logger.info(prefix_line + f"F1 Class 1: {f1_cls1:.4f}")

        # ROC AUC 计算
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
                except ValueError as e:
                    print(f"{progress_prefix}Class {i}: {e}")
                    roc_auc[i] = None
            else:
                print(f"{progress_prefix}Class {i}: Only one class present in y_true. Skipping AUC-ROC calculation.")
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

    net.to('cpu')
    if return_metrics:
        return test_acc, test_loss, {
            "threshold": best_threshold,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "balanced_acc": balanced_acc,
            "precision_cls1": precision_cls1,
            "recall_cls1": recall_cls1,
            "f1_cls1": f1_cls1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_auc": macro_auc,
        }
    return test_acc, test_loss


def convert_to_tensor_dataset(encoded_data, labels):
    input_ids = torch.tensor(encoded_data['input_ids'], dtype=torch.long)
    attention_mask = torch.tensor(encoded_data['attention_mask'], dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(input_ids, attention_mask, labels)
