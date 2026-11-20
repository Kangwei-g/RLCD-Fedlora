from contextlib import nullcontext
from numbers import Integral, Real
import threading
import os
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import recall_score
from utils.utils import get_accuracy
from lora.lora_finetune import build_lora_param_groups
from sklearn.metrics import classification_report, precision_score, recall_score, roc_auc_score, f1_score, balanced_accuracy_score
model_file = os.environ.get('JL_MODEL_PATH', 'FacebookAI/roberta-base')
_tok_lock = threading.Lock()
_tokenizer_batch_lock = threading.Lock()
_roberta_tokenizer = None
_gpt2_tokenizer = None
_opt_tokenizers = {}
_aslora_score_lock = threading.Lock()

def configure_dropout(model, policy):
    if policy not in ('model_default', 'disabled'):
        raise ValueError(f'Unsupported dropout policy: {policy}')
    modules = [module for module in model.modules() if isinstance(module, nn.Dropout)]
    if policy == 'disabled':
        for module in modules:
            module.p = 0.0
        for module in model.modules():
            for owner in (module, getattr(module, 'config', None)):
                for name in ('dropout', 'activation_dropout', 'attention_dropout', 'layerdrop'):
                    if isinstance(getattr(owner, name, None), Real):
                        setattr(owner, name, 0.0)
            if hasattr(module, 'dropout_prob'):
                module.dropout_prob = 0.0
            if isinstance(getattr(module, 'attention_dropout', None), Real):
                module.attention_dropout = 0.0
            config = getattr(module, 'config', None)
            if isinstance(getattr(config, 'attention_dropout', None), Real):
                config.attention_dropout = 0.0
    return len(modules)

def training_autocast(device, precision):
    if precision == 'fp32':
        return nullcontext()
    if precision != 'bf16':
        raise ValueError(f'Unsupported training precision: {precision}')
    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('BF16 training requires a supported CUDA device')
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported():
            raise ValueError('This CUDA device does not support BF16')
    return torch.autocast('cuda', dtype=torch.bfloat16)

def compute_class_weights(labels, max_positive_weight=3.0):
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=2)
    neg_count, pos_count = (counts[0], counts[1])
    if neg_count == 0 or pos_count == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float32)
    positive_weight = np.sqrt(neg_count / pos_count)
    positive_weight = float(np.clip(positive_weight, 1.0, max_positive_weight))
    return torch.tensor([1.0, positive_weight], dtype=torch.float32)

def select_threshold_by_macro_f1(actuals, positive_probs, min_threshold=0.05, max_threshold=0.95, step=0.01):
    thresholds = np.arange(min_threshold, max_threshold + 1e-08, step)
    best_threshold = 0.5
    best_macro_f1 = -1.0
    for threshold in thresholds:
        preds = (positive_probs >= threshold).astype(int)
        macro_f1 = f1_score(actuals, preds, average='macro', zero_division=0)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_threshold = float(threshold)
    return (best_threshold, best_macro_f1)

def _dataset_has_tokenized_inputs(dataset):
    try:
        column_names = set(dataset.column_names)
    except AttributeError:
        try:
            column_names = set(dataset.keys())
        except AttributeError:
            return False
    return {'input_ids', 'attention_mask', 'label'}.issubset(column_names)

def _prepare_split_for_model(dataset, tokenizer, max_length, progress_prefix, split_name):
    if _dataset_has_tokenized_inputs(dataset):
        print(f"{progress_prefix}使用预编码 {split_name} 集 (n={len(dataset['label'])}, max_length={(len(dataset['input_ids'][0]) if len(dataset['input_ids']) else 0)}) ...", flush=True)
        encoded = {'input_ids': dataset['input_ids'], 'attention_mask': dataset['attention_mask']}
        labels = dataset['label']
        return (encoded, labels)
    print(f"{progress_prefix}分词编码 {split_name} 集 (n={len(dataset['text'])}, max_length={max_length}) ...", flush=True)
    with _tokenizer_batch_lock:
        encoded = tokenizer(dataset['text'], padding='max_length', truncation=True, max_length=max_length)
    print(f'{progress_prefix}分词编码 {split_name} 集 完成。', flush=True)
    return (encoded, dataset['label'])

def get_tokenizer_for_model(model_name):
    from jl_opt_model import get_opt_tokenizer
    with _tok_lock:
        if model_file not in _opt_tokenizers:
            _opt_tokenizers[model_file] = get_opt_tokenizer(model_file)
        return _opt_tokenizers[model_file]

def prepare_tensor_dataloader(dataset, model_name, max_length, progress_prefix, split_name, batch_size, shuffle, generator=None):
    tokenizer = None if _dataset_has_tokenized_inputs(dataset) else get_tokenizer_for_model(model_name)
    encoded, labels = _prepare_split_for_model(dataset, tokenizer, max_length=max_length, progress_prefix=progress_prefix, split_name=split_name)
    tensor_dataset = convert_to_tensor_dataset(encoded, labels)
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)

def _backward_microbatches(net, input_ids, attention_mask, labels, criterion, device, precision, micro_batch_size):
    """Accumulate one logical weighted-CE gradient, retaining only detached logits."""
    if isinstance(micro_batch_size, bool) or not isinstance(micro_batch_size, Integral) or micro_batch_size < 1:
        raise ValueError('_jl_micro_batch_size must be a positive integer')
    valid = labels != criterion.ignore_index
    denominator = valid.sum() if criterion.weight is None else criterion.weight[labels[valid]].sum()
    if not bool(torch.isfinite(denominator)) or denominator.item() <= 0:
        raise RuntimeError('Logical batch must have positive finite class weight')
    total_loss = torch.zeros((), device=device)
    detached_logits = []
    for start in range(0, len(labels), micro_batch_size):
        end = start + micro_batch_size
        with training_autocast(device, precision):
            outputs = net(input_ids=input_ids[start:end], attention_mask=attention_mask[start:end])
            logits = outputs.logits
            loss = F.cross_entropy(logits, labels[start:end], weight=criterion.weight, reduction='sum', ignore_index=criterion.ignore_index, label_smoothing=criterion.label_smoothing) / denominator
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite training loss')
        loss.backward()
        total_loss += loss.detach()
        detached_logits.append(logits.detach().float().cpu())
        del outputs, logits, loss
    return (total_loss.detach(), torch.cat(detached_logits))

def fit(net, trainset, epochs, lr, args_optimizer, device, finetune, model, progress_prefix='', disable_tqdm=False, tqdm_position=0, max_length=128, batch_size=32, lr_head=None, lr_a_scale=1.0, lora_weight_decay=None, arc_aslora_config=None, precision='fp32', shuffle_seed=None):
    net = net.to(device)
    net.train()
    print(f'{progress_prefix}Training precision={precision}; parameters/optimizer remain FP32', flush=True)
    trainable_params = [p for p in net.parameters() if p.requires_grad]
    lora_param_groups = None
    if finetune != 'full' and (lr_a_scale != 1.0 or lora_weight_decay is not None):
        default_wd = 0.01 if args_optimizer == 'adam' else 1e-05
        lora_wd = 0.0 if lora_weight_decay is None else float(lora_weight_decay)
        groups, lr_a_eff, a_only, has_lora = build_lora_param_groups(net.named_parameters(), lr, lr_a_scale=lr_a_scale, lora_weight_decay=lora_wd, default_weight_decay=default_wd)
        if has_lora:
            lora_param_groups = groups
            print(f'{progress_prefix}LoRA optimizer groups: lr={lr}, lr_a={lr_a_eff} (a_only={a_only}, lr_a_scale={lr_a_scale}), lora_weight_decay={lora_wd}', flush=True)
    if finetune == 'full':
        backbone_params = []
        head_params = []
        for name, param in net.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith(('classifier.', 'score.')):
                head_params.append(param)
            else:
                backbone_params.append(param)
        backbone_lr = float(lr)
        head_lr = float(lr_head) if lr_head is not None else backbone_lr * 5.0
        param_groups = [{'params': backbone_params, 'lr': backbone_lr, 'weight_decay': 0.01}, {'params': head_params, 'lr': head_lr, 'weight_decay': 0.0}]
        print(f'{progress_prefix}Full FT optimizer: backbone_lr={backbone_lr}, head_lr={head_lr}, backbone_params={sum((p.numel() for p in backbone_params))}, head_params={sum((p.numel() for p in head_params))}', flush=True)
    else:
        param_groups = trainable_params
    if args_optimizer == 'adam':
        if finetune == 'full':
            optimizer = optim.Adam(param_groups)
        elif lora_param_groups is not None:
            optimizer = optim.Adam(lora_param_groups, lr=lr)
        else:
            optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=0.01)
    elif args_optimizer == 'amsgrad':
        if finetune == 'full':
            optimizer = optim.Adam(param_groups, amsgrad=True)
        elif lora_param_groups is not None:
            optimizer = optim.Adam(lora_param_groups, lr=lr, amsgrad=True)
        else:
            optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=1e-05, amsgrad=True)
    elif args_optimizer == 'sgd':
        if finetune == 'full':
            optimizer = optim.SGD(param_groups, momentum=0.9)
        elif lora_param_groups is not None:
            optimizer = optim.SGD(lora_param_groups, lr=lr, momentum=0.9)
        else:
            optimizer = optim.SGD(trainable_params, lr=lr, momentum=0.9, weight_decay=1e-05)
    max_positive_weight = 5.0 if finetune == 'full' else 3.0
    class_weights = compute_class_weights(trainset['label'], max_positive_weight=max_positive_weight)
    print(f'{progress_prefix}训练类别权重 class_weights={class_weights.tolist()}', flush=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    train_dataloader = prepare_tensor_dataloader(trainset, model_name=model, max_length=max_length, progress_prefix=progress_prefix, split_name='train', batch_size=batch_size, shuffle=True, generator=None if shuffle_seed is None else torch.Generator().manual_seed(shuffle_seed))
    n_batches = len(train_dataloader)
    train_loss = 0
    train_acc = 0
    last_score_batch = None
    for epoch in range(epochs):
        train_running_loss = 0
        train_running_acc = 0
        all_preds = []
        all_labels = []
        recall_class1 = 0.0
        tqdm_train_iterator = tqdm(enumerate(train_dataloader), desc=f'{progress_prefix}训练进度 epoch {epoch + 1}/{epochs}', ascii=True, leave=tqdm_position == 0, total=n_batches, colour='green', position=min(int(tqdm_position), 40), ncols=130, disable=disable_tqdm, mininterval=0.5)
        for batch_idx, batch in tqdm_train_iterator:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            last_score_batch = (input_ids, attention_mask, labels)
            optimizer.zero_grad()
            micro_batch_size = getattr(net, '_jl_micro_batch_size', None)
            if micro_batch_size is not None and micro_batch_size < len(labels):
                loss, logits = _backward_microbatches(net, input_ids, attention_mask, labels, criterion, device, precision, micro_batch_size)
            else:
                with training_autocast(device, precision):
                    if finetune is not None and finetune != 'full':
                        net.backbone_model.to(device)
                        outputs = net.backbone_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    else:
                        outputs = net(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    logits = outputs.logits
                    loss = criterion(logits, labels)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('Nonfinite training loss')
                loss.backward()
            if finetune == 'full':
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0, error_if_nonfinite=True)
            elif precision == 'bf16':
                torch.nn.utils.clip_grad_norm_(trainable_params, float('inf'), error_if_nonfinite=True)
            optimizer.step()
            preds = torch.argmax(logits, dim=1)
            train_running_loss += loss.item()
            train_running_acc += get_accuracy(logits.detach().float().cpu(), labels.cpu()).item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            train_acc = train_running_acc / (batch_idx + 1)
            train_loss = train_running_loss / (batch_idx + 1)
            if batch_idx % 50 == 0 or batch_idx == n_batches - 1:
                recall_class1 = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
            tqdm_train_iterator.set_postfix(avg_train_acc=f'{train_acc:0.4f}', avg_train_loss=f'{train_loss:0.4f}', recall_cls1=f'{recall_class1:0.4f}')
    arc_aslora_scores = None
    arc_aslora_diagnostics = None
    net.eval()
    print(f'{progress_prefix}** Training complete **')
    return (train_acc, train_loss)

@torch.no_grad()
def test(net, test_dl, args, verbose=True, device='gpu', finetune=None, logger=None, progress_prefix='', disable_tqdm=False, tqdm_position=0, return_metrics=False, threshold=None, threshold_name='macro-F1', max_length=128):
    net = net.to(device)
    net.eval()
    criterion = nn.CrossEntropyLoss()
    test_dataloader = prepare_tensor_dataloader(test_dl, model_name=getattr(args, 'model', 'roberta'), max_length=max_length, progress_prefix=progress_prefix, split_name='test', batch_size=getattr(args, 'jl_eval_batch_size', 32), shuffle=False)
    tqdm_test_iterator = tqdm(enumerate(test_dataloader), desc=f'{progress_prefix}测试进度', total=len(test_dataloader), ascii=True, leave=tqdm_position == 0, colour='green', position=min(int(tqdm_position), 40), ncols=130, disable=disable_tqdm, mininterval=0.5)
    test_running_loss = 0
    test_running_acc = 0
    test_seen = 0
    actuals = []
    predictions = []
    probabilities = []
    for batch_idx, batch in tqdm_test_iterator:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        if getattr(net, '_jl_micro_batch_size', None) is not None:
            outputs = net(input_ids=input_ids, attention_mask=attention_mask)
        elif finetune is not None and finetune != 'full':
            net.backbone_model.to(device)
            outputs = net.backbone_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        else:
            outputs = net(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        loss = criterion(logits, labels)
        test_running_loss += loss.item() * len(labels)
        test_seen += len(labels)
        actuals.extend(labels.cpu().numpy())
        probs = torch.softmax(logits, dim=1)
        probabilities.extend(probs.detach().cpu().numpy())
        test_running_acc += get_accuracy(logits.detach().cpu(), labels.cpu()).item() * len(labels)
        test_acc = test_running_acc / test_seen
        test_loss = test_running_loss / test_seen
        tqdm_test_iterator.set_postfix(avg_test_acc=f'{test_acc:0.4f}', avg_test_loss=f'{test_loss:0.4f}')
    actuals_arr = np.array(actuals)
    probabilities_arr = np.array(probabilities)
    positive = np.clip(probabilities_arr[:, 1].astype(np.float64), 1e-07, 1 - 1e-07)
    ce = -(actuals_arr * np.log(positive) + (1 - actuals_arr) * np.log1p(-positive))
    brier = (positive - actuals_arr) ** 2
    masks = [actuals_arr == label for label in (0, 1) if np.any(actuals_arr == label)]
    balanced_ce = float(np.mean([ce[mask].mean() for mask in masks]))
    balanced_brier = float(np.mean([brier[mask].mean() for mask in masks]))
    if threshold is None:
        best_threshold, threshold_macro_f1 = select_threshold_by_macro_f1(actuals_arr, probabilities_arr[:, 1])
        threshold_source = f'Best Threshold ({threshold_name})'
    else:
        best_threshold = float(threshold)
        threshold_macro_f1 = None
        threshold_source = f'Fixed Threshold ({threshold_name})'
    predictions_arr = (probabilities_arr[:, 1] >= best_threshold).astype(int)
    predictions = predictions_arr.tolist()
    test_acc = float(np.mean(predictions_arr == actuals_arr))
    macro_f1 = f1_score(actuals_arr, predictions_arr, average='macro', zero_division=0)
    micro_f1 = f1_score(actuals_arr, predictions_arr, average='micro', zero_division=0)
    balanced_acc = balanced_accuracy_score(actuals_arr, predictions_arr)
    recall_cls1 = recall_score(actuals, predictions, pos_label=1, zero_division=0)
    precision_cls1 = precision_score(actuals, predictions, pos_label=1, zero_division=0)
    f1_cls1 = f1_score(actuals, predictions, pos_label=1, zero_division=0)
    macro_precision = precision_score(actuals_arr, predictions_arr, average='macro', zero_division=0)
    macro_recall = recall_score(actuals_arr, predictions_arr, average='macro', zero_division=0)
    per_class_auc = []
    for class_id in range(args.out_dim):
        y_true_binary = (actuals_arr == class_id).astype(int)
        if np.any(y_true_binary == 1) and np.any(y_true_binary == 0):
            per_class_auc.append(roc_auc_score(y_true_binary, probabilities_arr[:, class_id]))
    macro_auc = float(np.mean(per_class_auc)) if per_class_auc else float('nan')
    print(f'{progress_prefix}{threshold_source}: {best_threshold:.2f}')
    print(f'{progress_prefix}Test Loss: {test_loss}')
    print(f'{progress_prefix}Test Accuracy: {test_acc}')
    if verbose:
        print(f'{progress_prefix}Metrics Report:')
        prefix_line = progress_prefix.strip() + ' ' if progress_prefix else ''
        logger.info(prefix_line + f'{threshold_source}: {best_threshold:.2f}')
        logger.info(prefix_line + '\n' + classification_report(actuals, predictions, target_names=['Class 0', 'Class 1'], zero_division=0))
        precision = precision_score(actuals, predictions, average='weighted', zero_division=0)
        recall = recall_score(actuals, predictions, average='weighted', zero_division=0)
        logger.info(prefix_line + '=== Class 1 (Minority) Metrics ===')
        logger.info(prefix_line + f'Precision Class 1: {precision_cls1:.4f}')
        logger.info(prefix_line + f'Recall Class 1: {recall_cls1:.4f}')
        logger.info(prefix_line + f'F1 Class 1: {f1_cls1:.4f}')
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
                    print(f'{progress_prefix}Class {i}: {e}')
                    roc_auc[i] = None
            else:
                print(f'{progress_prefix}Class {i}: Only one class present in y_true. Skipping AUC-ROC calculation.')
                roc_auc[i] = None
        if len(y_true_micro) > 0 and len(set(y_true_micro)) > 1:
            micro_auc = roc_auc_score(y_true_micro, y_scores_micro)
            valid_roc_auc = [roc_auc[i] for i in range(args.out_dim) if roc_auc[i] is not None]
            macro_avg_auc = np.mean(valid_roc_auc) if valid_roc_auc else 0
            logger.info(prefix_line + '=== Overall Metrics ===')
            logger.info(prefix_line + f'Precision (weighted): {precision:.4f}')
            logger.info(prefix_line + f'Recall (weighted): {recall:.4f}')
            logger.info(prefix_line + f'Balanced Accuracy: {balanced_acc:.4f}')
            logger.info(prefix_line + f'ROC AUC (macro): {macro_avg_auc:.4f}')
            logger.info(prefix_line + f'ROC AUC (micro): {micro_auc:.4f}')
            logger.info(prefix_line + f'F1 (macro): {macro_f1:.4f}')
            logger.info(prefix_line + f'F1 (micro): {micro_f1:.4f}')
    net.to('cpu')
    if return_metrics:
        return (test_acc, test_loss, {'threshold': best_threshold, 'balanced_ce': balanced_ce, 'balanced_brier': balanced_brier, 'fixed_macro_f1': float(f1_score(actuals_arr, (positive >= 0.5).astype(int), average='macro', zero_division=0)), 'macro_f1': macro_f1, 'micro_f1': micro_f1, 'balanced_acc': balanced_acc, 'precision_cls1': precision_cls1, 'recall_cls1': recall_cls1, 'f1_cls1': f1_cls1, 'macro_precision': macro_precision, 'macro_recall': macro_recall, 'macro_auc': macro_auc})
    return (test_acc, test_loss)

def convert_to_tensor_dataset(encoded_data, labels):
    input_ids = torch.tensor(encoded_data['input_ids'], dtype=torch.long)
    attention_mask = torch.tensor(encoded_data['attention_mask'], dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(input_ids, attention_mask, labels)
