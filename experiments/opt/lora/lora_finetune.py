
import torch 
from torch import nn
import os

from transformers import RobertaForSequenceClassification

from lora.lora import inject_lora


def add_lora(model,filter_names,r,model_name):
    if model_name=='roberta':
        filter_names = ['roberta.encoder.layer.'+str(i)+'.attention.self.' +postfix for i in filter_names for postfix in ['key','value']]
    elif model_name=='gpt':
        filter_names = ['transformer.h.'+str(i)+'.attn.c_attn.'+postfix for i in filter_names for postfix in ['k_proj','v_proj']]
    elif model_name == 'opt':
        filter_names = {f'model.decoder.layers.{i}.self_attn.{projection}'
                        for i in filter_names for projection in ('k_proj', 'v_proj')}
        modules = dict(model.named_modules())
        invalid = [name for name in filter_names if not isinstance(modules.get(name), nn.Linear)]
        if invalid:
            raise ValueError(f'OPT LoRA targets must be unwrapped linear k/v projections: {sorted(invalid)}')
    for name,layer in list(model.named_modules()):

        # 过滤出cross attention使用的linear权重
        matched = name in filter_names if model_name == 'opt' else any(n in name for n in filter_names)
        if matched and isinstance(layer,nn.Linear):
            inject_lora(model,name,layer,LORA_R=r)
    # 冻结非Lora参数
    for name,param in model.named_parameters():
        if name.split('.')[-1] not in ['lora_a','lora_b']:  # 非LOra部分不计算梯度
            param.requires_grad=False
        else:
            param.requires_grad=True
    return model

def freeze_layer(model,layer):
    for name, param in model.named_parameters():
        if name in layer:
            param.requires_grad = False
    return model

def unfrozen(model):
    if isinstance(model,dict):
        for netid,net in model.items():
            for name, param in net.named_parameters():
                if name.split('.')[-1] not in ['lora_a', 'lora_b']:  # 非LOra部分不计算梯度
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    else:
        for name, param in model.named_parameters():
            if name.split('.')[-1] not in ['lora_a', 'lora_b']:  # 非LOra部分不计算梯度
                param.requires_grad = False
            else:
                param.requires_grad = True
    return model


def get_lora_parameter_names(model):
    return [name for name, _ in model.named_parameters() if name.split('.')[-1] in ['lora_a', 'lora_b']]


def build_lora_param_groups(named_params, lr, lr_a_scale=1.0, lora_weight_decay=0.0,
                            default_weight_decay=0.01):
    """
    Optimizer param groups with a dedicated LR for lora_a.

    Coordinate-descent A rounds train lora_a alone while lora_b (zero-init,
    trained from scratch) is still small; the gradient w.r.t. A and the effect
    of an A update on the output are both scaled by ||B||, so with the shared
    LR the loss barely moves. The lr_a_scale boost is applied only when lora_a
    is the sole trainable LoRA factor, so mixed A+B baselines are unaffected.

    Returns (groups, lr_a_effective, a_only, has_lora).
    """
    lora_a, lora_b, others = [], [], []
    for name, param in named_params:
        if not getattr(param, "requires_grad", True):
            continue
        suffix = name.split('.')[-1]
        if suffix == 'lora_a':
            lora_a.append(param)
        elif suffix == 'lora_b':
            lora_b.append(param)
        else:
            others.append(param)

    a_only = bool(lora_a) and not lora_b
    lr_a = lr * lr_a_scale if a_only else lr
    groups = []
    if lora_a:
        groups.append({"params": lora_a, "lr": lr_a, "weight_decay": lora_weight_decay})
    if lora_b:
        groups.append({"params": lora_b, "lr": lr, "weight_decay": lora_weight_decay})
    if others:
        groups.append({"params": others, "lr": lr, "weight_decay": default_weight_decay})
    return groups, lr_a, a_only, bool(lora_a or lora_b)


def clone_lora_state_dict(model, device="cpu"):
    state = {}
    for name, param in model.named_parameters():
        if name.split('.')[-1] in ['lora_a', 'lora_b']:
            state[name] = param.detach().to(device=device).clone()
    return state
