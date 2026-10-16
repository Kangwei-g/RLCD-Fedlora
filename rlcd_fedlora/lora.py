"""Minimal LoRA layers used by the public RLCD-FedLoRA core."""

import math
from typing import Iterable, List

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Wrap a frozen linear layer with `x @ A @ B * alpha / rank`."""

    def __init__(self, linear: nn.Linear, rank: int, alpha: float = 8.0):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be positive")
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.lora_a = nn.Parameter(torch.empty(linear.in_features, self.rank))
        self.lora_b = nn.Parameter(torch.zeros(self.rank, linear.out_features))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.raw_linear = linear
        for parameter in self.raw_linear.parameters():
            parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.raw_linear(inputs)
        delta = inputs @ (self.lora_a @ self.lora_b)
        return base + delta * (self.alpha / self.rank)


def _resolve_parent(model: nn.Module, dotted_name: str):
    parts = dotted_name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora_linear(
    model: nn.Module,
    module_name: str,
    rank: int,
    alpha: float = 8.0,
) -> None:
    parent, child_name = _resolve_parent(model, module_name)
    linear = getattr(parent, child_name)
    if isinstance(linear, LoRALinear):
        return
    if not isinstance(linear, nn.Linear):
        raise TypeError("LoRA can only be injected into nn.Linear modules")
    setattr(parent, child_name, LoRALinear(linear, rank=rank, alpha=alpha))


def transformer_target_names(
    layers: Iterable[int], model_name: str = "roberta"
) -> List[str]:
    layer_ids = [int(layer_id) for layer_id in layers]
    if model_name == "roberta":
        return [
            "roberta.encoder.layer.{}.attention.self.{}".format(layer_id, projection)
            for layer_id in layer_ids
            for projection in ("key", "value")
        ]
    if model_name == "gpt":
        return [
            "transformer.h.{}.attn.c_attn.{}".format(layer_id, projection)
            for layer_id in layer_ids
            for projection in ("k_proj", "v_proj")
        ]
    raise ValueError("Unsupported model_name: {}".format(model_name))


def inject_transformer_lora(
    model: nn.Module,
    layers: Iterable[int],
    rank: int,
    alpha: float = 8.0,
    model_name: str = "roberta",
) -> nn.Module:
    available = dict(model.named_modules())
    target_names = transformer_target_names(layers, model_name=model_name)
    missing = [name for name in target_names if name not in available]
    if missing:
        raise ValueError("LoRA target modules are missing: {}".format(missing))
    for name in target_names:
        inject_lora_linear(model, name, rank=rank, alpha=alpha)
    freeze_non_lora_parameters(model)
    return model


def freeze_non_lora_parameters(model: nn.Module) -> nn.Module:
    """Freeze the backbone and classification head; enable both LoRA factors."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.rsplit(".", 1)[-1] in {"lora_a", "lora_b"}
    return model


def lora_parameter_names(model: nn.Module) -> List[str]:
    return [
        name
        for name, _ in model.named_parameters()
        if name.rsplit(".", 1)[-1] in {"lora_a", "lora_b"}
    ]
