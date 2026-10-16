"""Parameter masking and selected-factor payload utilities."""

from collections import OrderedDict
from typing import Dict, Iterable, List, Mapping, Sequence, Set

import torch
from torch import nn


TensorPayload = Mapping[str, torch.Tensor]


def factor_parameter_names(
    layer_id: int, factor: str, model_name: str = "roberta"
) -> List[str]:
    factor = str(factor).upper()
    if factor not in {"A", "B"}:
        raise ValueError("Factor must be A or B")
    suffix = "lora_a" if factor == "A" else "lora_b"
    if model_name == "roberta":
        return [
            "roberta.encoder.layer.{}.attention.self.key.{}".format(
                int(layer_id), suffix
            ),
            "roberta.encoder.layer.{}.attention.self.value.{}".format(
                int(layer_id), suffix
            ),
        ]
    if model_name == "gpt":
        return [
            "transformer.h.{}.attn.c_attn.k_proj.{}".format(
                int(layer_id), suffix
            ),
            "transformer.h.{}.attn.c_attn.v_proj.{}".format(
                int(layer_id), suffix
            ),
        ]
    raise ValueError("Unsupported model_name: {}".format(model_name))


def expected_payload_names(
    factor_map: Mapping[int, str], model_name: str = "roberta"
) -> Set[str]:
    expected = set()  # type: Set[str]
    for layer_id, factor in factor_map.items():
        expected.update(
            factor_parameter_names(layer_id, factor, model_name=model_name)
        )
    return expected


def set_trainable_factor_map(
    model: nn.Module,
    factor_map: Mapping[int, str],
    model_name: str = "roberta",
) -> List[str]:
    """Freeze every parameter except the selected per-layer LoRA factors."""
    expected = expected_payload_names(factor_map, model_name=model_name)
    found = set()
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name in expected
        if name in expected:
            found.add(name)
    missing = sorted(expected - found)
    if missing:
        raise ValueError("Selected LoRA parameters are missing: {}".format(missing))
    return sorted(found)


def trainable_parameter_names(model: nn.Module) -> List[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def assert_only_selected_factors_trainable(
    model: nn.Module,
    factor_map: Mapping[int, str],
    model_name: str = "roberta",
) -> None:
    expected = expected_payload_names(factor_map, model_name=model_name)
    actual = set(trainable_parameter_names(model))
    if actual != expected:
        raise AssertionError(
            "Trainable parameters do not match factor_map; unexpected={}, missing={}".format(
                sorted(actual - expected), sorted(expected - actual)
            )
        )


def extract_factor_payload(
    model: nn.Module,
    factor_map: Mapping[int, str],
    model_name: str = "roberta",
    device: str = "cpu",
) -> "OrderedDict[str, torch.Tensor]":
    expected = expected_payload_names(factor_map, model_name=model_name)
    payload = OrderedDict()
    for name, parameter in model.named_parameters():
        if name in expected:
            payload[name] = parameter.detach().to(device=device).clone()
    missing = sorted(expected - set(payload))
    if missing:
        raise ValueError("Cannot build payload; parameters are missing: {}".format(missing))
    return payload


def validate_factor_payloads(
    payloads: Sequence[TensorPayload],
    factor_map: Mapping[int, str],
    model_name: str = "roberta",
) -> None:
    expected = expected_payload_names(factor_map, model_name=model_name)
    for client_index, payload in enumerate(payloads):
        names = set(payload.keys())
        if names != expected:
            raise ValueError(
                "Client {} payload mismatch; unexpected={}, missing={}".format(
                    client_index,
                    sorted(names - expected),
                    sorted(expected - names),
                )
            )


def apply_factor_payload(model: nn.Module, payload: TensorPayload) -> None:
    parameter_map = dict(model.named_parameters())
    missing = sorted(set(payload) - set(parameter_map))
    if missing:
        raise ValueError("Global model is missing payload parameters: {}".format(missing))
    with torch.no_grad():
        for name, value in payload.items():
            parameter = parameter_map[name]
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def clone_lora_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name.rsplit(".", 1)[-1] in {"lora_a", "lora_b"}
    }


def selected_factor_delta_norm(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    factor_map: Mapping[int, str],
    model_name: str = "roberta",
) -> float:
    """Compute q_t as the sum of selected-factor Frobenius changes."""
    total = 0.0
    for name in sorted(expected_payload_names(factor_map, model_name=model_name)):
        if name not in before or name not in after:
            raise ValueError("LoRA state is missing {}".format(name))
        delta = after[name].float() - before[name].float()
        total += float(torch.norm(delta, p="fro").item())
    return total
