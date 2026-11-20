"""Local OPT-125M sequence-classification backbone and LoRA coordinates."""

from numbers import Integral
from pathlib import Path
import re

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers.models.opt import modeling_opt


def get_opt_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        raise ValueError("OPT tokenizer must define a separate padding token")
    tokenizer.padding_side = "right"
    return tokenizer


def _verify_attention(model, attention):
    if model.config._attn_implementation != attention:
        raise RuntimeError("Loaded OPT attention backend does not match the requested backend")
    for layer in model.model.decoder.layers:
        module = layer.self_attn
        # Older Transformers use separate classes; newer versions dispatch from OPTAttention.
        classes = getattr(modeling_opt, "OPT_ATTENTION_CLASSES", None)
        if classes is not None:
            valid = type(module) is classes.get(attention)
        else:
            valid = type(module) is modeling_opt.OPTAttention
            if attention == "sdpa":
                registry = getattr(modeling_opt, "ALL_ATTENTION_FUNCTIONS", {})
                from transformers.integrations.sdpa_attention import sdpa_attention_forward
                valid = valid and registry.get("sdpa") is sdpa_attention_forward
        if not valid:
            raise RuntimeError(f"Requested OPT {attention}, but loaded {type(module).__name__}")


def build_opt_model(model_path, out_dim, attention, micro_batch_size=64):
    """Build the frozen backbone; callers inject LoRA via add_lora(..., 'opt').

    Seed once before building the global model, then load its state into clients,
    including the randomly initialized, frozen score head. Do not reseed here.
    """
    if not Path(model_path).is_dir():
        raise ValueError("OPT model_path must be an existing local model directory")
    if out_dim != 2:
        raise ValueError("The OPT experiment requires exactly two labels")
    if attention not in ("sdpa", "eager"):
        raise ValueError("OPT attention must be 'sdpa' or 'eager'")
    if (isinstance(micro_batch_size, bool) or not isinstance(micro_batch_size, Integral)
            or micro_batch_size < 1):
        raise ValueError("micro_batch_size must be a positive integer")
    config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    if (config.model_type != "opt" or config.num_hidden_layers != 12
            or config.hidden_size != 768 or config.ffn_dim != 3072):
        raise ValueError("Expected an OPT-125M backbone with 12 decoder layers")
    pad_id = get_opt_tokenizer(model_path).pad_token_id
    if config.pad_token_id != pad_id or pad_id == config.eos_token_id:
        raise ValueError("OPT requires its original separate padding token")
    config.num_labels = out_dim
    config.use_cache = False
    config.output_attentions = False
    config.output_hidden_states = False
    config.return_dict = True
    model, loading_info = AutoModelForSequenceClassification.from_pretrained(
        model_path, config=config, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float32, attn_implementation=attention, output_loading_info=True,
        weights_only=True)
    if (not isinstance(model, modeling_opt.OPTForSequenceClassification)
            or len(model.model.decoder.layers) != 12):
        raise ValueError("Loaded model is not the expected OPT sequence classifier")
    _verify_attention(model, attention)
    model.float().requires_grad_(False)
    model._jl_loading_info = loading_info
    model._jl_micro_batch_size = int(micro_batch_size)
    model._jl_attention_implementation = attention
    return model


def model_layer_index(name):
    """Extract OPT, RoBERTa or legacy GPT layer indices, including wrapped names."""
    match = re.search(r"(?:^|\.)(?:model\.decoder\.layers|roberta\.encoder\.layer|transformer\.h)\.(\d+)(?:\.|$)", name)
    if match is None:
        raise ValueError(f"Unrecognized model layer parameter: {name}")
    return int(match.group(1))


def factor_param_names(i, factor):
    factor = str(factor).upper()
    if factor not in ("A", "B", "LORA_A", "LORA_B"):
        raise ValueError(f"Unknown LoRA factor: {factor}")
    suffix = "lora_a" if factor in ("A", "LORA_A") else "lora_b"
    return [f"model.decoder.layers.{int(i)}.self_attn.{projection}.{suffix}"
            for projection in ("k_proj", "v_proj")]


def frozen_for_selection(layer, matrix, total_layers=12):
    """Match legacy semantics: 0 uploads A, 1 uploads B; unselected freezes both.

    Selected factors excluded from communication remain trainable here. The
    coordinate-descent controller applies its separate training-factor mask.
    """
    selected = list(layer)
    if any(i < 0 or i >= total_layers for i in selected):
        raise ValueError("Selected OPT layer is outside the model")
    unselected = [i for i in range(total_layers) if i not in selected]
    frozen = [f"model.decoder.layers.{i}.self_attn.{projection}.lora_{factor}"
              for i in unselected for projection in ("k_proj", "v_proj") for factor in ("a", "b")]
    nocom_names = frozen.copy()
    nocom_marks = [f"{i}.{projection}.{factor}" for projection in ("key", "value")
                   for i in unselected for factor in (0, 1)]
    if 0 in matrix.values() or 1 in matrix.values():
        for i in selected:
            if matrix[i] in (0, 1):
                excluded = "B" if matrix[i] == 0 else "A"
                nocom_names.extend(factor_param_names(i, excluded))
                nocom_marks.extend(f"{i}.{projection}.{1 - matrix[i]}"
                                   for projection in ("key", "value"))
    return frozen, nocom_marks, nocom_names
