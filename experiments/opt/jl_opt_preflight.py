#!/usr/bin/env python3
"""Read-only OPT input checks, with an optional CPU-only model forward."""

import argparse
import json
from pathlib import Path
import pickle

from jl_opt_settings import MODEL_LAYERS, MODEL_REPO, MODEL_REVISION


def validate_token_split(split, vocab_size, pad_token_id):
    import numpy as np
    if not len(split):
        raise ValueError('Empty dataset split')
    for start in range(0, len(split), 128):
        rows = split[start:start + 128]
        ids, mask, labels = (np.asarray(rows[key]) for key in ('input_ids', 'attention_mask', 'label'))
        expected = (min(128, len(split) - start), 512)
        if ids.shape != expected or mask.shape != expected or labels.shape != expected[:1]:
            raise ValueError('Invalid OPT cached row dimensions')
        if (ids.dtype.kind not in 'iu' or mask.dtype.kind not in 'biu'
                or not np.isin(labels, [0, 1]).all()
                or (ids < 0).any() or (ids >= vocab_size).any()
                or not np.isin(mask, [0, 1]).all()):
            raise ValueError('Invalid OPT token IDs, masks or labels')
        if (mask.sum(axis=1) == 0).any() or ((mask[:, 1:] == 1) & (mask[:, :-1] == 0)).any():
            raise ValueError('OPT rows must have nonempty content and right padding')
        if (ids[mask == 0] != pad_token_id).any() or (ids[mask == 1] == pad_token_id).any():
            raise ValueError('Padding tokens would invalidate OPT classifier pooling')
    return len(split)


def validate_inputs(model_path, data_file):
    from jl_protocol import sha256
    from jl_opt_prepare_data import validate_opt_cache
    model_path = Path(model_path).resolve()
    config = json.loads((model_path / 'config.json').read_text())
    manifest = json.loads((model_path / 'jl_download_manifest.json').read_text())
    if (config['model_type'] != 'opt' or config['num_hidden_layers'] != MODEL_LAYERS
            or manifest['repo'] != MODEL_REPO or manifest['revision'] != MODEL_REVISION
            or manifest['status'] != 'verified'):
        raise ValueError('Incorrect OPT model or unverified download')
    for item in manifest['files']:
        if sha256(model_path / item['name']) != item['sha256']:
            raise ValueError('Downloaded model was modified: ' + item['name'])
    with Path(data_file).open('rb') as handle:
        data = pickle.load(handle)
    pad = config['pad_token_id']
    validate_opt_cache(data, pad_token_id=pad)
    if data['max_length'] != 512 or data['padding_side'] != 'right':
        raise ValueError('Incorrect OPT preprocessing contract')
    for name, digest in data['tokenizer_files_sha256'].items():
        if sha256(model_path / name) != digest:
            raise ValueError('Data cache tokenizer differs from model tokenizer: ' + name)
    audit = json.loads(Path(str(data_file) + '.audit.json').read_text())
    if audit['output_sha256'] != sha256(data_file) or not audit['audit']['all_equal']:
        raise ValueError('OPT data identity audit mismatch')
    for role in ('train', 'vali', 'test'):
        if len(data[role]) != 10:
            raise ValueError('Expected ten cached client splits')
        for split in data[role]:
            validate_token_split(split.with_format(None), config['vocab_size'], pad)
    validate_token_split(data['server_vali'].with_format(None), config['vocab_size'], pad)
    if 'server_test' in data:
        validate_token_split(data['server_test'].with_format(None), config['vocab_size'], pad)
    return dict(repo=MODEL_REPO, revision=MODEL_REVISION, layers=MODEL_LAYERS,
                source_data_sha256=data['source_sha256'], tokenizer_model=data['tokenizer_model'],
                data_sha256=audit['output_sha256'], sample_identity='verified', token_rows='verified',
                cached_split_sizes=audit['split_sizes'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--data-file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--load-model', action='store_true')
    args = parser.parse_args()
    result = validate_inputs(args.model_path, args.data_file)
    from jl_data import inspect_data
    result['datasets'] = [inspect_data(args.data_file, setting) for setting in ('5_5', '10_8')]
    result.update(training_performed=False, test_inference_performed=False, device='cpu')
    if args.load_model:
        import torch
        from jl_opt_model import build_opt_model
        from lora.lora_finetune import add_lora
        from jl_data import load_data
        torch.set_num_threads(1)
        torch.manual_seed(0)
        model = build_opt_model(args.model_path, 2, 'sdpa')
        add_lora(model, list(range(MODEL_LAYERS)), 10, 'opt')
        model.eval()
        rows = load_data(args.data_file, '5_5')['vali'][0][:2]
        with torch.no_grad():
            logits = model(input_ids=torch.tensor(rows['input_ids']),
                           attention_mask=torch.tensor(rows['attention_mask'])).logits
        if logits.shape != (2, 2) or not torch.isfinite(logits).all():
            raise RuntimeError('Invalid real OPT validation forward')
        adapters = [n for n, m in model.named_modules() if hasattr(m, 'lora_a')]
        result['model_check'] = dict(validation_forward='PASS', adapters=len(adapters),
                                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                                    head_frozen=not model.score.weight.requires_grad,
                                    missing_pretrained_keys=model._jl_loading_info.get('missing_keys', []),
                                    use_cache=model.config.use_cache)
        if len(adapters) != 24 or model.score.weight.requires_grad:
            raise RuntimeError('Unexpected OPT adapter/head configuration')
    from jl_protocol import write_json
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
