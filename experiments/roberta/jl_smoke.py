#!/usr/bin/env python3
"""Synthetic CPU-only end-to-end check. Never loads real data or model weights."""

import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


def main():
    from jl_run import METHODS, main as run
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--setting', choices=['5_5', '10_8'], default='5_5')
    p.add_argument('--output-root', required=True)
    p.add_argument('--dropout', choices=['model_default', 'disabled'], default='model_default')
    p.add_argument('--attention', choices=['eager', 'sdpa'], default='eager')
    p.add_argument('--workers', type=int, default=1)
    args = p.parse_args()
    import torch
    from datasets import Dataset
    from transformers import RobertaConfig, RobertaForSequenceClassification
    torch.set_num_threads(1)
    rounds_to_run = 6 if args.method == 'flexlora_cd' else 4
    if args.method == 'flexlora_cd':
        import jl_trainer
        actual_test = jl_trainer.test
        fixture = {'index': 0, 'last_loss': 1.0}
        losses = [1.0, 1.0, 1.1, 1.1, 1.0, 1.2]

        def server_feedback_fixture(*pos, **kw):
            accuracy, loss, metrics = actual_test(*pos, **kw)
            if kw.get('progress_prefix') == '[JL rank baseline] ':
                return accuracy, fixture['last_loss'], metrics
            # Explicit synthetic fixture exercises both directions, independent of luck.
            fixture['last_loss'] = losses[fixture['index']]
            fixture['index'] += 1
            return accuracy, fixture['last_loss'], metrics

        jl_trainer.test = server_feedback_fixture

    def tiny_model(*unused, **kwargs):
        config = RobertaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                               num_hidden_layers=12, num_attention_heads=2,
                               max_position_embeddings=514, num_labels=2)
        config._attn_implementation = kwargs.get('attn_implementation', 'eager')
        with torch.random.fork_rng():
            torch.manual_seed(123)
            model = RobertaForSequenceClassification(config)
        if kwargs.get('output_loading_info'):
            return model, {'missing_keys': ['classifier.dense.weight', 'classifier.out_proj.weight']}
        return model
    RobertaForSequenceClassification.from_pretrained = staticmethod(tiny_model)

    def split(n, offset):
        return Dataset.from_dict(dict(label=[i % 2 for i in range(n)],
            input_ids=[[0] + [4 + (i + offset + j) % 55 for j in range(6)] + [2] for i in range(n)],
            attention_mask=[[1] * 8 for _ in range(n)]))

    data = {role: [split(4 + 2 * (i % 2), 10 * r + i) for i in range(10)]
            for r, role in enumerate(['train', 'vali', 'test'])}
    data['client_names'] = ['synthetic%d_%d' % (i, 1000 if i % 2 == 0 else 1500) for i in range(10)]
    data['server_vali'] = split(8, 37)
    data['server_test'] = split(8, 47)
    with tempfile.TemporaryDirectory(prefix='jl_synthetic_') as folder:
        path = Path(folder) / 'synthetic.pkl'
        with path.open('wb') as handle:
            pickle.dump(data, handle)
        sys.argv = ['jl_smoke', '--method', args.method, '--setting', args.setting,
                    '--data-file', str(path), '--model-path', 'SYNTHETIC_ONLY', '--devices', 'cpu',
                    '--rounds', str(rounds_to_run), '--batch-size', '4', '--eval-batch-size', '4',
                    '--dropout', args.dropout,
                    '--attention', args.attention, '--workers', str(args.workers),
                    '--per-gpu-parallel', str(args.workers),
                    '--output-root', args.output_root, '--run-name', 'smoke']
        if args.method == 'flexlora_cd':
            sys.argv += ['--flex-rank-patience', '1', '--flex-rank-decrease-patience', '1', '--flex-rank-step', '4']
        run()
    root = Path(args.output_root) / args.setting / args.method / 'seed0_smoke'
    summary = json.loads((root / 'jl_summary.json').read_text())
    assert summary['n_clients'] == (1 if args.method == 'centralized' else (5 if args.setting == '5_5' else 10))
    if args.method == 'centralized':
        assert summary['primary_metric_scope'] == 'pooled'
        assert summary['n_source_clients'] == (5 if args.setting == '5_5' else 10)
    assert 0 <= summary['client_mean']['macro_f1'] <= 1
    assert not list(root.glob('jl_diagnostic_test_predictions/*'))
    progress = (root / 'jl_progress.log').read_text()
    total_rounds = rounds_to_run * (summary['n_clients'] if args.method == 'local_only' else 1)
    assert f"\u5df2\u5b8c\u6210 {total_rounds}/{total_rounds}" in progress
    assert "\u9884\u8ba1\u5269\u4f59" in progress
    assert "Validation" in progress and "Test" in progress
    assert "[Completed]" in progress and "[Failed]" not in progress
    assert progress.rindex("[Completed]") > progress.rindex("[Finalizing]")
    import csv
    for metrics_path in root.rglob('jl_round_metrics.csv'):
        with metrics_path.open() as handle:
            for row in csv.DictReader(handle):
                if row.get('training_aggregation_reward_seconds'):
                    assert 0 <= float(row['training_aggregation_reward_seconds']) < 3600
    if 'selection' in summary:
        assert 1 <= summary['selection']['round'] <= rounds_to_run
        import csv
        with (root / 'jl_client_test_metrics.csv').open() as handle:
            client_rows = list(csv.DictReader(handle))
        assert len({r['threshold'] for r in client_rows}) == 1
    if args.method in ('flexlora_cd', 'fixed_g12', 'spdcfl_cd'):
        import csv
        with (root / 'jl_round_metrics.csv').open() as handle:
            rank_rows = list(csv.DictReader(handle))
        if args.method != 'spdcfl_cd':
            assert all(int(r['G']) == 12 for r in rank_rows)
        if args.method == 'flexlora_cd':
            assert [int(r['lora_rank']) for r in rank_rows] == [10, 10, 10, 6, 10, 10]
            changes = json.loads((root / 'jl_rank_changes.json').read_text())
            assert [(r['old_rank'], r['new_rank']) for r in changes] == [(10, 6), (6, 10)]
            assert fixture['index'] == rounds_to_run
        else:
            assert all(int(r['lora_rank']) == 10 for r in rank_rows)
            assert not (root / 'jl_rank_changes.json').exists()
    if METHODS[args.method].get('controller_variant'):
        events = [json.loads(line) for line in (root / 'jl_controller_trajectory.jsonl').read_text().splitlines()]
        assert len(events) == rounds_to_run
        assert [e['executed_G'] for e in events[:2]] == [12, 12]
        assert events[-1]['terminal'] and events[-1]['next_decision'] is None
        for previous, current in zip(events[1:], events[2:]):
            assert previous['next_decision']['G'] == current['executed_G']
            assert current['credited_decision']['planned_for_round'] == current['round']
        checkpoint = torch.load(root / 'jl_study_controller_last.pt', map_location='cpu', weights_only=True)
        assert checkpoint['last_round'] == rounds_to_run
        if checkpoint['spec']['kind'] == 'ddqn':
            assert len(checkpoint['controller']['memory']) == rounds_to_run - 2
        else:
            assert sum(checkpoint['controller']['counts']) == rounds_to_run - 2
        assert not (root / 'jl_controller_last.pt').exists()
        if METHODS[args.method].get('component_ablation'):
            components = [json.loads(line) for line in
                          (root / 'jl_component_trajectory.jsonl').read_text().splitlines()]
            assert len(components) == rounds_to_run
            assert [set(e['factor_map'].values()) for e in components[:2]] == [{'B'}, {'A'}]
            for event, component in zip(events, components):
                assert event['round'] == component['round']
                assert event['executed_G'] == component['G'] == len(component['selected_layers'])
                assert len(set(component['selected_layers'])) == component['G']
                assert set(map(int, component['factor_map'])) == set(component['selected_layers'])
                if component['round'] > 2:
                    if METHODS[args.method]['component_ablation'] == 'random_layers':
                        assert component['layer_reason'] == 'uniform_random_layers'
                        assert all(m['reason'] != 'uniform_random_factor'
                                   for m in component['factor_metadata'].values())
                    else:
                        assert component['layer_reason'] != 'uniform_random_layers'
                        assert all(m['reason'] == 'uniform_random_factor'
                                   for m in component['factor_metadata'].values())
    elif METHODS[args.method]['layers'] == 'rl':
        import csv
        # This checkpoint was generated by the synthetic run above and is trusted.
        memory = torch.load(root / 'jl_controller_last.pt', map_location='cpu', weights_only=False)['memory']
        with (root / 'jl_round_metrics.csv').open() as handle:
            rounds = list(csv.DictReader(handle))
        assert len(memory) == 3
        for transition, row in zip(memory, rounds[1:]):
            assert transition[1] == int(row['G']) - 1
    print('JL_SYNTHETIC_PASS', args.method, args.setting)


if __name__ == '__main__':
    main()
