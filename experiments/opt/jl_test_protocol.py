"""CPU regression tests for JL selection, metrics and aggregation."""

import itertools
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from jl_aggregation import aggregate_uploaded_deltas
from jl_baseline_hooks import aggregate_flora_lora
from jl_protocol import (EvaluationProtocol, binary_metrics, confusion_metrics,
                         select_shared_threshold, summarize_predictions)
from jl_run import parser, engine_arguments, METHODS


def prediction(cid=0, labels=(0, 1), probabilities=(0.1, 0.9)):
    return dict(client_id=cid, client_name=str(cid), labels=np.array(labels),
                probabilities=np.array(probabilities), loss=0.2)


class ProtocolTests(unittest.TestCase):
    def test_signed_consistency_can_detect_cancellation(self):
        from jl_layer_selection import SPDCFLLayerSelector
        selector = SPDCFLLayerSelector(ema_decay=0)
        self.assertAlmostEqual(selector._sgc(np.array([1., -1.])), 0)
        self.assertAlmostEqual(selector._sgc(np.array([1., 1.])), 1, places=6)

    def test_defaults_and_method_entries(self):
        for method in METHODS:
            args = parser(method).parse_args([])
            self.assertEqual((args.seed, args.rounds, args.epochs, args.setting), (0, 200, 1, '5_5'))
            self.assertFalse(args.test_every_round)
            self.assertIn('--jl_data_file', engine_arguments(args, Path('/tmp/example')))

    def test_macro_recall_is_not_positive_recall(self):
        m = confusion_metrics([0, 0, 0, 1], [0.1, 0.2, 0.7, 0.8], 0.5)
        self.assertAlmostEqual(m['macro_recall'], (2 / 3 + 1) / 2)
        self.assertEqual(m['recall_cls1'], 1)
        self.assertNotEqual(m['macro_recall'], m['recall_cls1'])

    def test_equal_client_threshold_not_pooled(self):
        clients = [prediction(0, [0, 1], [.4, .45]),
                   prediction(1, [0, 1] * 10, [.55, .6] * 10)]
        threshold, score = select_shared_threshold(clients, [.425, .575])
        self.assertEqual(threshold, .425)
        self.assertAlmostEqual(score, 2 / 3)
        _, summary = summarize_predictions(clients, threshold)
        self.assertNotEqual(summary['client_mean']['macro_f1'], summary['pooled']['macro_f1'])

    def test_net_benefit_threshold_is_independent(self):
        a = binary_metrics([0, 1, 1, 0], [.4, .8, .2, .1], .5)
        b = binary_metrics([0, 1, 1, 0], [.4, .8, .2, .1], .9)
        self.assertEqual(a['net_benefit_p030'], b['net_benefit_p030'])
        self.assertNotEqual(a['decision_loss_per_10000'], b['decision_loss_per_10000'])
        self.assertIsNone(binary_metrics([0, 0], [.1, .2], .5)['auroc'])

    def test_delta_zero_and_untouched_parameters(self):
        state = {'x': torch.tensor([4.]), 'untouched': torch.tensor([7.])}
        updated, _, _, _ = aggregate_uploaded_deltas([{'x': state['x'].clone()}], [.7], state)
        self.assertTrue(torch.equal(updated['x'], state['x']))
        self.assertNotIn('untouched', updated)

    def test_inverse_probability_delta_unbiased(self):
        sizes, deltas, base = [1, 2, 4], [1., -2., 3.], torch.tensor([8.])
        estimates = []
        for selected in itertools.combinations(range(3), 2):
            params = [{'x': base + deltas[k]} for k in selected]
            weights = [3 / 2 * sizes[k] / sum(sizes) for k in selected]
            value, _, _, _ = aggregate_uploaded_deltas(params, weights, {'x': base})
            estimates.append(value['x'].item())
        expected = 8 + sum(n * d for n, d in zip(sizes, deltas)) / sum(sizes)
        self.assertAlmostEqual(np.mean(estimates), expected, places=6)

    def test_product_aggregation_scale_and_zero_update(self):
        prefix = 'roberta.encoder.layer.0.attention.self.key.'
        a, b = prefix + 'lora_a', prefix + 'lora_b'
        state = {a: torch.eye(2), b: torch.eye(2)}
        model = SimpleNamespace(state_dict=lambda: state)
        zero, _ = aggregate_flora_lora([state], [.6], global_model=model,
                                      next_round_layer=[0], rank=2, lora_alpha=8)
        self.assertTrue(torch.equal(zero[a], state[a]))
        payload = {a: torch.eye(2) * 2, b: torch.eye(2)}
        result, _ = aggregate_flora_lora([payload], [.6], global_model=model,
                                        next_round_layer=[0], rank=2, lora_alpha=8)
        torch.testing.assert_close(result[a] @ result[b] * 4, torch.eye(2) * 6.4)

    def test_checkpoint_earliest_tie_and_test_only_at_end(self):
        class FakeProtocol(EvaluationProtocol):
            def predict(self, model, datasets, split):
                self.calls.append(split)
                return [prediction(cid) for cid in range(2)]
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(lora=False, jl_model_path='synthetic', jl_method='rlcd',
                                   init_seed=0, jl_setting='5_5', jl_test_every_round=False)
            model = torch.nn.Linear(1, 1, bias=False)
            p = FakeProtocol(args, folder, [1, 2], [3, 4], ['a', 'b'])
            p.calls = []
            with torch.no_grad():
                model.weight.fill_(1)
            p.validate(model, 1)
            with torch.no_grad():
                model.weight.fill_(9)
            p.validate(model, 2)
            self.assertEqual(p.calls, ['validation', 'validation'])
            summary = p.finalize(model)
            self.assertEqual(summary['selection']['round'], 1)
            self.assertEqual(summary['n_clients'], 2)
            self.assertEqual(model.weight.item(), 1)
            self.assertEqual(p.calls.count('test'), 1)


if __name__ == '__main__':
    unittest.main()
