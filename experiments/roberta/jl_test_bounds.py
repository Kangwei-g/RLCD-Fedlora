"""Regression checks for joint, all-layer LoRA training in both controls."""

import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
from unittest.mock import patch

import torch
from datasets import Dataset
from transformers import RobertaConfig, RobertaForSequenceClassification

import jl_bounds
from jl_run import METHODS, engine_arguments, parser
from lora.lora_finetune import add_lora


class BoundsTests(unittest.TestCase):
    def test_centralized_validation_and_test_are_pooled_separately(self):
        validation = [Dataset.from_dict(dict(label=[0, 1])), Dataset.from_dict(dict(label=[1]))]
        tests = [Dataset.from_dict(dict(label=[0])), Dataset.from_dict(dict(label=[0, 0]))]
        data = dict(client_names=['a', 'b'], vali=validation, test=tests)
        with tempfile.TemporaryDirectory() as output:
            protocol = jl_bounds.PooledCentralizedProtocol(SimpleNamespace(), output, data)
            self.assertEqual(protocol.names, ['pooled_all_clients'])
            self.assertEqual(protocol.validation[0]['label'], [0, 1, 1])
            self.assertEqual(protocol.tests[0]['label'], [0, 0, 0])

    def engine_argv(self, method, setting='5_5'):
        options = parser(method).parse_args(['--setting', setting])
        return ['jl_engine.py'] + engine_arguments(options, Path('/tmp/jl_bounds_test'))

    def test_resolved_configuration_for_all_methods_and_settings(self):
        for setting in ('5_5', '10_8'):
            for method in METHODS:
                with self.subTest(setting=setting, method=method):
                    with patch('sys.argv', self.engine_argv(method, setting)):
                        args = jl_bounds.base.get_args()
                    if method in ('centralized', 'local_only'):
                        self.assertEqual(args.factor_selection_mode, 'joint')
                        self.assertEqual(args.coordinate_schedule, 'none')
                        self.assertEqual(args.factor_warmup_rounds, 0)
                        self.assertEqual(args.layer_select_method, 'all')
                        self.assertEqual((args.initial_rank, args.min_rank, args.max_rank), (10, 10, 10))
                        self.assertTrue(args.lora)
                        self.assertFalse(args.if_rl)
                    else:
                        self.assertNotEqual(args.factor_selection_mode, 'joint')
                        self.assertEqual(args.coordinate_schedule, 'baba')

    def test_controls_reject_alternating_schedule(self):
        for method in ('centralized', 'local_only'):
            argv = self.engine_argv(method)
            argv[argv.index('--coordinate_schedule') + 1] = 'bba'
            with patch('sys.argv', argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    jl_bounds.base.get_args()

    def test_federated_entry_rejects_control_only_joint_mode(self):
        argv = self.engine_argv('rlcd')
        argv[argv.index('--factor_selection_mode') + 1] = 'joint'
        with patch('sys.argv', argv), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                jl_bounds.base.get_args()

    def test_all_adapters_train_and_non_lora_weights_stay_frozen(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        model = RobertaForSequenceClassification(RobertaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=12, num_attention_heads=2,
            max_position_embeddings=32, num_labels=2,
            hidden_dropout_prob=0, attention_probs_dropout_prob=0))
        add_lora(model, list(range(12)), 10, 'roberta')
        dataset = Dataset.from_dict(dict(
            label=[0, 1] * 4,
            input_ids=[[0, 4 + i, 12 + i, 2] for i in range(8)],
            attention_mask=[[1] * 4 for _ in range(8)]))
        args = SimpleNamespace(epochs=1, lr=5e-4, optimizer='adam', device='cpu',
                               finetune=None, model='roberta', batch_size=2,
                               max_length=32, lr_a_scale=1, lora_weight_decay=0)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        factors = [name for name in before if name.endswith(('lora_a', 'lora_b'))]
        self.assertEqual(len(factors), 12 * 2 * 2)

        # Start with stale freezes to ensure every epoch restores all adapters.
        with patch.object(jl_bounds.base, 'get_coordinate_factor', side_effect=AssertionError('No CD schedule')):
            for _ in range(3):
                model.requires_grad_(False)
                jl_bounds.train_epoch(model, dataset, args)
                for name, parameter in model.named_parameters():
                    self.assertEqual(parameter.requires_grad, name in factors, name)
                    if name in factors:
                        self.assertIsNotNone(parameter.grad, name)
        for name, parameter in model.named_parameters():
            if name in factors:
                self.assertFalse(torch.equal(before[name], parameter.detach()), name)
            else:
                torch.testing.assert_close(before[name], parameter.detach(), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
