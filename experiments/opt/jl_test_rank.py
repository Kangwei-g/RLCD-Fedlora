"""CPU tests for common-rank resizing and cross-rank checkpoint restore."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from jl_rank import ValidationFeedbackRank, common_rank, lora_layout, match_checkpoint_layout, resize_common_rank


class Adapter(nn.Module):
    def __init__(self, rank=4):
        super().__init__()
        self.r, self.alpha = rank, 8.0
        self.lora_a = nn.Parameter(torch.randn(8, rank, dtype=torch.float64))
        self.lora_b = nn.Parameter(torch.randn(rank, 7, dtype=torch.float64))
        self.backbone = nn.Linear(8, 7, bias=False, dtype=torch.float64)
        self.backbone.requires_grad_(False)

    def product(self):
        return (self.alpha / self.r) * (self.lora_a @ self.lora_b)

    def forward(self, x):
        return self.backbone(x) + x @ self.product()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = Adapter()
        self.classifier = nn.Linear(7, 2, dtype=torch.float64)
        self.classifier.requires_grad_(False)

    def forward(self, x):
        return self.classifier(self.adapter(x))


class RankTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10)
        self.model = Model()

    def test_expansion_preserves_product_backbone_and_head(self):
        product = self.model.adapter.product().detach().clone()
        frozen = {n: p.clone() for n, p in self.model.named_parameters() if not p.requires_grad}
        changes = resize_common_rank(self.model, 6)
        torch.testing.assert_close(self.model.adapter.product(), product)
        self.assertEqual(common_rank(self.model), 6)
        self.assertEqual(changes[0]['relative_product_error'], 0)
        for name, p in self.model.named_parameters():
            if name in frozen:
                self.assertTrue(torch.equal(p, frozen[name]))
                self.assertFalse(p.requires_grad)
        self.assertTrue(torch.count_nonzero(self.model.adapter.lora_a[:, 4:]))
        self.assertEqual(torch.count_nonzero(self.model.adapter.lora_b[4:]), 0)
        self.model(torch.randn(3, 8, dtype=torch.float64)).square().sum().backward()
        self.assertGreater(self.model.adapter.lora_b.grad[4:].abs().sum().item(), 0)

    def test_compression_matches_scaled_product_truncated_svd(self):
        product = self.model.adapter.product().detach()
        u, s, vh = torch.linalg.svd(product, full_matrices=False)
        expected = (u[:, :2] * s[:2]) @ vh[:2]
        changes = resize_common_rank(self.model, 2)
        torch.testing.assert_close(self.model.adapter.product(), expected)
        self.assertAlmostEqual(changes[0]['relative_product_error'],
                               (s[2:].square().sum() / s.square().sum()).sqrt().item())

    def test_zero_product_remains_zero_but_learnable(self):
        with torch.no_grad():
            self.model.adapter.lora_b.zero_()
        resize_common_rank(self.model, 2)
        self.assertEqual(torch.count_nonzero(self.model.adapter.product()), 0)
        self.assertGreater(torch.count_nonzero(self.model.adapter.lora_a).item(), 0)

    def test_matching_client_shapes_does_not_change_rng(self):
        other = Model()
        resize_common_rank(self.model, 6)
        rng = torch.get_rng_state().clone()
        match_checkpoint_layout(other, self.model.state_dict(), lora_layout(self.model))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        other.load_state_dict(self.model.state_dict())
        torch.testing.assert_close(other.adapter.product(), self.model.adapter.product())
        self.assertEqual(common_rank(other), 6)

    def test_invalid_rank_leaves_model_unchanged(self):
        before = {n: t.clone() for n, t in self.model.state_dict().items()}
        for rank in [0, -1, 2.5, True, 8]:
            with self.assertRaises(ValueError):
                resize_common_rank(self.model, rank)
            self.assertEqual(common_rank(self.model), 4)
            for n, t in self.model.state_dict().items():
                self.assertTrue(torch.equal(t, before[n]))

    def test_restore_best_checkpoint_at_earlier_rank(self):
        from jl_protocol import load_checkpoint, save_checkpoint
        x = torch.randn(3, 8, dtype=torch.float64)
        expected = self.model(x).detach()
        args = SimpleNamespace(lora=True, jl_model_path='synthetic')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'best.pt'
            save_checkpoint(path, self.model, args, round=1)
            resize_common_rank(self.model, 6)
            with torch.no_grad():
                self.model.adapter.lora_b.add_(1)
            checkpoint = load_checkpoint(path, self.model)
            self.assertEqual(checkpoint['round'], 1)
            self.assertEqual(common_rank(self.model), 4)
            torch.testing.assert_close(self.model(x), expected)


class RankPolicyTests(unittest.TestCase):
    def test_worsening_shrinks_and_plateau_grows(self):
        policy = ValidationFeedbackRank()
        rows = [policy.observe(i + 1, loss) for i, loss in enumerate([.9, .8, .9, 1., 1.1])]
        self.assertEqual([r['rank'] for r in rows], [10] * 5)
        self.assertEqual(rows[-1]['proposed_next_rank'], 9)
        self.assertEqual(rows[-1]['reason'], 'shrink_after_worsening')
        policy.rebase(1.1)
        rows = [policy.observe(i, 1.1) for i in range(6, 11)]
        self.assertEqual([r['proposed_next_rank'] for r in rows], [9, 9, 9, 9, 10])
        self.assertEqual(rows[-1]['reason'], 'grow_after_plateau')

    def test_improvement_resets_plateau(self):
        policy = ValidationFeedbackRank(initial_rank=8, patience=2, warmup_rounds=0)
        ranks = [policy.observe(i + 1, loss)['proposed_next_rank']
                 for i, loss in enumerate([1., 1., .8, .8, .6, .6])]
        self.assertEqual(ranks, [8] * 6)

    def test_bounds_and_worsening_priority(self):
        policy = ValidationFeedbackRank(initial_rank=7, step=2, patience=1,
                                        decrease_patience=1, warmup_rounds=0)
        policy.observe(1, 1.)
        self.assertEqual(policy.observe(2, 2.)['proposed_next_rank'], 6)
        policy.rebase(2.)
        self.assertEqual(policy.observe(3, 3.)['reason'], 'rank_floor')
        self.assertEqual(policy.observe(4, 3.)['proposed_next_rank'], 8)
        policy.rebase(3.)
        self.assertEqual(policy.observe(5, 3.)['proposed_next_rank'], 10)
        policy.rebase(3.)
        self.assertEqual(policy.observe(6, 3.)['reason'], 'rank_cap')

    def test_feedback_validation(self):
        policy = ValidationFeedbackRank()
        for loss in [float('nan'), float('inf'), -1.]:
            with self.assertRaises(ValueError):
                policy.observe(1, loss)
        policy.observe(1, 1.)
        with self.assertRaises(ValueError):
            policy.observe(1, 1.)
        for kwargs in [dict(min_rank=10), dict(max_rank=9), dict(step=0),
                       dict(decrease_patience=0), dict(patience=0), dict(min_delta=float('nan'))]:
            with self.assertRaises(ValueError):
                ValidationFeedbackRank(**kwargs)

    def test_fixed_and_adaptive_method_definitions_are_distinct(self):
        from jl_run import engine_arguments, parser
        argv = {}
        for method in ['flexlora_cd', 'fixed_g12', 'spdcfl_cd', 'rlcd']:
            values = engine_arguments(parser(method).parse_args([]), Path('/tmp/synthetic'))
            argv[method] = {key: values[i + 1] for i, key in enumerate(values[:-1])
                            if key.startswith('--') and not values[i + 1].startswith('--')}
        self.assertEqual(argv['flexlora_cd']['--jl_rank_policy'], 'server_validation_feedback')
        self.assertEqual(argv['flexlora_cd']['--min_rank'], '6')
        for method in ['fixed_g12', 'spdcfl_cd', 'rlcd']:
            self.assertEqual(argv[method]['--min_rank'], '10')
            self.assertEqual(argv[method]['--max_rank'], '10')
            self.assertNotIn('--jl_rank_policy', argv[method])
        self.assertEqual(argv['fixed_g12']['--factor_selection_mode'], argv['flexlora_cd']['--factor_selection_mode'])
        self.assertEqual(argv['fixed_g12']['--layer_select_method'], 'all')
        self.assertEqual(argv['flexlora_cd']['--layer_select_method'], 'all')
        self.assertEqual(argv['spdcfl_cd']['--layer_select_method'], 'spdcfl')


if __name__ == '__main__':
    unittest.main()
