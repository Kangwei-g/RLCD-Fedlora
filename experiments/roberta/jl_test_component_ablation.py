import ast
import json
from pathlib import Path
import random
import tempfile
import unittest

from jl_component_ablation import ComponentAblation, METHOD_MODES
from jl_run import parser, engine_arguments


class ComponentTests(unittest.TestCase):
    def test_only_component_cli_differs_from_d01(self):
        for setting in ('5_5', '10_8'):
            for seed in (0, 4, 5):
                for method, mode in METHOD_MODES.items():
                    def arguments(name):
                        args = parser().parse_args(['--method', name, '--setting', setting,
                                                    '--seed', str(seed)])
                        return engine_arguments(args, Path('/tmp/unused'))
                    actual, expected = arguments(method), arguments('ddqn_s01')
                    self.assertEqual(actual[actual.index('--jl_component_ablation') + 1], mode)
                    i = actual.index('--jl_component_ablation')
                    del actual[i:i + 2]
                    actual[actual.index('--jl_method') + 1] = 'ddqn_s01'
                    self.assertEqual(actual, expected)

    def test_warmup_and_other_component_unchanged(self):
        layer_ref = ([3, 7], 'reference', .75)
        factor_ref = ({3: 'A', 7: 'B'}, {3: {'reason': 'reference'}})
        for mode in ('none', 'random_layers', 'random_factors'):
            a = ComponentAblation(mode, 0)
            state = a.rng.getstate()
            for t in (1, 2):
                self.assertIs(a.layers(layer_ref, range(12), 2, t), layer_ref)
                self.assertIs(a.factors(factor_ref, [3, 7], t), factor_ref)
            self.assertEqual(state, a.rng.getstate())
            if mode != 'random_layers':
                self.assertIs(a.layers(layer_ref, range(12), 2, 3), layer_ref)
            if mode != 'random_factors':
                self.assertIs(a.factors(factor_ref, [3, 7], 3), factor_ref)

    def test_layer_budget_uniformity_and_reproducibility(self):
        a, b = [ComponentAblation('random_layers', 4) for _ in range(2)]
        counts = [0] * 12
        for t in range(3, 12003):
            budget = (t % 12) + 1
            result = a.layers(([], 'unused', None), range(12), budget, t)
            self.assertEqual(result, b.layers(([], 'unused', None), range(12), budget, t))
            selected, reason, draw = result
            self.assertEqual(len(set(selected)), budget)
            self.assertTrue(set(selected) <= set(range(12)))
            self.assertEqual(reason, 'uniform_random_layers')
            for layer in selected:
                counts[layer] += 1
        self.assertLess(max(counts) - min(counts), 400)

    def test_factor_distribution_is_per_layer_not_common(self):
        a, b = [ComponentAblation('random_factors', 5) for _ in range(2)]
        count_a, mixed = 0, 0
        for t in range(3, 1003):
            result = a.factors(({}, {}), range(12), t)
            self.assertEqual(result, b.factors(({}, {}), range(12), t))
            factors, metadata = result
            count_a += list(factors.values()).count('A')
            mixed += len(set(factors.values())) == 2
            self.assertEqual(set(factors), set(range(12)))
            self.assertTrue(all(m['reason'] == 'uniform_random_factor' for m in metadata.values()))
        self.assertTrue(5700 < count_a < 6300)
        self.assertGreater(mixed, 990)

    def test_global_rng_untouched_and_diagnostics(self):
        random.seed(56)
        before = random.getstate()
        with tempfile.TemporaryDirectory() as directory:
            a = ComponentAblation('random_factors', 0, directory)
            factors, meta = a.factors(({}, {}), [2, 5], 3)
            a.record(3, [2, 5], factors, 'score_80pct', meta)
            event = json.loads(Path(directory, 'jl_component_trajectory.jsonl').read_text())
            self.assertEqual(event['G'], 2)
            self.assertFalse(event['warmup'])
            self.assertEqual(event['stage'], 'selected_before_training')
        self.assertEqual(before, random.getstate())

    def test_engine_hooks_follow_reference_and_record_once(self):
        tree = ast.parse(Path(__file__).with_name('jl_engine.py').read_text())
        hooks = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                 and n.func.value.id == 'jl_component']
        self.assertCountEqual([n.func.attr for n in hooks], ['factors', 'layers', 'record'])
        for hook in hooks:
            if hook.func.attr in ('layers', 'factors'):
                self.assertIsInstance(hook.args[-1], ast.BinOp)
                self.assertEqual(hook.args[-1].right.value, 2 if hook.func.attr == 'layers' else 1)


if __name__ == '__main__':
    unittest.main()
