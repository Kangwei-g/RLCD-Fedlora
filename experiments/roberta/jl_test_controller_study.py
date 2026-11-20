import json
import ast
import random
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from jl_controller_variants import VARIANTS, ORDER, get_spec, client_shuffle_seed
from jl_controller_study import ControllerStudy, StudyDDQN, encode_state, epsilon_at, reward_potential

class ControllerTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_registry_and_cli(self):
        from jl_run import MAIN_METHODS, parser, engine_arguments
        self.assertEqual(len(ORDER), 1)
        self.assertEqual([sum((s['kind'] == k for s in VARIANTS.values())) for k in ('ddqn', 'ucb', 'ts')], [1, 0, 0])
        self.assertEqual(len(MAIN_METHODS), 9)
        for name in ORDER:
            args = parser().parse_args(['--method', name])
            argv = engine_arguments(args, Path('/tmp/test'))
            self.assertEqual(argv[argv.index('--jl_controller_variant') + 1], name)
            self.assertEqual(float(argv[argv.index('--factor_epsilon') + 1]), VARIANTS[name]['coordinate_exploration'])

    def test_state_and_schedule(self):
        self.assertEqual(encode_state((200, 4, 1), 200, True), [1, 0.8, 0.5])
        self.assertEqual(encode_state((200, 4, 1), 200, False), [200, 4, 1])
        with self.assertRaises(ValueError):
            encode_state((2, float('nan'), 1), 200, True)
        self.assertEqual(epsilon_at(2, True), 0.5)
        self.assertAlmostEqual(epsilon_at(80, True), 0.05)
        self.assertEqual(epsilon_at(200, True), 0.05)
        self.assertGreater(epsilon_at(200, False), 0.36)

    def test_rewards(self):
        a = dict(macro_f1=0.4, balanced_ce=1, balanced_brier=0.3)
        b = dict(macro_f1=0.6, balanced_ce=0.5, balanced_brier=0.2)
        for mode in {s['reward'] for s in VARIANTS.values()}:
            self.assertLess(reward_potential(a, mode), reward_potential(b, mode))
            self.assertTrue(0 <= reward_potential(b, mode) <= 1)

    def test_private_rng(self):
        torch.manual_seed(19)
        before = torch.get_rng_state().clone()
        a = StudyDDQN(get_spec('ddqn_s01'), 0, 12)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        b = StudyDDQN(get_spec('ddqn_s01'), 0, 12)
        self.assertEqual([a.choose([2, 1, 1], t)[0] for t in range(2, 30)], [b.choose([2, 1, 1], t)[0] for t in range(2, 30)])
        seeds = [client_shuffle_seed(0, r, c) for r in range(200) for c in range(5)]
        self.assertEqual(len(set(seeds)), 1000)

        def order(c):
            return torch.randperm(20, generator=torch.Generator().manual_seed(client_shuffle_seed(0, 5, c)))
        first = {c: order(c) for c in range(5)}
        second = {c: order(c) for c in reversed(range(5))}
        self.assertTrue(all((torch.equal(first[c], second[c]) for c in range(5))))

    def test_round_alignment_and_safe_snapshots(self):
        for name in ORDER:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                controller = ControllerStudy(name, 0, 4, directory)
                g = 12
                for t in range(1, 5):
                    metrics = dict(macro_f1=0.4 + t / 100, balanced_ce=1 - t / 100, balanced_brier=0.3 - t / 100, threshold=0.5)
                    next_g, event = controller.finish_round(t, (t, 0.3, 0.5), metrics, g)
                    if t <= 2:
                        self.assertIsNone(event['learning'])
                    else:
                        self.assertEqual(event['credited_decision']['planned_for_round'], t)
                        self.assertAlmostEqual(event['reward'], event['potential'] - previous)
                    previous = event['potential']
                    g = 12 if next_g is None else next_g
                self.assertIsNone(event['next_decision'])
                events = [json.loads(s) for s in Path(directory, 'jl_controller_trajectory.jsonl').read_text().splitlines()]
                self.assertEqual(len(events), 4)
                state = torch.load(Path(directory, 'jl_study_controller_last.pt'), weights_only=True)
                self.assertEqual(state['last_round'], 4)
                self.assertIsNone(state['pending'])

    def test_rejects_misattributed_action(self):
        controller = ControllerStudy('ddqn_s01', 0, 4)
        metrics = dict(macro_f1=0.5, balanced_ce=0.7, balanced_brier=0.2, threshold=0.5)
        controller.finish_round(1, (1, 0, 0.7), metrics, 12)
        g, _ = controller.finish_round(2, (2, 0, 0.7), metrics, 12)
        with self.assertRaises(RuntimeError):
            controller.finish_round(3, (3, 0, 0.7), metrics, g % 12 + 1)

    def test_zero_exploration_has_deterministic_factor_ties(self):
        source = ast.parse(Path(__file__).with_name('jl_engine.py').read_text())
        function = next((node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'select_factor_for_layer'))
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<factor_selector>', 'exec'), namespace)
        select = namespace['select_factor_for_layer']
        for seed in range(20):
            result = select(0, {0: {'A': 0.5, 'B': 0.5}}, {0: {'A': 1, 'B': 1}}, random.Random(seed), 0)
            self.assertEqual(result['factor'], 'B')
            self.assertEqual(result['reason'], 'score_tie_B')
if __name__ == '__main__':
    unittest.main()
