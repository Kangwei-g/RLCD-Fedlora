import json
import ast
import random
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from jl_controller_variants import VARIANTS, ORDER, get_spec, client_shuffle_seed
from jl_controller_study import (ControllerStudy, StudyDDQN, LinearBandit, encode_state,
                                 epsilon_at, reward_potential)


class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_registry_and_cli(self):
        from jl_run import MAIN_METHODS, parser, engine_arguments
        self.assertEqual(len(ORDER), 16)
        self.assertEqual([sum(s['kind'] == k for s in VARIANTS.values())
                          for k in ('ddqn', 'ucb', 'ts')], [10, 3, 3])
        self.assertEqual(len(MAIN_METHODS), 9)
        for name in ORDER:
            args = parser().parse_args(['--method', name])
            argv = engine_arguments(args, Path('/tmp/test'))
            self.assertEqual(argv[argv.index('--jl_controller_variant') + 1], name)
            self.assertEqual(float(argv[argv.index('--factor_epsilon') + 1]),
                             VARIANTS[name]['coordinate_exploration'])

    def test_state_and_schedule(self):
        self.assertEqual(encode_state((200, 4, 1), 200, True), [1, .8, .5])
        self.assertEqual(encode_state((200, 4, 1), 200, False), [200, 4, 1])
        with self.assertRaises(ValueError):
            encode_state((2, float('nan'), 1), 200, True)
        self.assertEqual(epsilon_at(2, True), .5)
        self.assertAlmostEqual(epsilon_at(80, True), .05)
        self.assertEqual(epsilon_at(200, True), .05)
        self.assertGreater(epsilon_at(200, False), .36)

    def test_rewards(self):
        a = dict(macro_f1=.4, balanced_ce=1, balanced_brier=.3)
        b = dict(macro_f1=.6, balanced_ce=.5, balanced_brier=.2)
        for mode in {s['reward'] for s in VARIANTS.values()}:
            self.assertLess(reward_potential(a, mode), reward_potential(b, mode))
            self.assertTrue(0 <= reward_potential(b, mode) <= 1)

    def test_private_rng(self):
        torch.manual_seed(19)
        before = torch.get_rng_state().clone()
        a = StudyDDQN(get_spec('ddqn_s01'), 0, 12)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        b = StudyDDQN(get_spec('ddqn_s01'), 0, 12)
        self.assertEqual([a.choose([2, 1, 1], t)[0] for t in range(2, 30)],
                         [b.choose([2, 1, 1], t)[0] for t in range(2, 30)])
        seeds = [client_shuffle_seed(0, r, c) for r in range(200) for c in range(5)]
        self.assertEqual(len(set(seeds)), 1000)
        def order(c):
            return torch.randperm(20, generator=torch.Generator().manual_seed(client_shuffle_seed(0, 5, c)))
        first = {c: order(c) for c in range(5)}
        second = {c: order(c) for c in reversed(range(5))}
        self.assertTrue(all(torch.equal(first[c], second[c]) for c in range(5)))

    def test_terminal_target_and_bounds(self):
        agent = StudyDDQN(get_spec('ddqn_s06'), 0, 12)
        with torch.no_grad():
            for p in agent.target.parameters():
                p.fill_(1)
        info = agent.observe([.1, .3, .4], 2, -.2, [.2, .3, .4], True)
        self.assertAlmostEqual(info['target_max'], -.2, places=6)
        values = agent.q(torch.ones(4, 3) * 200).detach()
        self.assertTrue(torch.all(values.abs() <= 1))

    def test_replay_warmup_and_updates(self):
        agent = StudyDDQN(get_spec('ddqn_s08'), 0, 12)
        for _ in range(31):
            result = agent.observe([.1, .3, .4], 1, .01, [.2, .4, .5], False)
            self.assertEqual(result['update_count'], 0)
        result = agent.observe([.1, .3, .4], 1, .01, [.2, .4, .5], False)
        self.assertEqual(result['update_count'], 4)
        self.assertEqual(len(result['losses']), 4)

    def test_bandit_statistics_and_coverage(self):
        for name in [n for n in ORDER if not n.startswith('ddqn')]:
            a, b = [LinearBandit(get_spec(name), 0, 12) for _ in range(2)]
            actions = []
            for t in range(12):
                action, _ = a.choose([.2, .3, .4], t)
                self.assertEqual(action, b.choose([.2, .3, .4], t)[0])
                actions.append(action)
                a.observe([.2, .3, .4], action, .1, [], False)
                b.observe([.2, .3, .4], action, .1, [], False)
            self.assertEqual(len(set(actions)), 12)
            self.assertTrue(np.all(np.linalg.eigvalsh(a.A) > 0))
        a = LinearBandit(get_spec('ucb_s01'), 0, 12)
        x = np.array([1, .2, .3, .4])
        a.observe(x[1:], 4, -.2, [], False)
        np.testing.assert_allclose(a.A[4], np.eye(4) + np.outer(x, x))
        np.testing.assert_allclose(a.b[4], -.2 * x)

    def test_round_alignment_and_safe_snapshots(self):
        for name in ORDER:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                controller = ControllerStudy(name, 0, 4, directory)
                g = 12
                for t in range(1, 5):
                    metrics = dict(macro_f1=.4 + t / 100, balanced_ce=1 - t / 100,
                                   balanced_brier=.3 - t / 100, threshold=.5)
                    next_g, event = controller.finish_round(t, (t, .3, .5), metrics, g)
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
        controller = ControllerStudy('ucb_s01', 0, 4)
        metrics = dict(macro_f1=.5, balanced_ce=.7, balanced_brier=.2, threshold=.5)
        controller.finish_round(1, (1, 0, .7), metrics, 12)
        g, _ = controller.finish_round(2, (2, 0, .7), metrics, 12)
        with self.assertRaises(RuntimeError):
            controller.finish_round(3, (3, 0, .7), metrics, g % 12 + 1)

    def test_zero_exploration_has_deterministic_factor_ties(self):
        source = ast.parse(Path(__file__).with_name('jl_engine.py').read_text())
        function = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'select_factor_for_layer')
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<factor_selector>', 'exec'), namespace)
        select = namespace['select_factor_for_layer']
        for seed in range(20):
            result = select(0, {0: {'A': .5, 'B': .5}}, {0: {'A': 1, 'B': 1}}, random.Random(seed), 0)
            self.assertEqual(result['factor'], 'B')
            self.assertEqual(result['reason'], 'score_tie_B')


if __name__ == '__main__':
    unittest.main()
