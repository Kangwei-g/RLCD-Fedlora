"""Action indices are not layer counts on the OPT budget grid."""

import unittest

from jl_controller_study import ControllerStudy
from jl_opt_settings import G_VALUES, action_to_budget, budget_to_action


class ActionTests(unittest.TestCase):
    def test_round_trip(self):
        self.assertEqual(len(G_VALUES), 12)
        for action, budget in enumerate(G_VALUES):
            self.assertEqual(action_to_budget(action), budget)
            self.assertEqual(budget_to_action(budget), action)
        for budget in (0, 13, 23, 25):
            with self.assertRaises(ValueError):
                budget_to_action(budget)

    def test_warmup_and_credit(self):
        study = ControllerStudy('ddqn_s01', 0, 4, actions=12, budgets=G_VALUES)
        metrics = dict(macro_f1=0.6, threshold=0.5, balanced_ce=0.7, balanced_brier=0.2)
        g, event = study.finish_round(1, (1, 0.1, 0.7), metrics, 12)
        self.assertIsNone(g)
        self.assertEqual(len(study.agent.memory), 0)
        g, event = study.finish_round(2, (2, 0.1, 0.7), metrics, 12)
        self.assertIn(g, G_VALUES)
        planned = event['next_decision']['action']
        next_g, event = study.finish_round(3, (3, 0.1, 0.7), metrics, g)
        self.assertEqual(study.agent.memory[0][1], planned)
        self.assertEqual(G_VALUES[planned], event['executed_G'])
        last_g, event = study.finish_round(4, (4, 0.1, 0.7), metrics, next_g)
        self.assertIsNone(last_g)
        self.assertEqual(len(study.agent.memory), 2)

    def test_mismatched_action_rejected(self):
        study = ControllerStudy('ddqn_s01', 0, 3, actions=12, budgets=G_VALUES)
        metrics = dict(macro_f1=0.6, threshold=0.5, balanced_ce=0.7, balanced_brier=0.2)
        study.finish_round(1, (1, 0.1, 0.7), metrics, 12)
        g, _ = study.finish_round(2, (2, 0.1, 0.7), metrics, 12)
        wrong_g = 2 if g != 2 else 4
        with self.assertRaises(RuntimeError):
            study.finish_round(3, (3, 0.1, 0.7), metrics, wrong_g)


if __name__ == '__main__':
    unittest.main()
