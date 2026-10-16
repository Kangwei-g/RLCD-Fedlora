from rlcd_fedlora.server import RLCDFedLoRAController


def test_two_round_all_layer_warmup_and_factor_coverage():
    controller = RLCDFedLoRAController(
        total_layers=4,
        warmup_rounds=2,
        layer_random_probability=0.0,
        factor_random_probability=0.0,
        seed=0,
        ddqn_kwargs={"batch_size": 2},
    )
    first = controller.plan_round(0)
    assert first.layer_decision.layers == [0, 1, 2, 3]
    assert set(first.factor_map.values()) == {"B"}
    controller.observe_round(0, 0.1, 0.7, 0.5)

    second = controller.plan_round(1)
    assert second.layer_decision.layers == [0, 1, 2, 3]
    assert set(second.factor_map.values()) == {"A"}
    feedback = controller.observe_round(1, 0.2, 0.6, 0.52)
    assert 1 <= feedback.next_budget <= 4

    third = controller.plan_round(2)
    assert len(third.layer_decision.layers) == feedback.next_budget
