from rlcd_fedlora.factor_selector import PerLayerFactorSelector


def test_unseen_coverage_then_score_selection():
    selector = PerLayerFactorSelector(
        total_layers=2,
        random_probability=0.0,
        score_beta=1.0,
        seed=0,
    )
    first, _ = selector.select([0, 1])
    assert first == {0: "B", 1: "B"}
    selector.observe(first, reward=0.2)

    second, _ = selector.select([0, 1])
    assert second == {0: "A", 1: "A"}
    selector.observe(second, reward=-0.2)

    third, _ = selector.select([0, 1])
    assert third == {0: "B", 1: "B"}


def test_equal_credit_sums_to_reward():
    selector = PerLayerFactorSelector(total_layers=3, seed=0)
    factor_map, _ = selector.select([0, 1, 2])
    credits = selector.observe(factor_map, reward=0.3)
    assert abs(sum(credits.values()) - 0.3) < 1e-12
