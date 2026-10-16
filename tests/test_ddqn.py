from rlcd_fedlora.ddqn import DoubleDQNAgent


def test_double_dqn_action_and_replay():
    agent = DoubleDQNAgent(
        state_size=3,
        action_size=12,
        batch_size=2,
        epsilon_start=1.0,
        seed=7,
    )
    action = agent.act((1.0, 0.2, 0.7))
    assert 0 <= action < 12
    agent.remember((1.0, 0.2, 0.7), action, 0.01, (2.0, 0.3, 0.6))
    loss = agent.replay()
    assert loss is not None
    assert loss >= 0.0
    assert agent.epsilon < 1.0
