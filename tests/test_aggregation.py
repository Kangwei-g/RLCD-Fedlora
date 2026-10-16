import torch

from rlcd_fedlora.aggregation import aggregate_payloads, fedavg_weights


def test_standard_fedavg_payload():
    payloads = [
        {"layer.lora_b": torch.tensor([1.0, 3.0])},
        {"layer.lora_b": torch.tensor([3.0, 5.0])},
    ]
    weights = fedavg_weights([1, 3])
    result = aggregate_payloads(payloads, weights)
    assert torch.allclose(result["layer.lora_b"], torch.tensor([2.5, 4.5]))


def test_mk_weights_for_equal_client_sizes_sum_to_one():
    weights = fedavg_weights(
        [100] * 8,
        total_clients=10,
        total_examples=1000,
        use_mk_correction=True,
    )
    assert abs(sum(weights) - 1.0) < 1e-12
