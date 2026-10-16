from examples.synthetic_smoke_test import build_toy_model
from rlcd_fedlora.masking import (
    assert_only_selected_factors_trainable,
    extract_factor_payload,
    set_trainable_factor_map,
)


def test_only_selected_factor_is_trainable_and_uploaded():
    model = build_toy_model(total_layers=2, rank=2)
    factor_map = {0: "B", 1: "A"}
    selected = set_trainable_factor_map(model, factor_map)
    assert len(selected) == 4
    assert_only_selected_factors_trainable(model, factor_map)
    payload = extract_factor_payload(model, factor_map)
    assert set(payload) == set(selected)
    assert all(not name.startswith("classifier.") for name in payload)
