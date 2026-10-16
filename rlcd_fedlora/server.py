"""RLCD-FedLoRA controller and generic federated-round orchestration."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence

from torch import nn

from .aggregation import aggregate_payloads, fedavg_weights
from .ddqn import DoubleDQNAgent
from .factor_selector import FactorDecision, PerLayerFactorSelector
from .layer_controller import LayerController, LayerDecision
from .masking import (
    apply_factor_payload,
    assert_only_selected_factors_trainable,
    clone_lora_state,
    extract_factor_payload,
    selected_factor_delta_norm,
    set_trainable_factor_map,
    validate_factor_payloads,
)


@dataclass(frozen=True)
class RoundPlan:
    round_index: int
    layer_decision: LayerDecision
    factor_map: Dict[int, str]
    factor_metadata: Dict[int, FactorDecision]


@dataclass(frozen=True)
class RoundFeedback:
    reward: float
    state: tuple
    ddqn_loss: Optional[float]
    next_budget: int
    next_layer_decision: LayerDecision
    factor_credit: Dict[int, float]


@dataclass(frozen=True)
class RoundExecution:
    payloads: Sequence[Mapping]
    aggregation_weights: Sequence[float]
    update_norm_q: float


class RLCDFedLoRAController:
    """Stateful implementation of the paper-primary controller."""

    def __init__(
        self,
        total_layers: int = 12,
        warmup_rounds: int = 2,
        layer_random_probability: float = 0.20,
        factor_random_probability: float = 0.20,
        factor_score_beta: float = 0.30,
        seed: int = 0,
        ddqn_kwargs: Optional[Mapping[str, object]] = None,
    ):
        self.total_layers = int(total_layers)
        self.warmup_rounds = int(warmup_rounds)
        self.layer_controller = LayerController(
            total_layers=self.total_layers,
            random_probability=layer_random_probability,
            seed=int(seed),
        )
        self.factor_selector = PerLayerFactorSelector(
            total_layers=self.total_layers,
            random_probability=factor_random_probability,
            score_beta=factor_score_beta,
            seed=int(seed) + 7919,
        )
        agent_options = dict(ddqn_kwargs or {})
        agent_options.setdefault("state_size", 3)
        agent_options.setdefault("action_size", self.total_layers)
        agent_options.setdefault("seed", int(seed))
        self.agent = DoubleDQNAgent(**agent_options)

        self._previous_validation_macro_f1 = None  # type: Optional[float]
        self._previous_state = None  # type: Optional[tuple]
        self._previous_action = None  # type: Optional[int]
        self._pending_layer_decision = self.layer_controller.all_layers()
        self._active_plan = None  # type: Optional[RoundPlan]
        self._next_round_index = 0

    def plan_round(self, round_index: int) -> RoundPlan:
        round_index = int(round_index)
        if round_index != self._next_round_index:
            raise ValueError(
                "Expected round {}, received {}".format(
                    self._next_round_index, round_index
                )
            )
        if self._active_plan is not None:
            raise RuntimeError("The active round must be observed before planning again")

        if round_index < self.warmup_rounds:
            layer_decision = self.layer_controller.all_layers()
        else:
            layer_decision = self._pending_layer_decision
        factor_map, metadata = self.factor_selector.select(layer_decision.layers)
        plan = RoundPlan(round_index, layer_decision, factor_map, metadata)
        self._active_plan = plan
        return plan

    def observe_round(
        self,
        round_index: int,
        update_norm_q: float,
        validation_loss: float,
        validation_macro_f1: float,
    ) -> RoundFeedback:
        if self._active_plan is None or self._active_plan.round_index != int(round_index):
            raise RuntimeError("No matching active round plan")
        plan = self._active_plan
        has_previous_metric = self._previous_validation_macro_f1 is not None
        reward = (
            float(validation_macro_f1) - self._previous_validation_macro_f1
            if has_previous_metric
            else 0.0
        )

        if has_previous_metric:
            self.layer_controller.observe(plan.layer_decision.layers, reward)
        factor_credit = self.factor_selector.observe(plan.factor_map, reward)

        state = (
            float(int(round_index) + 1),
            float(update_norm_q),
            float(validation_loss),
        )
        ddqn_loss = None
        if self._previous_state is not None and self._previous_action is not None:
            self.agent.remember(
                self._previous_state,
                self._previous_action,
                reward,
                state,
            )
            ddqn_loss = self.agent.replay()

        action = self.agent.act(state)
        next_budget = int(action) + 1
        next_layer_decision = self.layer_controller.select(next_budget)
        self._pending_layer_decision = next_layer_decision
        self._previous_state = state
        self._previous_action = action
        self._previous_validation_macro_f1 = float(validation_macro_f1)
        self._active_plan = None
        self._next_round_index += 1

        return RoundFeedback(
            reward=reward,
            state=state,
            ddqn_loss=ddqn_loss,
            next_budget=next_budget,
            next_layer_decision=next_layer_decision,
            factor_credit=factor_credit,
        )

    def state_dict(self) -> Dict[str, object]:
        if self._active_plan is not None:
            raise RuntimeError("Checkpoint only between completed rounds")
        return {
            "total_layers": self.total_layers,
            "warmup_rounds": self.warmup_rounds,
            "layer_controller": self.layer_controller.state_dict(),
            "factor_selector": self.factor_selector.state_dict(),
            "ddqn": self.agent.state_dict(),
            "previous_validation_macro_f1": self._previous_validation_macro_f1,
            "previous_state": self._previous_state,
            "previous_action": self._previous_action,
            "pending_layer_decision": {
                "budget": self._pending_layer_decision.budget,
                "layers": self._pending_layer_decision.layers,
                "reason": self._pending_layer_decision.reason,
                "random_draw": self._pending_layer_decision.random_draw,
            },
            "next_round_index": self._next_round_index,
        }


ClientTrainFn = Callable[[int, nn.Module, Mapping[int, str]], None]


def run_federated_round(
    global_model: nn.Module,
    plan: RoundPlan,
    selected_client_ids: Sequence[int],
    client_train_sizes: Mapping[int, int],
    client_train_fn: ClientTrainFn,
    total_clients: int,
    total_train_examples: int,
    model_name: str = "roberta",
    use_mk_correction: bool = True,
) -> RoundExecution:
    """Run one round with a full-state downlink and selected-factor uplinks."""
    if not selected_client_ids:
        raise ValueError("At least one client must participate")
    state_before = clone_lora_state(global_model)
    payloads = []

    for client_id in selected_client_ids:
        client_model = deepcopy(global_model)
        set_trainable_factor_map(
            client_model, plan.factor_map, model_name=model_name
        )
        assert_only_selected_factors_trainable(
            client_model, plan.factor_map, model_name=model_name
        )
        client_train_fn(int(client_id), client_model, plan.factor_map)
        payloads.append(
            extract_factor_payload(
                client_model, plan.factor_map, model_name=model_name
            )
        )

    validate_factor_payloads(payloads, plan.factor_map, model_name=model_name)
    selected_sizes = [client_train_sizes[int(client_id)] for client_id in selected_client_ids]
    weights = fedavg_weights(
        selected_sizes,
        total_clients=total_clients,
        total_examples=total_train_examples,
        use_mk_correction=use_mk_correction,
    )
    aggregate = aggregate_payloads(payloads, weights)
    apply_factor_payload(global_model, aggregate)
    state_after = clone_lora_state(global_model)
    update_norm_q = selected_factor_delta_norm(
        state_before,
        state_after,
        plan.factor_map,
        model_name=model_name,
    )
    return RoundExecution(payloads, weights, update_norm_q)
