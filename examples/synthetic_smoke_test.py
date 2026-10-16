"""Exercise RLCD-FedLoRA without clinical data or a pretrained model."""

from typing import Mapping

import torch
from torch import nn

from rlcd_fedlora import (
    RLCDFedLoRAController,
    inject_transformer_lora,
    run_federated_round,
)


class ToySelfAttention(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)


class ToyAttention(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        setattr(self, "self", ToySelfAttention(width))


class ToyLayer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.attention = ToyAttention(width)


class ToyEncoder(nn.Module):
    def __init__(self, total_layers: int, width: int):
        super().__init__()
        self.layer = nn.ModuleList(
            [ToyLayer(width) for _ in range(total_layers)]
        )


class ToyRoberta(nn.Module):
    def __init__(self, total_layers: int, width: int):
        super().__init__()
        self.encoder = ToyEncoder(total_layers, width)


class ToyModel(nn.Module):
    def __init__(self, total_layers: int = 12, width: int = 4):
        super().__init__()
        self.roberta = ToyRoberta(total_layers, width)
        self.classifier = nn.Linear(width, 2)


def build_toy_model(total_layers: int = 12, rank: int = 2) -> ToyModel:
    model = ToyModel(total_layers=total_layers)
    return inject_transformer_lora(
        model,
        layers=range(total_layers),
        rank=rank,
        alpha=8.0,
        model_name="roberta",
    )


def _ddqn_options(config: Mapping):
    ddqn = config["ddqn"]
    return {
        "state_size": int(ddqn["state_size"]),
        "action_size": int(ddqn["action_size"]),
        "hidden_size": int(ddqn["hidden_size"]),
        "learning_rate": float(ddqn["learning_rate"]),
        "gamma": float(ddqn["gamma"]),
        "epsilon_start": float(ddqn["epsilon_start"]),
        "epsilon_end": float(ddqn["epsilon_end"]),
        "epsilon_decay": float(ddqn["epsilon_decay"]),
        "batch_size": int(ddqn["batch_size"]),
        "memory_size": int(ddqn["replay_buffer_size"]),
        "tau": float(ddqn["target_tau"]),
    }


def run_smoke_test(config: Mapping) -> None:
    experiment = config["experiment"]
    model_config = config["model"]
    controller_config = config["controller"]
    total_layers = int(model_config["total_layers"])
    num_clients = int(experiment["num_clients"])
    clients_per_round = int(experiment["clients_per_round"])
    seed = int(experiment["seed"])

    torch.manual_seed(seed)
    model = build_toy_model(
        total_layers=total_layers,
        rank=min(2, int(model_config["lora_rank"])),
    )
    classifier_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if name.startswith("classifier.")
    }
    controller = RLCDFedLoRAController(
        total_layers=total_layers,
        warmup_rounds=int(controller_config["warmup_rounds"]),
        layer_random_probability=float(
            controller_config["layer_random_probability"]
        ),
        factor_random_probability=float(
            controller_config["factor_random_probability"]
        ),
        factor_score_beta=float(controller_config["factor_score_beta"]),
        seed=seed,
        ddqn_kwargs=_ddqn_options(config),
    )
    client_sizes = {client_id: 100 for client_id in range(num_clients)}
    selected_clients = list(range(clients_per_round))

    for round_index in range(4):
        plan = controller.plan_round(round_index)

        def synthetic_local_update(client_id, client_model, factor_map):
            del factor_map
            scale = 1e-3 * (round_index + 1) * (client_id + 1)
            with torch.no_grad():
                for parameter in client_model.parameters():
                    if parameter.requires_grad:
                        parameter.add_(scale)

        execution = run_federated_round(
            global_model=model,
            plan=plan,
            selected_client_ids=selected_clients,
            client_train_sizes=client_sizes,
            client_train_fn=synthetic_local_update,
            total_clients=num_clients,
            total_train_examples=sum(client_sizes.values()),
            model_name="roberta",
            use_mk_correction=bool(experiment["use_mk_aggregation"]),
        )
        feedback = controller.observe_round(
            round_index=round_index,
            update_norm_q=execution.update_norm_q,
            validation_loss=0.70 - 0.01 * round_index,
            validation_macro_f1=0.50 + 0.01 * round_index,
        )
        print(
            "round={:02d} layers={} A={} B={} q={:.6f} reward={:.4f} next_budget={}".format(
                round_index + 1,
                len(plan.layer_decision.layers),
                sum(factor == "A" for factor in plan.factor_map.values()),
                sum(factor == "B" for factor in plan.factor_map.values()),
                execution.update_norm_q,
                feedback.reward,
                feedback.next_budget,
            )
        )

    for name, value in classifier_before.items():
        if not torch.equal(value, model.state_dict()[name]):
            raise AssertionError("Classification head changed during the smoke test")
    print("RLCD-FedLoRA synthetic smoke test passed.")


if __name__ == "__main__":
    run_smoke_test(
        {
            "experiment": {
                "seed": 0,
                "num_clients": 10,
                "clients_per_round": 8,
                "use_mk_aggregation": True,
            },
            "model": {"total_layers": 12, "lora_rank": 10},
            "controller": {
                "warmup_rounds": 2,
                "layer_random_probability": 0.20,
                "factor_random_probability": 0.20,
                "factor_score_beta": 0.30,
            },
            "ddqn": {
                "state_size": 3,
                "action_size": 12,
                "hidden_size": 64,
                "learning_rate": 0.001,
                "gamma": 0.99,
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 0.995,
                "batch_size": 64,
                "replay_buffer_size": 5000,
                "target_tau": 0.05,
            },
        }
    )
