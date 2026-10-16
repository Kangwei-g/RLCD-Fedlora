# RLCD-FedLoRA

RLCD-FedLoRA is a federated parameter-efficient fine-tuning method that
combines reinforcement learning-based layer coordination with layer-wise
coordinate descent over LoRA factors. This repository contains the Double DQN
resource controller, adaptive layer and factor selection, selected-factor
training and aggregation, experiment configurations, and reproducibility
tests.

## Method

For communication round `t`, RLCD-FedLoRA performs the following steps:

1. A Double DQN observes `(t, q_t, validation_loss_t)` and chooses the next
   active-layer budget `G_(t+1)` from `1, ..., 12`.
2. During the first two rounds, all layers are active. Afterwards, layers are
   selected with an epsilon-greedy policy: random sampling with probability
   `epsilon_L`, otherwise top-`G_t` selection by the stored layer scores.
3. Each selected layer independently chooses LoRA factor `A` or `B`. An unseen
   factor is evaluated first; afterwards selection is epsilon-greedy over the
   layer-specific factor scores.
4. Only the selected factor in each selected layer is trainable and uploaded.
   The backbone and classification head remain frozen.
5. The validation reward is the round-to-round change in server validation
   Macro-F1. The paper-primary configuration assigns equal credit to selected
   layers and updates the selected factor score with an exponential moving
   average.

The implementation uses the paper configuration of per-layer
`adaptive_score` selection with equal reward credit.

## Layout

```text
rlcd_fedlora/
  ddqn.py              Double DQN and replay buffer
  layer_controller.py  Layer-budget realization and layer feedback
  factor_selector.py   Per-layer adaptive A/B selection
  masking.py           Trainable-parameter and upload masks
  aggregation.py       Selected-factor FedAvg and M/K weights
  lora.py              Minimal LoRA linear layer and injection helpers
  client.py            Generic local training and binary evaluation helpers
  server.py            RLCD-FedLoRA controller and round orchestration
configs/               Experiment configurations
data/README.md         Dataset interface specification
examples/              Synthetic end-to-end example
tests/                 Method-level tests
```

## Installation

Python 3.8 or newer is supported.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest
```

The package uses PyTorch and NumPy. RoBERTa integrations also use the
appropriate `transformers` version.

## Quick Check

Run the synthetic end-to-end pipeline:

```bash
python train.py --config configs/rlcd_fedlora_m10_k8.yaml --smoke-test
pytest -q
```

The smoke test creates a small model with 12 LoRA-enabled layers, executes
four federated rounds, verifies that the classifier remains frozen, and checks
that client uploads contain only the selected factors.

## Data Interface

Client train/validation/test splits follow the interface in
[data/README.md](data/README.md). The training application creates the model
and passes a local training callback to `run_federated_round`.

The validation split is used to choose the binary decision threshold and to
produce server feedback. The test split is reserved for final performance
reporting.

## Reproducibility Configurations

- `configs/rlcd_fedlora_m10_k8.yaml`: `M=10`, `K=8`, 200 rounds.
- `configs/rlcd_fedlora_m5_k5.yaml`: `M=5`, `K=5`, 200 rounds.

Both configurations use LoRA rank 10, one local epoch, equal A/B learning
rates, two all-layer warm-up rounds, `epsilon_L=0.2`, `epsilon_F=0.2`, factor
EMA coefficient `0.3`, and a cold-start Double DQN.

## Communication Semantics

The generic round runner mirrors the research implementation: clients receive
the global model state and train and upload the selected LoRA factors. Server
aggregation updates the corresponding factors while preserving all other
global parameters.

## License

RLCD-FedLoRA is distributed under the MIT License. See `LICENSE` and `NOTICE`
for attribution.
