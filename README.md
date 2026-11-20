# RLCD-FedLoRA

Reinforcement Learning and Contribution Driven Federated Low-Rank Adaptation
for one-year post-discharge mortality prediction from clinical text.
Includes RoBERTa-base and OPT-125M experiments, baselines, and ablations.

## Installation

Use Python 3.12 and a PyTorch build appropriate for your CPU/CUDA environment.
The model integration uses Transformers 4.46.3.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-experiments.txt
pip install -e .
```

Run the commands below from the repository root.

## Models

Download model weights from the official pages:

- [RoBERTa-base](https://huggingface.co/FacebookAI/roberta-base): encoder backbone.
- [OPT-125M](https://huggingface.co/facebook/opt-125m): decoder backbone.

Both integrations coordinate 12 blocks and apply LoRA to key and value
projections. The pretrained backbone and seed-initialized classification
head are frozen.

The download helper records model revisions and checksums:

```bash
python download_models.py --backbone roberta --output-dir models/roberta
python download_models.py --backbone opt --output-dir models/opt
```

OPT is pinned to `27dcfa74d334bc871f3234de431e71c6eeba5dd6`.
Preflight checks verify its model manifest and tokenizer/cache identity.

## Data Preparation

MIMIC-III data require authorized access and are not included.
See [preprocessing/README.md](preprocessing/README.md) for cohort extraction,
partitioning, overlap checks, and tokenization.

```bash
python preprocessing/extract_cohort.py \
  --mimic-dir /private/path/mimic-iii --output private_data/cohort.csv

python preprocessing/prepare_data.py \
  --input-file private_data/cohort.csv --roberta-model models/roberta \
  --output-dir private_data/prepared

python experiments/opt/jl_opt_prepare_data.py \
  --model models/opt --input-file private_data/prepared/raw.pkl \
  --output-file private_data/prepared/opt.pkl
```

Prepare a separate token cache for each backbone. OPT preparation preserves
examples, labels, ordering, and splits while changing tokenization.

## Experiment Settings

All experiments use **200 communication rounds per run** and three different
random seeds, with one run per method/configuration/seed. Results are reported
as means and sample standard deviations. Data splits are fixed across methods
and model seeds; the partition seed is separate from the model seeds.

| Backbone | Configuration |
| --- | --- |
| RoBERTa | Primary: 5 clients, all 5 participating |
| RoBERTa | Supplementary: 10 clients, 8 participating |
| OPT-125M | Primary: 5 clients, all 5 participating |

Standard LoRA settings: rank 10, alpha 8, Adam learning rate 5e-4, batch size
64, and one local epoch. FlexLoRA + CD adapts a common rank within 6..10.

| Method ID | Implementation |
| --- | --- |
| `d01` | RLCD-FedLoRA: DDQN layer count, contribution-based layers and factors |
| `flexlora_cd` | All 12 layers; validation-controlled common rank and contribution-based factors |
| `spdcfl_cd` | Progressive layer-budget reduction proxy and contribution-based factors |
| `fedfmsl_cd` | Progressive SAL layer-selection proxy and contribution-based factors |
| `rl_ffa_lora` | RL layers; freeze A and train/upload B |
| `rl_fedavg_lora` | RL layers; train both factors, aggregate their deltas separately |
| `rl_rolora` | RL layers; alternate the shared factor instruction B/A |
| `rl_flora` | RL layers; fixed-rank SVD of aggregated product deltas |
| `rl_aslora` | RL layers; adaptive AS-LoRA factor selector |
| `fixed_g12` | Fix G=12; contribution-based factor selection |
| `random_g` | Uniform G in 1..12 after two warmup rounds |
| `random_layers` | Retain D01 G; uniformly select layers without replacement |
| `random_factors` | Retain D01 G/layers; random A/B, shared across clients |
| `local_only` | Independent all-layer LoRA training per client |
| `centralized` | Pool training, validation, and test splits separately |

Baseline adaptations share the training and evaluation pipeline. Their rules
are in `jl_rank.py`, `jl_layer_selection.py`, `jl_baseline_hooks.py`, and
`arc_aslora.py` under `experiments/roberta`. OPT supports RLCD-FedLoRA,
FlexLoRA + CD, RL + RoLoRA, and RL + FedAvg-LoRA.

## Run

Preview without loading data or models:

```bash
python run_experiments.py --backbone roberta \
  --data-file private_data/prepared/roberta.pkl --dry-run
```

The commands below reproduce the five experiments in the paper:

| Experiment | Backbone | Settings | Methods |
| --- | --- | --- | --- |
| 1 | RoBERTa | `5_5` | RLCD-FedLoRA and eight baselines |
| 2 | RoBERTa | `10_8` | Same methods as Experiment 1 |
| 3 | RoBERTa | Both | RLCD-FedLoRA and four ablations |
| 4 | RoBERTa | `5_5` | RLCD-FedLoRA, local-only, centralized |
| 5 | OPT-125M | `5_5` | RLCD-FedLoRA, FlexLoRA + CD, RL + RoLoRA, RL + FedAvg-LoRA |

```bash
python run_experiments.py --backbone roberta --settings 5_5 10_8 \
  --methods d01 flexlora_cd spdcfl_cd fedfmsl_cd rl_ffa_lora \
    rl_fedavg_lora rl_rolora rl_flora rl_aslora \
  --data-file private_data/prepared/roberta.pkl --output-root runs/primary

python run_experiments.py --backbone roberta --settings 5_5 10_8 \
  --methods d01 fixed_g12 random_g random_layers random_factors \
  --data-file private_data/prepared/roberta.pkl --output-root runs/ablations

python run_experiments.py --backbone roberta --settings 5_5 \
  --methods d01 local_only centralized \
  --data-file private_data/prepared/roberta.pkl --output-root runs/paradigms

python run_experiments.py --backbone opt \
  --data-file private_data/prepared/opt.pkl --output-root runs/decoder
```

The default plans contain 84 RoBERTa runs and 12 OPT runs. Local-only and
centralized training use the primary setting. RLCD-FedLoRA runs can be reused
across tables by filtering the same plan with `summarize_results.py --methods`.

Defaults are BF16 training, SDPA attention, disabled dropout, and `cuda:0`.
Validation, feedback, and final testing use FP32. For CPU runs, add
`--precision fp32 --attention eager --device cpu`. OPT uses batch and
microbatch size 64. Jobs and clients run sequentially; `--workers 4` enables
four concurrent clients within each experiment.

`--resume` skips completed runs from an identical saved plan. Use a new output
directory for interrupted runs.

## Evaluation

1. The dedicated server validation split supplies adaptive training feedback.
2. After aggregation, the global model is evaluated on all client validation
   splits, including clients not sampled for training that round.
3. A shared threshold in 0.05..0.95 (step 0.01) maximizes equal-client mean
   validation Macro-F1. The best validation round supplies the checkpoint and
   threshold for final testing. Ties favor the smallest threshold and earliest round.
4. Predictive metrics are equal-client means; AUROC uses probabilities.
   Centralized training uses pooled validation and testing. Local-only clients
   select their own validation checkpoints and thresholds.
5. Partial participation uses sampling-corrected weights on deltas:
   `global_new = global_old + sum_k omega_k * (local_k - global_old)`, where
   `omega_k = (M/K) * n_k / sum_all_clients(n)`. Unchanged parameters are retained.

## Outputs

Each run saves its configuration, provenance, selected checkpoint/threshold,
final metrics, and predictions. Validation metrics and G are in
`jl_round_metrics.csv`; RLCD-FedLoRA also saves `jl_controller_trajectory.jsonl`.

```bash
python summarize_results.py runs/decoder/opt/experiment_plan.json \
  --output outputs/decoder_summary.csv
```

The collector requires one completed run for every configured seed.
DV uses pooled confusion counts, loss per 10,000 cases, FN/FP ratio 7/3,
and scenario coefficient 0.574. Its zero reference is the largest mean loss
among the methods selected with `--methods`; the reference loss is exported.

## Tests

Tests use synthetic data and tiny random models; no MIMIC access, model
download, or GPU is required:

```bash
python -m pytest -q
python smoke_test.py --output-root /tmp/rlcd-smoke
```

CI also checks validation, rank changes, centralized pooling, controllers,
factor masks, and model-specific preprocessing.

## License

Code is distributed under MIT; see `LICENSE` and `NOTICE`.
Model and data licenses are separate.
