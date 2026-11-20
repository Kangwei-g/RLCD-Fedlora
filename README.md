# RLCD-FedLoRA

Reinforcement Learning and Contribution Driven Federated Low-Rank Adaptation.
This release includes runnable RoBERTa and OPT experiments, hybrid baselines,
component ablations, private-data preparation, and seed-level reporting.

## Installation

Use Python 3.12 and a PyTorch build appropriate for your CPU/CUDA environment.
The model integration uses Transformers 4.46.3.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-experiments.txt
pip install -e .
```

The original lightweight `rlcd_fedlora` API and `train.py --smoke-test` remain
available. Complete experiments use `run_experiments.py` and the isolated
implementations in `experiments/roberta` and `experiments/opt`. This separation
prevents OPT-specific changes from altering the RoBERTa path. Run commands
below from the repository root.

## Models

Weights are not included. Obtain them from the official model pages and
observe each model's license:

- [RoBERTa-base](https://huggingface.co/FacebookAI/roberta-base): encoder backbone.
- [OPT-125M](https://huggingface.co/facebook/opt-125m): decoder backbone.

Both integrations coordinate 12 blocks and apply LoRA to key and value
projections. For LoRA methods, the pretrained backbone and seed-initialized
classification head are frozen. FedAvg-Full is a separate control that trains
all model parameters.

Optional download helper (records the resolved revision and file checksums):

```bash
python download_models.py --backbone roberta --output-dir models/roberta
python download_models.py --backbone opt --output-dir models/opt
```

OPT is pinned to `27dcfa74d334bc871f3234de431e71c6eeba5dd6`. Its preflight
checks the download manifest and tokenizer/cache identity. Training does not
automatically fetch a missing OPT checkpoint.

## Data Preparation

No MIMIC records, patient identifiers, token caches, predictions, or trained
weights are distributed. Users must obtain authorized MIMIC-III access.
See [preprocessing/README.md](preprocessing/README.md) for cohort extraction,
the input contract, partitioning, overlap checks, and tokenization.

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

Never reuse a RoBERTa token cache for OPT. OPT preparation preserves examples,
labels, ordering, and splits while changing tokenization. Pickle caches are
trusted local artifacts: never load an untrusted pickle.

## Configurations and Seeds

| Backbone | Configuration | Rounds | Seeds |
| --- | --- | --- | --- |
| RoBERTa | Primary: 5 clients, all 5 participating | 200 | 0, 4, 5 |
| RoBERTa | Supplementary: 10 clients, 8 participating | 200 | 0, 4, 5 |
| OPT-125M | Primary: 5 clients, all 5 participating | 100 | 0, 4, 5 |

Each plan has one run per method/configuration/seed. The partition seed is
separate and fixed across model seeds. Existing run directories are never
overwritten. All methods use the same splits within a comparison.

Standard LoRA settings: rank 10, alpha 8, Adam learning rate 5e-4, batch size
64, and one local epoch. FlexLoRA + CD adapts a common rank within 6..10.
FedAvg-Full uses batch size 32 and learning rate 2e-5.

| Method ID | Implementation |
| --- | --- |
| `d01` | RLCD-FedLoRA: D01 DDQN, contribution-based layers and factors |
| `flexlora_cd` | All 12 layers; validation-controlled common rank and contribution-based factors |
| `spdcfl_cd` | Progressive layer-budget reduction proxy and contribution-based factors |
| `fedfmsl_cd` | Progressive SAL layer-selection proxy and contribution-based factors |
| `rl_ffa_lora` | RL layers; freeze A and train/upload B |
| `rl_fedavg_lora` | RL layers; train both factors, aggregate their deltas separately |
| `rl_rolora` | RL layers; alternate the shared factor instruction B/A |
| `rl_flora` | RL layers; aggregate product deltas and project to the fixed rank by SVD |
| `rl_aslora` | RL layers; adaptive AS-LoRA factor selector |
| `fixed_g12` | Fix G=12; contribution-based factor selection |
| `random_g` | Uniform G in 1..12 after two warmup rounds |
| `random_layers` | Retain D01 G; uniformly select layers without replacement |
| `random_factors` | Retain D01 G/layers; random A/B, shared across clients |
| `local_only` | Independent all-layer LoRA training per client |
| `centralized` | Pool training, validation, and test splits separately |
| `fedavg_full` | Train and aggregate all model parameters |

These are hybrid controls within a common experimental framework, not
unmodified upstream baseline repositories. FlexLoRA uses a common rank rather
than heterogeneous client ranks; SPD-CFL and FedFMSL use the stated layer
proxies; FLoRA uses fixed-rank product-SVD rather than unbounded stacking.
Their rules are in `jl_rank.py`, `jl_layer_selection.py`, `jl_baseline_hooks.py`,
and `arc_aslora.py` inside the RoBERTa implementation. OPT supports D01,
FlexLoRA + CD, RL + RoLoRA, and RL + FedAvg-LoRA.

## Run

Preview without loading data or models:

```bash
python run_experiments.py --backbone roberta \
  --data-file private_data/prepared/roberta.pkl --dry-run
```

Primary comparison, component ablations, and decoder comparison:

```bash
python run_experiments.py --backbone roberta --settings 5_5 \
  --methods d01 flexlora_cd spdcfl_cd fedfmsl_cd rl_ffa_lora \
    rl_fedavg_lora rl_rolora rl_flora rl_aslora \
  --data-file private_data/prepared/roberta.pkl --output-root runs/primary

python run_experiments.py --backbone roberta --settings 5_5 10_8 \
  --methods d01 fixed_g12 random_g random_layers random_factors \
  --data-file private_data/prepared/roberta.pkl --output-root runs/ablations

python run_experiments.py --backbone opt \
  --data-file private_data/prepared/opt.pkl --output-root runs/decoder
```

`--seeds 0 4 5` is the default. Use `--settings 10_8` for the supplementary
comparison, or `--methods d01 local_only centralized` for the training-paradigm
comparison. The complete RoBERTa plan contains all 16 methods under both
settings; `--methods` and `--settings` select the desired subset.

Jobs run sequentially on `--device cuda:0`. `--workers 4` enables four
concurrent clients within an experiment; serial clients are the default.
Parallel execution and GPU kernels may affect numerical reproducibility.
BF16 training, SDPA attention, and disabled dropout are explicit release
defaults. Validation, feedback, and final testing use FP32. CPU runs use
`--precision fp32 --attention eager --device cpu`. OPT microbatch equals its
logical batch size (64). No server credentials or automatic GPU queue are needed.

`--resume` skips completed runs only for an identical saved plan. It does not
resume interrupted checkpoints or overwrite failed outputs. Preserve an
incomplete run before using a new output directory.

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

Runs record configuration, seed, dependency versions, model/data/code hashes,
selected checkpoint/threshold, final metrics and predictions. The validation
trajectory and G are in `jl_round_metrics.csv`; D01 also records
`jl_controller_trajectory.jsonl`. Generated records remain private by default.

```bash
python summarize_results.py runs/decoder/opt/experiment_plan.json \
  --output outputs/decoder_summary.csv

python plot_trajectories.py \
  runs/decoder/opt/5_5/ddqn_s01/seed0_standard/jl_round_metrics.csv \
  --output outputs/decoder_trajectory.pdf
```

The collector uses every configured seed equally, reporting means and sample
standard deviations. It rejects duplicate identities, missing seeds, and
incomplete runs; it does not select runs using test scores. DV uses pooled
confusion counts, loss per 10,000 cases with FN/FP ratio 7/3, and scenario
coefficient 0.574. Its zero reference is the largest mean loss among the
methods in that comparison. Choose table membership with `--methods`; the
reference loss is exported because DV changes with comparison membership.

## Tests

No MIMIC access, model download, or GPU is required:

```bash
python -m pytest -q
python smoke_test.py --output-root /tmp/rlcd-smoke
```

CI additionally checks the validation protocol, rank changes, centralized
pooling, controller, factor masks, and model-specific preprocessing. Tiny
random models test execution and invariants, not predictive performance.

## License

Code is distributed under MIT; see `LICENSE` and `NOTICE`. Model and data
licenses are separate. This is research software, not a clinical tool.
