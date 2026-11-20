# Source Release: 2026-11-20

## Included

- Isolated RoBERTa and OPT-125M training integrations.
- RLCD-FedLoRA D01, eight RoBERTa hybrid baselines, four component ablations,
  and local-only, centralized, and full-parameter FedAvg controls.
- OPT D01, FlexLoRA + CD, RL + RoLoRA, and RL + FedAvg-LoRA.
- Standard plans with seeds 0, 4, and 5; one job per method/setting/seed.
- Validation-selected global checkpoints, shared thresholds, and final testing.
- Centralized pooling of training, validation, and test splits separately.
- Delta aggregation for sampling-corrected client weights.
- Private cohort extraction, split preparation, overlap auditing, and tokenization.
- Official model links and an optional explicit download/checksum helper.
- Seed-level metric summaries, validation/G plots, and offline synthetic tests.

The baseline table in README documents the implemented hybrid adaptations.
The original lightweight API remains available separately from the complete
experiment integrations. There is no server-specific job scheduler.

## Verification

Local CPU checks completed on 2026-11-20:

| Check | Result |
| --- | --- |
| Core, public entry points, reporting, and preprocessing | 14 tests passed |
| RoBERTa protocol, rank, controller, controls, and ablations | 41 tests passed; 80 subtests passed |
| OPT actions, model, preprocessing, and preflight | 32 tests passed; 35 subtests passed |
| Synthetic end-to-end experiments | 21 passed |
| Python compilation and local documentation links | Passed |

The 21 synthetic checks cover all 16 RoBERTa methods in (5,5), all four OPT
methods in (5,5), and D01 under partial participation in (10,8). They use tiny
random models and synthetic inputs, with no MIMIC data or downloaded weights.
The preparation integration is also tested with synthetic records and a fake
tokenizer. These checks verify execution and protocol invariants, not model
quality or full-size GPU memory requirements. Full-data experiments and actual
model downloads were not rerun for this packaging task. GitHub Actions has
been configured but was not executed remotely.

Local test environment: Python 3.12.3, PyTorch 2.7.1, Transformers 4.46.3,
datasets 3.1.0, NumPy 2.0.2, scikit-learn 1.9.1, pytest 9.2.0.

## Excluded

Patient data, partition manifests containing IDs, token caches, predictions,
experiment results, model weights, checkpoints, credentials, machine-specific
paths, test outputs, Python caches, and Git history are not part of this folder.
`RELEASE_MANIFEST.json` lists SHA-256 hashes of the distributed source files.
No GitHub upload is performed by any release-preparation step.
