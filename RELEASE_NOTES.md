# Source Release: 2026-11-20

## Experiment Protocol

- All methods use 200 communication rounds per run under three different
  random seeds, with one run per method/configuration/seed.
- Validation selects checkpoints and thresholds before final testing.
- Results are reported as means and sample standard deviations across seeds.

## Contents

- RoBERTa-base: RLCD-FedLoRA, eight hybrid baselines, four component
  ablations, and local-only, centralized, and FedAvg-Full controls.
- OPT-125M: RLCD-FedLoRA, FlexLoRA + CD, RL + RoLoRA, and RL + FedAvg-LoRA.
- Cohort preparation, overlap auditing, model downloads, metric summaries,
  trajectory plots, and synthetic tests.

See [README.md](README.md) for setup, method definitions, and commands.
Patient data, generated outputs, and model weights are excluded.
`RELEASE_MANIFEST.json` records SHA-256 hashes of the source files.
