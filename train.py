#!/usr/bin/env python3
"""Public entry point for validating an RLCD-FedLoRA configuration."""

import argparse
from pathlib import Path

import yaml

from examples.synthetic_smoke_test import run_smoke_test


def load_config(path: str):
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {"experiment", "model", "local_training", "controller", "ddqn"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError("Configuration sections are missing: {}".format(missing))
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run the data-independent end-to-end method check.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if not args.smoke_test:
        parser.error(
            "Use --smoke-test or connect a dataset adapter as described in "
            "data/README.md."
        )
    run_smoke_test(config)


if __name__ == "__main__":
    main()
