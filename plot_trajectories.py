#!/usr/bin/env python3
"""Plot validation Macro-F1 and executed G from per-round records."""
import argparse
import csv
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("metrics", nargs="+", type=Path, help="jl_round_metrics.csv files")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True, constrained_layout=True)
    for path in args.metrics:
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        rounds = [int(r["round"]) for r in rows]
        label = path.parent.name
        axes[0].plot(rounds, [float(r["validation_macro_f1"]) for r in rows], lw=1, label=label)
        if all(r.get("G") not in (None, "") for r in rows):
            axes[1].step(rounds, [int(r["G"]) for r in rows], lw=0.7, where="mid", label=label)
    axes[0].set_ylabel("Validation Macro-F1")
    axes[1].set_ylabel(r"$G_t$")
    axes[1].set_xlabel("Communication round")
    axes[1].set_yticks([1, 4, 8, 12])
    axes[0].legend(frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)


if __name__ == "__main__":
    main()
