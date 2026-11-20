#!/usr/bin/env python3
"""Report equal-seed means/stds from an explicit, complete experiment plan."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean, stdev

from run_experiments import METHODS

METRICS = ("macro_precision", "macro_recall", "macro_f1", "auroc")


def summarize(plan, cost=0.574):
    groups, seen = defaultdict(list), set()
    for job in plan["jobs"]:
        identity = (job["backbone"], job["setting"], job["method"], job["seed"])
        if identity in seen:
            raise ValueError("Duplicate method/setting/seed in plan")
        seen.add(identity)
        path = Path(job["summary"])
        config = json.loads((path.parent / "jl_config.json").read_text())
        if config.get("status") != "completed":
            raise ValueError("Run is not completed: " + str(path.parent))
        report = json.loads(path.read_text())
        if report["seed"] != job["seed"] or report["setting"] != job["setting"]:
            raise ValueError("Summary identity does not match plan")
        if report["method"] != config["method"] or report["method"] != METHODS[job["method"]]:
            raise ValueError("Method mismatch")
        metrics = report["pooled"] if report.get("primary_metric_scope") == "pooled" else report["client_mean"]
        counts = report["pooled"]
        loss = 10000 * ((7 / 3) * counts["fn"] + counts["fp"]) / counts["n"]
        groups[identity[:3]].append(dict(seed=job["seed"], loss=loss, n=counts["n"],
                                        positive_n=counts["positive_n"], **{k: metrics[k] for k in METRICS}))
    rows = []
    for (backbone, setting, method), reports in sorted(groups.items()):
        if sorted(r["seed"] for r in reports) != sorted(plan["seeds"]):
            raise ValueError("Missing seeds for " + method)
        row = dict(backbone=backbone, setting=setting, method=method, n_seeds=len(reports),
                   mean_decision_loss=mean(r["loss"] for r in reports))
        for metric in METRICS:
            values = [r[metric] for r in reports]
            if any(value is None for value in values):
                raise ValueError("Undefined " + metric + " for " + method)
            row[metric + "_mean"] = mean(values)
            row[metric + "_std"] = stdev(values) if len(values) > 1 else 0.0
        rows.append(row)
    # A table-dependent reference must be the same for every method in a setting.
    for row in rows:
        related = [r for r in rows if (r["backbone"], r["setting"]) == (row["backbone"], row["setting"])]
        counts = {(r["n"], r["positive_n"]) for key, reports in groups.items()
                  if key[:2] == (row["backbone"], row["setting"]) for r in reports}
        if len(counts) != 1:
            raise ValueError("DV requires matching test cohort sizes and class counts")
        row["dv_reference_loss"] = max(r["mean_decision_loss"] for r in related)
        row["dv"] = cost * (row["dv_reference_loss"] - row["mean_decision_loss"])
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--methods", nargs="+", help="Comparison table membership for the DV reference")
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.methods:
        if set(args.methods) - {j["method"] for j in plan["jobs"]}:
            raise SystemExit("Unknown requested method")
        plan["jobs"] = [j for j in plan["jobs"] if j["method"] in args.methods]
    rows = summarize(plan)
    if not rows:
        raise SystemExit("No results")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
