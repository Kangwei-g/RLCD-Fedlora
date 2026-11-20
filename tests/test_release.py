import copy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from rlcd_fedlora.aggregation import aggregate_payloads

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_delta_aggregation_leaves_unchanged_factors_unchanged():
    base = {"a": torch.tensor([2.0, 3.0])}
    for weights in ([0.3, 0.65], [0.4, 0.65]):
        result = aggregate_payloads([base, base], weights, base_state=base)
        torch.testing.assert_close(result["a"], base["a"], rtol=0, atol=0)
        with pytest.raises(ValueError, match="base_state"):
            aggregate_payloads([base, base], weights)


def test_delta_aggregation_applies_only_weighted_changes():
    result = aggregate_payloads([{"a": torch.tensor([3.0])}], [0.5], {"a": torch.tensor([1.0])})
    assert result["a"].item() == 2.0


def test_standard_seed_plans_and_baselines():
    run = module("run_experiments")
    for backbone, count in (("roberta", 96), ("opt", 12)):
        args = run.parser().parse_args(["--backbone", backbone, "--data-file", "private_data/cache.pkl"])
        jobs = run.make_plan(args)
        assert len(jobs) == count
        assert {j["seed"] for j in jobs} == {0, 4, 5}
        assert len({j["summary"] for j in jobs}) == count
        assert all("--test-every-round" not in j["command"] for j in jobs)
        assert all(Path(j["command"][1]).is_file() for j in jobs)
    args.seeds = [0, 0]
    with pytest.raises(ValueError):
        run.make_plan(args)


def test_summary_averages_seeds_and_rejects_duplicates():
    collect = module("summarize_results")
    with TemporaryDirectory() as folder:
        jobs = []
        for method in ("d01", "rl_rolora"):
            for seed in (0, 4, 5):
                internal = collect.METHODS[method]
                directory = Path(folder) / method / str(seed)
                directory.mkdir(parents=True)
                (directory / "jl_config.json").write_text(json.dumps(dict(status="completed", method=internal)))
                score = 0.6 + seed / 100
                report = dict(method=internal, seed=seed, setting="5_5",
                              client_mean={k: score for k in collect.METRICS},
                              pooled=dict(n=100, positive_n=20, fn=5, fp=10 if method == "d01" else 20))
                path = directory / "jl_summary.json"
                path.write_text(json.dumps(report))
                jobs.append(dict(backbone="opt", setting="5_5", method=method, seed=seed, summary=str(path)))
        plan = dict(seeds=[0, 4, 5], jobs=jobs)
        rows = collect.summarize(plan)
        assert rows[0]["macro_f1_mean"] == pytest.approx(0.63)
        assert rows[0]["dv"] == pytest.approx(574)
        assert rows[1]["dv"] == pytest.approx(0)
        bad = copy.deepcopy(plan)
        bad["jobs"].append(bad["jobs"][0])
        with pytest.raises(ValueError, match="Duplicate"):
            collect.summarize(bad)
        bad = copy.deepcopy(plan)
        bad["jobs"].pop()
        with pytest.raises(ValueError, match="Missing seeds"):
            collect.summarize(bad)
