"""Subset helpers for ICU10 federated partitions."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


def select_icu5_large_indices(client_names: Sequence[str], train_splits: Sequence[Any]) -> List[int]:
    """
    Pick the 5 larger ICU clients from the 10-client partition.

    The ICU10 scheme assigns 1000/1500 samples per care unit; we keep the *_1500 clients.
    Falls back to the top-5 clients by train size if naming does not match.
    """
    names = [str(n) for n in client_names]
    by_1500 = [i for i, n in enumerate(names) if n.endswith("_1500")]
    if len(by_1500) == 5:
        return sorted(by_1500)

    sizes = [len(ds) for ds in train_splits]
    top5 = sorted(range(len(sizes)), key=lambda i: sizes[i], reverse=True)[:5]
    return sorted(top5)


def subset_icu10_to_large5(sst_dataset: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of the dataset restricted to the 5 larger ICU clients."""
    names = sst_dataset.get("client_names", [])
    train = sst_dataset["train"]
    indices = select_icu5_large_indices(names, train)
    if len(indices) != 5:
        raise ValueError(f"Expected 5 ICU large clients, got {len(indices)} from {list(names)}")

    out = dict(sst_dataset)
    for split in ("train", "vali", "test"):
        out[split] = [sst_dataset[split][i] for i in indices]
    out["client_names"] = [names[i] for i in indices]
    if "client_train_sizes" in sst_dataset:
        out["client_train_sizes"] = [sst_dataset["client_train_sizes"][i] for i in indices]
    out["total_client_train_size"] = sum(len(ds) for ds in out["train"])
    out["partition_scheme"] = "icu5_large_from_icu10"
    out["parent_partition"] = sst_dataset.get("partition_scheme", "icu10_server500")
    out["client_indices_in_parent"] = indices
    out["sample_fraction_default"] = 1.0
    out["clients_per_round_default"] = 5
    return out
