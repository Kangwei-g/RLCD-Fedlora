"""Read the explicitly selected dataset; never fall back to a legacy .bak."""

import pickle


def load_data(path, setting):
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    if setting == "5_5":
        from data.icu_client_partition import subset_icu10_to_large5
        data = subset_icu10_to_large5(data)
    expected = 5 if setting == "5_5" else 10
    if any(len(data[role]) != expected for role in ("train", "vali", "test")):
        raise ValueError("Dataset/client setting mismatch")
    if "server_vali" not in data:
        raise ValueError("Dedicated server reward split is required")
    return data


def inspect_data(path, setting):
    data = load_data(path, setting)
    rows = []
    for role in ("train", "vali", "test"):
        for cid, split in enumerate(data[role]):
            labels = list(split["label"])
            if not labels or not set(labels).issubset({0, 1}):
                raise ValueError("Invalid binary data split")
            rows.append(dict(role=role, client_id=cid, client_name=data["client_names"][cid],
                             n=len(labels), positive_n=sum(int(x) for x in labels)))
    labels = list(data["server_vali"]["label"])
    rows.append(dict(role="server_reward", n=len(labels), positive_n=sum(int(x) for x in labels)))
    return {"setting": setting, "path": str(path), "splits": rows,
            "note": "Counts only; patient-level deduplication is documented in the separate partition audit."}
