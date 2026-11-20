#!/usr/bin/env python3
"""Audit ICU federated splits for exact-text and token-level overlap."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


CLIENT_SPLITS = ("train", "vali", "test")
ICU_UNITS = ("CCU", "CSRU", "MICU", "SICU", "TSICU")
TEXT_COL = "TEXT"
LABEL_COL = "1_YEAR_MORTALITY"
CLIENT_COL = "FIRST_CAREUNIT"
ID_COLS = ("ROW_ID", "SUBJECT_ID", "HADM_ID")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("data/mimic_icu10_server500_clients.pkl"),
    )
    parser.add_argument(
        "--tokenized",
        type=Path,
        default=Path("data/mimic_icu10_server500_tokenized_512.pkl.bak"),
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=Path("ceshishujuji/LABELED_DATASET_CLIENTS_FULL.csv"),
    )
    parser.add_argument("--partition-seed", type=int, default=42)
    parser.add_argument(
        "--manifest", type=Path,
        help="Use repaired row provenance instead of replaying the original random split.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_pickle(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def normalized_text(value):
    value = unicodedata.normalize("NFKC", str(value)).strip()
    return re.sub(r"\s+", " ", value)


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def token_digest(row):
    ids = row["input_ids"]
    mask = row.get("attention_mask")
    if mask is not None:
        ids = [token for token, active in zip(ids, mask) if int(active)]
    return digest(",".join(str(int(token)) for token in ids))


def selected_client_indices(dataset, mode):
    names = [str(name) for name in dataset["client_names"]]
    if mode == "icu10":
        return list(range(len(names)))
    indices = [idx for idx, name in enumerate(names) if name.endswith("_1500")]
    if len(indices) != 5:
        raise ValueError(f"Expected five *_1500 clients, got {indices}")
    return indices


def unit_seed(base_seed, unit_name):
    return base_seed + sum(ord(char) for char in unit_name)


def split_client_frame(frame, seed):
    stratify = frame["label"] if frame["label"].nunique() == 2 else None
    train, temporary = train_test_split(
        frame, test_size=0.2, stratify=stratify, random_state=seed
    )
    temporary_stratify = (
        temporary["label"] if temporary["label"].nunique() == 2 else None
    )
    validation, test = train_test_split(
        temporary,
        test_size=0.5,
        stratify=temporary_stratify,
        random_state=seed,
    )
    return train, validation, test


def replay_partition_from_source(path, seed):
    columns = [*ID_COLS, TEXT_COL, LABEL_COL, CLIENT_COL]
    frame = pd.read_csv(path, usecols=columns)
    frame = frame.dropna(subset=[TEXT_COL, LABEL_COL, CLIENT_COL]).copy()
    frame[TEXT_COL] = frame[TEXT_COL].astype(str).str.strip()
    frame[CLIENT_COL] = frame[CLIENT_COL].astype(str).str.strip()
    frame = frame[frame[TEXT_COL] != ""]
    frame[LABEL_COL] = pd.to_numeric(frame[LABEL_COL], errors="coerce")
    frame = frame.dropna(subset=[LABEL_COL])
    frame[LABEL_COL] = frame[LABEL_COL].astype(int)
    frame = frame[frame[LABEL_COL].isin([0, 1])]
    frame = frame[~frame[CLIENT_COL].isin({"NICU", "Non-ICU"})]
    frame = frame[frame[CLIENT_COL].isin(ICU_UNITS)].copy()
    frame = frame.rename(columns={TEXT_COL: "text", LABEL_COL: "label"})
    frame = frame.reset_index(drop=True)

    groups = {}
    server_parts = []
    for unit in ICU_UNITS:
        unit_frame = frame[frame[CLIENT_COL] == unit].reset_index(drop=True)
        shuffled = unit_frame.sample(
            n=2600, random_state=unit_seed(seed, unit)
        ).reset_index(drop=True)
        server_parts.append(shuffled.iloc[:100].copy())
        client_frames = {
            f"{unit}_1000": shuffled.iloc[100:1100].copy(),
            f"{unit}_1500": shuffled.iloc[1100:2600].copy(),
        }
        for client_name, client_frame in client_frames.items():
            parts = split_client_frame(
                client_frame, unit_seed(seed, client_name)
            )
            for split, part in zip(CLIENT_SPLITS, parts):
                groups[f"client:{client_name}:{split}"] = part

    server = pd.concat(server_parts, ignore_index=True)
    stratify = server["label"] if server["label"].nunique() == 2 else None
    # The archived dataset was first built as validation=400/test=100 and then
    # swapped by swap_server_vali_test.py to validation=100/test=400.
    original_validation, original_test = train_test_split(
        server, test_size=0.2, stratify=stratify, random_state=seed
    )
    groups["server:server_vali"] = original_test
    groups["server:server_test"] = original_validation
    return groups


def validate_replay(raw, replayed):
    mismatches = {}
    for group, frame in replayed.items():
        if group.startswith("server:"):
            split = group.split(":", 1)[1]
            dataset = raw[split]
        else:
            _, client_name, split = group.split(":", 2)
            client_idx = list(raw["client_names"]).index(client_name)
            dataset = raw[split][client_idx]
        replay_counter = Counter(
            (digest(normalized_text(row.text)), int(row.label))
            for row in frame.itertuples(index=False)
        )
        pickle_counter = Counter(
            (digest(normalized_text(row["text"])), int(row["label"]))
            for row in dataset
        )
        if replay_counter != pickle_counter:
            mismatches[group] = {
                "replayed_size": len(frame),
                "pickle_size": len(dataset),
                "only_replayed": sum((replay_counter - pickle_counter).values()),
                "only_pickle": sum((pickle_counter - replay_counter).values()),
            }
    return {
        "matched_all_groups": not mismatches,
        "mismatches": mismatches,
    }


def groups_from_manifest(manifest_path, source_path, raw, tokenized):
    manifest = pd.read_csv(manifest_path)
    source = pd.read_csv(source_path, usecols=[*ID_COLS, TEXT_COL, LABEL_COL, CLIENT_COL])
    source = source.set_index("ROW_ID", verify_integrity=True)
    expected = {
        f"client:{name}:{split}"
        for name in raw["client_names"] for split in CLIENT_SPLITS
    }
    expected.update(f"server:{split}" for split in ("server_vali", "server_test") if split in raw)
    if set(manifest["group"]) != expected:
        raise ValueError("Manifest does not contain exactly the dataset groups")
    groups = {}
    for group, part in manifest.groupby("group", sort=False):
        part = part.sort_values("index")
        if group.startswith("server:"):
            split = group.split(":", 1)[1]
            raw_split, tok_split = raw[split], tokenized[split]
        else:
            _, client, split = group.split(":")
            client_id = list(raw["client_names"]).index(client)
            raw_split, tok_split = raw[split][client_id], tokenized[split][client_id]
        if part["index"].tolist() != list(range(len(raw_split))) or len(tok_split) != len(raw_split):
            raise ValueError(f"Manifest/raw/token row count or index mismatch: {group}")
        frame = source.loc[part["ROW_ID"].tolist()].reset_index()
        frame[TEXT_COL] = frame[TEXT_COL].astype(str).str.strip()
        frame[CLIENT_COL] = frame[CLIENT_COL].astype(str).str.strip()
        frame = frame.rename(columns={TEXT_COL: "text", LABEL_COL: "label"})
        for index, (record, source_row) in enumerate(zip(part.to_dict("records"), frame.to_dict("records"))):
            for key in (*ID_COLS, "label"):
                if int(record[key]) != int(source_row[key]):
                    raise ValueError(f"Manifest/source {key} mismatch: {group}[{index}]")
            if record[CLIENT_COL] != source_row[CLIENT_COL]:
                raise ValueError(f"Manifest/source care-unit mismatch: {group}[{index}]")
            raw_row, tok_row = raw_split[index], tok_split[index]
            if raw_row["text"] != source_row["text"] or int(raw_row["label"]) != int(record["label"]):
                raise ValueError(f"Raw/source mismatch: {group}[{index}]")
            if digest(normalized_text(raw_row["text"])) != record["text_key"]:
                raise ValueError(f"Manifest/text hash mismatch: {group}[{index}]")
            if token_digest(tok_row) != record["token_key"] or int(tok_row["label"]) != int(record["label"]):
                raise ValueError(f"Manifest/token mismatch: {group}[{index}]")
        groups[group] = frame
    return groups


def filter_replayed_groups(groups, mode):
    if mode == "icu10":
        return groups
    return {
        name: frame
        for name, frame in groups.items()
        if name.startswith("server:")
        or name.split(":", 2)[1].endswith("_1500")
    }


def collect_groups(raw, tokenized, mode):
    groups = {}
    indices = selected_client_indices(raw, mode)
    for idx in indices:
        client_name = str(raw["client_names"][idx])
        for split in CLIENT_SPLITS:
            raw_split = raw[split][idx]
            token_split = tokenized[split][idx]
            if len(raw_split) != len(token_split):
                raise ValueError(f"Length mismatch for {client_name}/{split}")
            rows = []
            for row_idx in range(len(raw_split)):
                raw_row = raw_split[row_idx]
                token_row = token_split[row_idx]
                text_key = digest(normalized_text(raw_row["text"]))
                label = int(raw_row["label"])
                if label != int(token_row["label"]):
                    raise ValueError(f"Label mismatch for {client_name}/{split}/{row_idx}")
                rows.append(
                    {
                        "text": text_key,
                        "text_label": f"{text_key}:{label}",
                        "tokens": token_digest(token_row),
                        "tokens_label": f"{token_digest(token_row)}:{label}",
                        "label": label,
                    }
                )
            groups[f"client:{client_name}:{split}"] = rows

    for split in ("server_vali", "server_test"):
        if split not in raw or split not in tokenized:
            continue
        raw_split = raw[split]
        token_split = tokenized[split]
        if len(raw_split) != len(token_split):
            raise ValueError(f"Length mismatch for {split}")
        rows = []
        for row_idx in range(len(raw_split)):
            raw_row = raw_split[row_idx]
            token_row = token_split[row_idx]
            text_key = digest(normalized_text(raw_row["text"]))
            label = int(raw_row["label"])
            if label != int(token_row["label"]):
                raise ValueError(f"Label mismatch for {split}/{row_idx}")
            token_key = token_digest(token_row)
            rows.append(
                {
                    "text": text_key,
                    "text_label": f"{text_key}:{label}",
                    "tokens": token_key,
                    "tokens_label": f"{token_key}:{label}",
                    "label": label,
                }
            )
        groups[f"server:{split}"] = rows
    return groups


def group_kind(name):
    if name.startswith("server:"):
        return "server", name, name.split(":", 1)[1]
    _, client, split = name.split(":", 2)
    return "client", client, split


def pair_category(left, right):
    left_kind, left_owner, _ = group_kind(left)
    right_kind, right_owner, _ = group_kind(right)
    if left_kind == right_kind == "server":
        return "server_validation_vs_test"
    if left_kind != right_kind:
        return "client_vs_server"
    if left_owner == right_owner:
        return "within_client_cross_split"
    return "cross_client"


def audit_key(groups, key):
    within = {}
    sets = {}
    labels_by_content = defaultdict(set)
    locations_by_content = defaultdict(set)
    for group, rows in groups.items():
        values = [row[key] for row in rows]
        counts = Counter(values)
        duplicate_rows = sum(count - 1 for count in counts.values() if count > 1)
        if duplicate_rows:
            within[group] = {
                "duplicate_rows": duplicate_rows,
                "duplicate_values": sum(count > 1 for count in counts.values()),
            }
        sets[group] = set(values)
        base_key = key.replace("_label", "")
        for row in rows:
            labels_by_content[row[base_key]].add(row["label"])
            locations_by_content[row[base_key]].add(group)

    names = sorted(groups)
    overlaps = []
    category_totals = Counter()
    for pos, left in enumerate(names):
        for right in names[pos + 1 :]:
            shared = sets[left] & sets[right]
            if not shared:
                continue
            category = pair_category(left, right)
            overlaps.append(
                {
                    "left": left,
                    "right": right,
                    "category": category,
                    "shared_unique": len(shared),
                }
            )
            category_totals[category] += len(shared)

    cross_group_values = sum(
        len(locations) > 1 for locations in locations_by_content.values()
    )
    return {
        "within_group_duplicates": within,
        "overlap_pair_count": len(overlaps),
        "cross_group_unique_values": cross_group_values,
        "pairwise_shared_totals": dict(sorted(category_totals.items())),
        "pairwise_overlaps": overlaps,
        "conflicting_label_values": sum(
            len(labels) > 1 for labels in labels_by_content.values()
        ),
    }


def audit_identifier(groups, key):
    sets = {}
    within = {}
    locations = defaultdict(set)
    for group, frame in groups.items():
        values = [int(value) for value in frame[key].tolist()]
        counts = Counter(values)
        duplicate_rows = sum(count - 1 for count in counts.values() if count > 1)
        if duplicate_rows:
            within[group] = {
                "duplicate_rows": duplicate_rows,
                "duplicate_identifiers": sum(count > 1 for count in counts.values()),
            }
        sets[group] = set(values)
        for value in values:
            locations[value].add(group)

    names = sorted(groups)
    overlaps = []
    category_totals = Counter()
    for pos, left in enumerate(names):
        for right in names[pos + 1 :]:
            shared = sets[left] & sets[right]
            if not shared:
                continue
            category = pair_category(left, right)
            overlaps.append(
                {
                    "left": left,
                    "right": right,
                    "category": category,
                    "shared_unique": len(shared),
                }
            )
            category_totals[category] += len(shared)

    def union_for(predicate):
        selected = [values for name, values in sets.items() if predicate(name)]
        return set().union(*selected) if selected else set()

    client_train = union_for(
        lambda name: name.startswith("client:") and name.endswith(":train")
    )
    client_validation = union_for(
        lambda name: name.startswith("client:") and name.endswith(":vali")
    )
    client_test = union_for(
        lambda name: name.startswith("client:") and name.endswith(":test")
    )
    server_validation = sets.get("server:server_vali", set())
    server_test = sets.get("server:server_test", set())
    client_all = client_train | client_validation | client_test
    server_all = server_validation | server_test

    def overlap_summary(source, target):
        shared = source & target
        return {
            "shared_unique": len(shared),
            "target_unique": len(target),
            "target_fraction": len(shared) / len(target) if target else 0.0,
        }

    return {
        "unique_identifiers": len(locations),
        "within_group_duplicates": within,
        "overlap_pair_count": len(overlaps),
        "cross_group_unique_identifiers": sum(
            len(group_names) > 1 for group_names in locations.values()
        ),
        "pairwise_shared_totals": dict(sorted(category_totals.items())),
        "pairwise_overlaps": overlaps,
        "role_unique_identifiers": {
            "client_train": len(client_train),
            "client_validation": len(client_validation),
            "client_test": len(client_test),
            "server_validation": len(server_validation),
            "server_test": len(server_test),
        },
        "leakage_summary": {
            "client_train_to_client_validation": overlap_summary(
                client_train, client_validation
            ),
            "client_train_to_client_test": overlap_summary(client_train, client_test),
            "client_train_to_server_validation": overlap_summary(
                client_train, server_validation
            ),
            "client_train_to_server_test": overlap_summary(client_train, server_test),
            "client_train_to_all_evaluation": overlap_summary(
                client_train,
                client_validation | client_test | server_validation | server_test,
            ),
            "client_validation_to_client_test": overlap_summary(
                client_validation, client_test
            ),
            "server_validation_to_server_test": overlap_summary(
                server_validation, server_test
            ),
            "any_client_to_any_server": overlap_summary(client_all, server_all),
        },
    }


def audit_mode(raw, tokenized, replayed, mode):
    groups = collect_groups(raw, tokenized, mode)
    replayed_mode = filter_replayed_groups(replayed, mode)
    return {
        "mode": mode,
        "group_sizes": {name: len(rows) for name, rows in sorted(groups.items())},
        "total_occurrences": sum(len(rows) for rows in groups.values()),
        "normalized_text": audit_key(groups, "text"),
        "normalized_text_plus_label": audit_key(groups, "text_label"),
        "token_sequence": audit_key(groups, "tokens"),
        "token_sequence_plus_label": audit_key(groups, "tokens_label"),
        "row_id": audit_identifier(replayed_mode, "ROW_ID"),
        "subject_id": audit_identifier(replayed_mode, "SUBJECT_ID"),
        "hadm_id": audit_identifier(replayed_mode, "HADM_ID"),
    }


def main():
    args = parse_args()
    raw = load_pickle(args.raw)
    tokenized = load_pickle(args.tokenized)
    replayed = (
        groups_from_manifest(args.manifest, args.source_csv, raw, tokenized)
        if args.manifest
        else replay_partition_from_source(args.source_csv, args.partition_seed)
    )
    replay_validation = validate_replay(raw, replayed)
    if not replay_validation["matched_all_groups"]:
        raise ValueError("Source reconstruction differs from raw data; use the matching repair --manifest")
    report = {
        "raw_file": str(args.raw),
        "tokenized_file": str(args.tokenized),
        "source_csv": str(args.source_csv),
        "partition_seed": args.partition_seed,
        "manifest": str(args.manifest) if args.manifest else None,
        "replay_validation": replay_validation,
        "icu10": audit_mode(raw, tokenized, replayed, "icu10"),
        "icu5": audit_mode(raw, tokenized, replayed, "icu5"),
    }
    rendered = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
