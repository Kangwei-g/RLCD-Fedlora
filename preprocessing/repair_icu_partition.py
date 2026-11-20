#!/usr/bin/env python3
"""Stage and install a label/care-unit preserving repair of the ICU10 partition.

Keep patients in one client/split, remove duplicate admissions and model inputs,
and replace conflicts using patients absent from the entire original partition.
Different admissions of a retained patient may remain within the same split.
No training results or prediction scores enter the selection procedure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import shutil
from collections import Counter, defaultdict, deque
from pathlib import Path


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def group_priority(name):
    if name.startswith("client:"):
        _, client, split = name.split(":")
        role = {"test": 0, "vali": 3, "train": 4}[split]
        return role, not client.endswith("_1500"), client
    return {"server:server_test": 1, "server:server_vali": 2}[name], False, name


UNIQUE_KEYS = ("ROW_ID", "HADM_ID", "text_key", "token_key")


def mark_conflicts(groups):
    """Preserve held-out observations first, without inspecting model outcomes."""
    patient_owner = {}
    seen = {key: set() for key in UNIQUE_KEYS}
    conflicts = []
    for group in sorted(groups, key=group_priority):
        for index, row in enumerate(groups[group]):
            reasons = [f"duplicate_{key}" for key in UNIQUE_KEYS if row[key] in seen[key]]
            owner = patient_owner.get(row["SUBJECT_ID"])
            if owner is not None and owner != group:
                reasons.append("patient_cross_group")
            if reasons:
                conflicts.append((group, index, reasons))
                continue
            patient_owner[row["SUBJECT_ID"]] = group
            for key in UNIQUE_KEYS:
                seen[key].add(row[key])
    return conflicts


def select_replacements(groups, candidates, encode_one, seed):
    """Return repaired rows and an audit trail; fail rather than relax constraints."""
    conflicts = mark_conflicts(groups)
    used = {key: set() for key in (*UNIQUE_KEYS, "SUBJECT_ID")}
    for rows in groups.values():
        for row in rows:
            for key in used:
                used[key].add(row[key])
    pools = defaultdict(list)
    for row in candidates:
        if any(row[key] in used[key] for key in used if key != "token_key"):
            continue
        pools[(row["FIRST_CAREUNIT"], row["label"])].append(row)
    pool_sizes = {f"{unit}:{label}": len(rows) for (unit, label), rows in pools.items()}
    for stratum, rows in sorted(pools.items()):
        random.Random(seed + sum(map(ord, str(stratum)))).shuffle(rows)
    repaired = {group: [dict(row) for row in rows] for group, rows in groups.items()}
    replacements = []
    for group, index, reasons in conflicts:
        old = groups[group][index]
        stratum = old["FIRST_CAREUNIT"], old["label"]
        pool = pools[stratum]
        while pool:
            new = dict(pool.pop())
            if any(new[key] in used[key] for key in used if key != "token_key"):
                continue
            new.update(encode_one(new["text"]))
            if new["token_key"] in used["token_key"]:
                continue
            break
        else:
            raise ValueError(f"No unused same-label/care-unit candidate for {group}[{index}]: {stratum}")
        repaired[group][index] = new
        for key in used:
            used[key].add(new[key])
        replacements.append({
            "group": group, "index": index, "reasons": reasons,
            "label": old["label"], "care_unit": old["FIRST_CAREUNIT"],
            "old_ROW_ID": old["ROW_ID"], "new_ROW_ID": new["ROW_ID"],
            "old_SUBJECT_ID": old["SUBJECT_ID"], "new_SUBJECT_ID": new["SUBJECT_ID"],
            "old_HADM_ID": old["HADM_ID"], "new_HADM_ID": new["HADM_ID"],
        })
    if mark_conflicts(repaired):
        raise AssertionError("The repaired partition still has conflicts")
    for group, rows in groups.items():
        signature = lambda data: Counter((row["FIRST_CAREUNIT"], row["label"]) for row in data)
        if signature(rows) != signature(repaired[group]):
            raise AssertionError(f"Stratum counts changed in {group}")
    return repaired, replacements, pool_sizes


def dataset_group(dataset, group):
    if group.startswith("server:"):
        return dataset[group.split(":", 1)[1]]
    _, client, split = group.split(":")
    return dataset[split][list(dataset["client_names"]).index(client)]


def set_dataset_group(dataset, group, value):
    if group.startswith("server:"):
        dataset[group.split(":", 1)[1]] = value
    else:
        _, client, split = group.split(":")
        dataset[split][list(dataset["client_names"]).index(client)] = value


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def assert_clean(report):
    for mode in ("icu10", "icu5"):
        for key in ("normalized_text", "token_sequence"):
            section = report[mode][key]
            if section["within_group_duplicates"] or section["cross_group_unique_values"]:
                raise AssertionError(f"Remaining {mode} {key} duplicates")
        for key in ("row_id", "hadm_id", "subject_id"):
            section = report[mode][key]
            if section["cross_group_unique_identifiers"]:
                raise AssertionError(f"Remaining {mode} {key} cross-group overlap")
            if key != "subject_id" and section["within_group_duplicates"]:
                raise AssertionError(f"Remaining {mode} {key} duplicates")


def stage(args):
    import pandas as pd
    from datasets import Dataset
    from transformers import RobertaTokenizerFast
    from audit_icu_partition_overlap import (
        ICU_UNITS, audit_mode, digest, normalized_text, replay_partition_from_source,
        token_digest, validate_replay,
    )
    from prepare_mimic_careunit_tokenized_512 import _encode_head_tail_texts

    work = args.output_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    backup = work / "backup"
    staged = work / "staged"
    backup.mkdir()
    staged.mkdir()
    paths = [args.raw, args.tokenized, args.tokenized_main, args.raw_stats, args.token_stats]
    originals = {}
    for path in dict.fromkeys(path.resolve() for path in paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        target = backup / path.name
        shutil.copy2(path, target)
        sha = file_hash(path)
        if file_hash(target) != sha:
            raise RuntimeError(f"Backup verification failed: {path}")
        originals[str(path)] = {"sha256": sha, "backup": str(target)}
    write_json(work / "originals.json", originals)
    print("Verified backups of all active data/statistics files.", flush=True)

    raw = pickle.load(args.raw.open("rb"))
    tokenized = pickle.load(args.tokenized.open("rb"))
    replayed = replay_partition_from_source(args.source_csv, args.partition_seed)
    replay_check = validate_replay(raw, replayed)
    if not replay_check["matched_all_groups"]:
        raise ValueError(f"Cannot safely recover original IDs: {replay_check}")
    if tokenized.get("tokenizer_model") != "roberta" or tokenized.get("tokenization_strategy") != "head_tail":
        raise ValueError("Unexpected tokenizer configuration")
    if tokenized.get("max_length") != 512:
        raise ValueError("Expected max_length=512")
    tokenizer = RobertaTokenizerFast.from_pretrained(args.tokenizer, local_files_only=True)

    def encode(texts):
        return _encode_head_tail_texts(texts, tokenizer, 512, 256, 256)

    def encode_one(text):
        ids, masks, _ = encode([text])
        row = {"input_ids": ids[0], "attention_mask": masks[0]}
        return {**row, "token_key": token_digest(row)}

    groups = {}
    for group, frame in replayed.items():
        queues = defaultdict(deque)
        for source_row in frame.to_dict("records"):
            queues[(source_row["text"], int(source_row["label"]))].append(source_row)
        original = dataset_group(raw, group)
        original_tokens = dataset_group(tokenized, group)
        if len(original) != len(original_tokens):
            raise ValueError(f"Raw/token length mismatch: {group}")
        rows = []
        for index, old in enumerate(original):
            source_row = queues[(old["text"], int(old["label"]))].popleft()
            tok = original_tokens[index]
            if int(tok["label"]) != int(old["label"]):
                raise ValueError(f"Raw/token label mismatch: {group}[{index}]")
            row = {**source_row, **tok}
            row.update({key: int(row[key]) for key in ("ROW_ID", "SUBJECT_ID", "HADM_ID", "label")})
            row["text_key"] = digest(normalized_text(row["text"]))
            row["token_key"] = token_digest(tok)
            rows.append(row)
        # Check every existing model input before reusing its tokenization recipe.
        for start in range(0, len(rows), 128):
            batch = rows[start:start + 128]
            ids, masks, _ = encode([row["text"] for row in batch])
            for offset, row in enumerate(batch):
                if ids[offset] != row["input_ids"] or masks[offset] != row["attention_mask"]:
                    raise ValueError(f"Tokenization differs from original: {group}[{start + offset}]")
        groups[group] = rows
        print(f"Verified source IDs and tokenization: {group} n={len(rows)}", flush=True)

    before = {mode: audit_mode(raw, tokenized, replayed, mode) for mode in ("icu10", "icu5")}
    before["replay_validation"] = replay_check
    write_json(work / "before.json", before)
    columns = ["ROW_ID", "SUBJECT_ID", "HADM_ID", "TEXT", "1_YEAR_MORTALITY", "FIRST_CAREUNIT"]
    source = pd.read_csv(args.source_csv, usecols=columns).dropna(subset=columns).copy()
    source["TEXT"] = source["TEXT"].astype(str).str.strip()
    source["FIRST_CAREUNIT"] = source["FIRST_CAREUNIT"].astype(str).str.strip()
    source["1_YEAR_MORTALITY"] = pd.to_numeric(source["1_YEAR_MORTALITY"], errors="coerce")
    source = source[source["1_YEAR_MORTALITY"].isin([0, 1]) & source["FIRST_CAREUNIT"].isin(ICU_UNITS) & source["TEXT"].ne("")]
    source = source.rename(columns={"TEXT": "text", "1_YEAR_MORTALITY": "label"})
    candidates = source.to_dict("records")
    for row in candidates:
        for key in ("ROW_ID", "SUBJECT_ID", "HADM_ID", "label"):
            row[key] = int(row[key])
        row["text_key"] = digest(normalized_text(row["text"]))
    repaired, replacements, pools = select_replacements(groups, candidates, encode_one, args.seed)
    print(f"Selected {len(replacements)} replacements with unchanged per-group label/care-unit counts.", flush=True)

    new_raw = {**raw, **{split: list(raw[split]) for split in ("train", "vali", "test")}}
    new_tok = {**tokenized, **{split: list(tokenized[split]) for split in ("train", "vali", "test")}}
    metadata = {
        "seed": args.seed, "partition_seed": args.partition_seed,
        "source_sha256": file_hash(args.source_csv),
        "policy": "test_first; patient_disjoint_groups; unique_admissions_text_tokens; unused_patient_same_label_care_unit",
        "within_group_repeat_patients": "allowed only for distinct admissions",
        "manifest": str(work / "partition_manifest.csv"),
        "replacements": len(replacements),
    }
    token_stats = {"split_tokenization_stats": {}, "encode_device": "cpu", "partition_repair": metadata}
    manifest = []
    frames = {}
    replaced_locations = {(item["group"], item["index"]): item for item in replacements}
    for group, rows in repaired.items():
        original = dataset_group(raw, group)
        original_tokens = dataset_group(tokenized, group)
        new_raw_split = Dataset.from_dict({key: [row[key] for row in rows] for key in original.column_names}, features=original.features)
        new_tok_split = Dataset.from_dict({key: [row[key] for row in rows] for key in original_tokens.column_names}, features=original_tokens.features)
        set_dataset_group(new_raw, group, new_raw_split)
        set_dataset_group(new_tok, group, new_tok_split)
        frames[group] = pd.DataFrame([{key: row[key] for key in (*columns[:3], "text", "label", "FIRST_CAREUNIT")} for row in rows])
        for index, row in enumerate(rows):
            manifest.append({
                "group": group, "index": index,
                **{key: row[key] for key in (*columns[:3], "label", "FIRST_CAREUNIT", "text_key", "token_key")},
                "replaced": int((group, index) in replaced_locations),
                "original_ROW_ID": groups[group][index]["ROW_ID"],
            })
        # Recompute descriptive tokenization statistics and verify the staged inputs.
        ids, masks, stats = encode([row["text"] for row in rows])
        if ids != new_tok_split["input_ids"] or masks != new_tok_split["attention_mask"]:
            raise AssertionError(f"Staged raw/token mismatch: {group}")
        split = group.split(":")[-1]
        token_stats["split_tokenization_stats"].setdefault(split, []).append({"group": group, **stats})
    new_raw["partition_repair"] = metadata
    new_tok["partition_repair"] = metadata
    new_tok["encode_device"] = "cpu"
    pd.DataFrame(manifest).to_csv(work / "partition_manifest.csv", index=False)
    write_json(work / "replacements.json", replacements)
    after = {mode: audit_mode(new_raw, new_tok, frames, mode) for mode in ("icu10", "icu5")}
    assert_clean(after)
    write_json(work / "after.json", after)
    token_stats["split_sizes"] = {
        **{split: [len(part) for part in new_tok[split]] for split in ("train", "vali", "test")},
        **{split: len(new_tok[split]) for split in ("server_vali", "server_test")},
    }
    raw_stats = json.loads(args.raw_stats.read_text())
    raw_stats["partition_repair"] = metadata
    for path, payload in ((staged / args.raw.name, new_raw), (staged / args.tokenized_main.name, new_tok)):
        with path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(staged / args.raw_stats.name, raw_stats)
    write_json(staged / args.token_stats.name, token_stats)
    targets = {
        str(args.raw.resolve()): str(staged / args.raw.name),
        str(args.tokenized.resolve()): str(staged / args.tokenized_main.name),
        str(args.tokenized_main.resolve()): str(staged / args.tokenized_main.name),
        str(args.raw_stats.resolve()): str(staged / args.raw_stats.name),
        str(args.token_stats.resolve()): str(staged / args.token_stats.name),
    }
    plan = {
        "metadata": metadata, "originals": originals, "targets": targets,
        "staged_sha256": {path: file_hash(path) for path in set(targets.values())},
        "candidate_pool_rows": pools,
        "replacement_counts": dict(Counter(item["group"] for item in replacements)),
    }
    write_json(work / "install_plan.json", plan)
    print(json.dumps({"work_dir": str(work), "replacement_counts": plan["replacement_counts"], "audit": "PASS", "installed": False}, indent=2), flush=True)


def install(work):
    """Publish only verified artifacts; restore original files on any failure."""
    work = work.resolve()
    plan = json.loads((work / "install_plan.json").read_text())
    assert_clean(json.loads((work / "after.json").read_text()))
    if (work / "installed.json").exists():
        raise FileExistsError("This repair has already been installed")
    for target, info in plan["originals"].items():
        if file_hash(target) != info["sha256"] or file_hash(info["backup"]) != info["sha256"]:
            raise RuntimeError(f"Original or backup changed since staging: {target}")
    for staged, sha in plan["staged_sha256"].items():
        if file_hash(staged) != sha:
            raise RuntimeError(f"Staged file changed: {staged}")
    pending = {}
    replaced = []
    try:
        for target, staged in plan["targets"].items():
            temporary = Path(target + ".repair-pending")
            if temporary.exists():
                raise FileExistsError(temporary)
            pending[target] = temporary
            with temporary.open("xb") as output, Path(staged).open("rb") as source:
                shutil.copyfileobj(source, output)
                output.flush()
                os.fsync(output.fileno())
            shutil.copymode(target, temporary)
            if file_hash(temporary) != plan["staged_sha256"][staged]:
                raise RuntimeError(f"Temporary file verification failed: {target}")
        for target, temporary in pending.items():
            os.replace(temporary, target)
            replaced.append(target)
        hashes = {target: file_hash(target) for target in plan["targets"]}
        for target, sha in hashes.items():
            if sha != plan["staged_sha256"][plan["targets"][target]]:
                raise RuntimeError(f"Installed file verification failed: {target}")
        write_json(work / "installed.json", {"sha256": hashes, "backup_dir": str(work / "backup")})
    except BaseException:
        for target in reversed(replaced):
            shutil.copy2(plan["originals"][target]["backup"], target)
        raise
    finally:
        for temporary in pending.values():
            if temporary.exists():
                temporary.unlink()
    print("Installed verified partition; original files retained in " + str(work / "backup"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--partition-seed", type=int, default=42)
    parser.add_argument("--raw", type=Path, default=Path("data/mimic_icu10_server500_clients.pkl"))
    parser.add_argument("--tokenized", type=Path, default=Path("data/mimic_icu10_server500_tokenized_512.pkl.bak"))
    parser.add_argument("--tokenized-main", type=Path, default=Path("data/mimic_icu10_server500_tokenized_512.pkl"))
    parser.add_argument("--raw-stats", type=Path, default=Path("data/mimic_icu10_server500_clients_stats.json"))
    parser.add_argument("--token-stats", type=Path, default=Path("data/mimic_icu10_server500_tokenized_512_stats.json"))
    parser.add_argument("--source-csv", type=Path, default=Path("ceshishujuji/LABELED_DATASET_CLIENTS_FULL.csv"))
    parser.add_argument("--tokenizer", default="FacebookAI/roberta-base")
    args = parser.parse_args()
    if args.install:
        install(args.output_dir)
    else:
        stage(args)


if __name__ == "__main__":
    main()
