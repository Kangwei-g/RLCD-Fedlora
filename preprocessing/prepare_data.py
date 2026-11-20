#!/usr/bin/env python3
"""Prepare private, disjoint ICU data and both model-specific token caches."""
import argparse
import json
from pathlib import Path
import pickle

from audit_icu_partition_overlap import (
    ICU_UNITS, audit_mode, digest, normalized_text, replay_partition_from_source, token_digest,
)
from repair_icu_partition import assert_clean, file_hash, select_replacements
from prepare_mimic_careunit_tokenized_512 import _encode_head_tail_texts


def build(args):
    import pandas as pd
    from datasets import Dataset
    from transformers import RobertaTokenizerFast

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = RobertaTokenizerFast.from_pretrained(str(args.roberta_model), local_files_only=True)

    def encode_one(text):
        ids, masks, _ = _encode_head_tail_texts([text], tokenizer, 512, 256, 256)
        row = dict(input_ids=ids[0], attention_mask=masks[0])
        row["token_key"] = token_digest(row)
        return row

    def enrich(row, tokenize=True):
        row = dict(row)
        for key in ("ROW_ID", "SUBJECT_ID", "HADM_ID", "label"):
            row[key] = int(row[key])
        row["text_key"] = digest(normalized_text(row["text"]))
        if tokenize:
            row.update(encode_one(row["text"]))
        return row

    frames = replay_partition_from_source(args.input_file, args.partition_seed)
    groups = {name: [enrich(row) for row in frame.to_dict("records")]
              for name, frame in frames.items()}
    columns = ["ROW_ID", "SUBJECT_ID", "HADM_ID", "TEXT", "1_YEAR_MORTALITY", "FIRST_CAREUNIT"]
    source = pd.read_csv(args.input_file, usecols=columns).dropna(subset=columns).copy()
    source["TEXT"] = source["TEXT"].astype(str).str.strip()
    source["FIRST_CAREUNIT"] = source["FIRST_CAREUNIT"].astype(str).str.strip()
    source["1_YEAR_MORTALITY"] = pd.to_numeric(source["1_YEAR_MORTALITY"], errors="coerce")
    source = source[source["1_YEAR_MORTALITY"].isin([0, 1]) & source["FIRST_CAREUNIT"].isin(ICU_UNITS) & source["TEXT"].ne("")]
    source = source.rename(columns={"TEXT": "text", "1_YEAR_MORTALITY": "label"})
    candidates = [enrich(row, tokenize=False) for row in source.to_dict("records")]
    groups, replacements, _ = select_replacements(groups, candidates, encode_one, args.repair_seed)
    names = [unit + "_" + str(size) for unit in ICU_UNITS for size in (1000, 1500)]
    metadata = dict(client_names=names, care_units=list(ICU_UNITS), partition_scheme="icu10_repaired",
                    partition_seed=args.partition_seed, repair_seed=args.repair_seed,
                    source_sha256=file_hash(args.input_file))
    raw, tokenized = dict(metadata), dict(metadata)
    for role in ("train", "vali", "test"):
        raw[role], tokenized[role] = [], []
        for name in names:
            rows = groups["client:" + name + ":" + role]
            raw[role].append(Dataset.from_dict({key: [r[key] for r in rows] for key in ("text", "label")}))
            tokenized[role].append(Dataset.from_dict({key: [r[key] for r in rows] for key in ("input_ids", "attention_mask", "label")}))
    for role in ("server_vali", "server_test"):
        rows = groups["server:" + role]
        raw[role] = Dataset.from_dict({key: [r[key] for r in rows] for key in ("text", "label")})
        tokenized[role] = Dataset.from_dict({key: [r[key] for r in rows] for key in ("input_ids", "attention_mask", "label")})
    tokenized.update(tokenized=True, tokenizer_model="roberta", max_length=512, tokenization_strategy="head_tail")
    frames = {name: pd.DataFrame(rows) for name, rows in groups.items()}
    audits = {mode: audit_mode(raw, tokenized, frames, mode) for mode in ("icu10", "icu5")}
    assert_clean(audits)
    for name, data in (("raw.pkl", raw), ("roberta.pkl", tokenized)):
        with (output / name).open("xb") as stream:
            pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    # These identifier-bearing files are private audit artifacts, never repository inputs.
    manifest = [dict(group=name, index=i, **{key: row[key] for key in ("ROW_ID", "HADM_ID", "SUBJECT_ID", "label", "FIRST_CAREUNIT")})
                for name, rows in groups.items() for i, row in enumerate(rows)]
    pd.DataFrame(manifest).to_csv(output / "partition_manifest.csv", index=False)
    (output / "audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    (output / "preparation.json").write_text(json.dumps(dict(metadata, replacements=len(replacements)), indent=2) + "\n")
    print("Created private raw.pkl and roberta.pkl; overlap audits passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True, help="Authorized labeled cohort CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="New private output directory")
    parser.add_argument("--roberta-model", type=Path, required=True)
    parser.add_argument("--partition-seed", type=int, default=42)
    parser.add_argument("--repair-seed", type=int, default=20261106)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
