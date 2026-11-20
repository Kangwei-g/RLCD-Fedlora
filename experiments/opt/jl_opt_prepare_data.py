#!/usr/bin/env python3
"""Build a separate CPU-only OPT cache from a trusted, repaired raw pickle.

Required CLI: --model LOCAL_DIR_OR_CACHED_ID --input-file RAW --output-file NEW.
Optional: --revision REV --cache-dir DIR --batch-size 128 --stats-file NEW_JSON
          --source-file READ_ONLY_FILE (repeatable).
The default audit path is NEW.audit.json. No destination may already exist.
Pickle inputs must be trusted: unpickling can execute code. This is an identity
audit, not a new patient-overlap audit or a repair of the supplied partition.
"""

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile


MAX_LENGTH = 512
FRONT_TOKENS = BACK_TOKENS = 256
MODEL_ID = "facebook/opt-125m"
CLIENT_SPLITS = ("train", "vali", "test")
SERVER_SPLITS = ("server_vali", "server_test")
TEXT_SHA = "opt_text_sha256"
SOURCE_INDEX = "opt_source_index"
GENERATED_KEYS = {
    "tokenized", "tokenizer_model", "max_length", "tokenization_strategy",
    "encode_device", "encode_batch_size", "opt_preparation",
    "padding_side", "pad_token_id", "tokenizer_files_sha256", "source_sha256",
}
TOKENIZER_FILES = {
    "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
    "merges.txt", "special_tokens_map.json", "added_tokens.json",
    "tokenizer.model", "chat_template.jinja", "jl_download_manifest.json",
}


def json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def text_sha256(text):
    """Fingerprint exact UTF-8 text; do not strip, normalize, or retain it."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_state(path):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in fields):
        raise RuntimeError("A protected file changed while being hashed")
    return {"path": str(path), "sha256": digest.hexdigest(),
            **{key: getattr(after, key) for key in fields}}


def assert_files_unchanged(states):
    for path, expected in states.items():
        if file_state(path) != expected:
            raise RuntimeError("A protected raw/source/tokenizer file changed")


def configure_tokenizer(tokenizer):
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None or tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a pad token nor a usable EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer could not assign EOS as padding")
    return tokenizer


def load_tokenizer(model, revision=None, cache_dir=None):
    # Import lazily so offline fake-tokenizer tests need no ML dependencies.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model), revision=revision, cache_dir=cache_dir,
        local_files_only=True, trust_remote_code=False, use_fast=False,
    )
    # The slow tokenizer's Python builder preserves OPT's BOS for head/tail IDs.
    if type(tokenizer).__name__ != "GPT2Tokenizer":
        raise ValueError("Expected OPT's GPT2 tokenizer family")
    if tokenizer.pad_token_id != 1 or tokenizer.eos_token_id != 2:
        raise ValueError("OPT requires its original distinct PAD=1 and EOS=2")
    return configure_tokenizer(tokenizer)


def tokenizer_source_files(model, revision=None, cache_dir=None):
    directory = Path(model).expanduser()
    if directory.is_dir():
        files = {name: directory / name for name in sorted(TOKENIZER_FILES)
                 if (directory / name).is_file()}
    else:
        if str(model) != MODEL_ID:
            raise ValueError("Use a local OPT directory or the canonical OPT model ID")
        from huggingface_hub import try_to_load_from_cache

        files = {}
        for name in sorted(TOKENIZER_FILES):
            cached = try_to_load_from_cache(str(model), name, revision=revision, cache_dir=cache_dir)
            if isinstance(cached, str) and Path(cached).is_file():
                files[name] = Path(cached)
    if not {"tokenizer.json", "vocab.json", "tokenizer.model"}.intersection(files):
        raise ValueError("No local tokenizer vocabulary files found; download separately first")
    return files


def validate_opt_cache(cache, *, tokenizer_files_sha256=None, source_sha256=None,
                        pad_token_id=None):
    """Preflight metadata gate; optional expected values pin the selected assets.

    This does not replay tokenization or unpickle a path. Pass the loaded cache.
    RoBERTa caches and legacy caches without OPT provenance are rejected.
    """
    expected = {"tokenized": True, "tokenizer_model": MODEL_ID,
                "max_length": MAX_LENGTH, "padding_side": "right"}
    if not isinstance(cache, dict) or any(cache.get(key) != value for key, value in expected.items()):
        raise ValueError("Not a compatible OPT tokenized cache (RoBERTa is not accepted)")
    if cache["tokenized"] is not True:
        raise ValueError("tokenized must be True")
    pad = cache.get("pad_token_id")
    if not isinstance(pad, int) or isinstance(pad, bool) or pad < 0:
        raise ValueError("Missing or invalid OPT pad_token_id")

    def valid_sha(value):
        return (isinstance(value, str) and len(value) == 64
                and all(character in "0123456789abcdef" for character in value))

    hashes = cache.get("tokenizer_files_sha256")
    if (not isinstance(hashes, dict) or not hashes
            or not all(isinstance(name, str) and valid_sha(sha) for name, sha in hashes.items())
            or not valid_sha(cache.get("source_sha256"))):
        raise ValueError("Missing or invalid OPT tokenizer/source SHA-256 metadata")
    for supplied, actual in ((tokenizer_files_sha256, hashes),
                             (source_sha256, cache["source_sha256"]), (pad_token_id, pad)):
        if supplied is not None and supplied != actual:
            raise ValueError("OPT cache does not match the expected tokenizer/source")
    return True


def head_tail_budget(tokenizer, max_length=MAX_LENGTH,
                     front_tokens=FRONT_TOKENS, back_tokens=BACK_TOKENS):
    if max_length <= 0 or min(front_tokens, back_tokens) < 0:
        raise ValueError("Invalid length or negative head/tail budget")
    special = tokenizer.num_special_tokens_to_add(pair=False)
    budget = max_length - special
    if special < 0 or budget <= 0 or front_tokens + back_tokens <= 0:
        raise ValueError("No usable content budget")
    front, back = front_tokens, back_tokens
    if front + back > budget:
        front = max(0, min(budget, int(round(front * budget / (front + back)))))
        back = budget - front
    return front, back, special, budget


def encode_texts(texts, tokenizer):
    front, back, special, budget = head_tail_budget(tokenizer)
    try:
        encoded = tokenizer(texts, add_special_tokens=False,
                            truncation=False, padding=False)["input_ids"]
        if len(encoded) != len(texts):
            raise ValueError("Tokenizer batch size mismatch")
        ids, masks, lengths = [], [], []
        for raw_ids in encoded:
            raw_ids = list(raw_ids)
            lengths.append(len(raw_ids))
            kept = raw_ids
            if len(kept) > budget:
                kept = raw_ids[:front] + (raw_ids[-back:] if back else [])
            sequence = list(tokenizer.build_inputs_with_special_tokens(kept))
            if len(sequence) != len(kept) + special or len(sequence) > MAX_LENGTH:
                raise ValueError("Tokenizer special-token budget mismatch")
            padding = MAX_LENGTH - len(sequence)
            ids.append(sequence + [tokenizer.pad_token_id] * padding)
            masks.append([1] * len(sequence) + [0] * padding)
        return ids, masks, lengths
    except Exception:
        # Third-party exceptions can include clinical text; never propagate it.
        raise RuntimeError("Tokenizer batch failed; sample text omitted") from None


def audit_split(raw, tokenized):
    raw, tokenized = raw.with_format(None), tokenized.with_format(None)
    if len(raw) != len(tokenized) or "text" in tokenized.column_names:
        raise ValueError("Sample count changed or raw text retained")
    preserved = [key for key in raw.column_names if key != "text"]
    for key in preserved:
        if list(raw[key]) != list(tokenized[key]) or raw.features[key] != tokenized.features[key]:
            raise ValueError("Label/identifier values, order, or schema changed")
    hashes = [text_sha256(text) for text in raw["text"]]
    indices = list(range(len(raw)))
    if list(tokenized[TEXT_SHA]) != hashes or list(tokenized[SOURCE_INDEX]) != indices:
        raise ValueError("Sample fingerprint/order audit failed")
    labels = list(raw["label"])
    identity = [[index, digest, label] for index, digest, label in zip(indices, hashes, labels)]
    return {
        "total_examples": len(raw),
        "label_counts": dict(sorted(Counter(str(label) for label in labels).items())),
        "labels_sha256": hashlib.sha256(json_bytes(labels)).hexdigest(),
        "ordered_identity_sha256": hashlib.sha256(json_bytes(identity)).hexdigest(),
        "preserved_columns": preserved, "counts_equal": True, "labels_equal": True,
        "sample_order_equal": True, "identifiers_equal": True,
    }


def tokenize_split(dataset, tokenizer, batch_size=128):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    dataset = dataset.with_format(None)
    columns = set(dataset.column_names)
    if not {"text", "label"}.issubset(columns):
        raise ValueError("Each raw split must contain text and label")
    if columns.intersection({"input_ids", "attention_mask", TEXT_SHA, SOURCE_INDEX}):
        raise ValueError("Raw split contains reserved output columns")
    configure_tokenizer(tokenizer)
    front, back, special, budget = head_tail_budget(tokenizer)
    ids, masks, hashes = [], [], []
    original_total = effective_total = truncated = longest = 0
    for start in range(0, len(dataset), batch_size):
        texts = dataset[start:start + batch_size]["text"]
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("Raw text must contain strings, not nulls or coerced values")
        batch_ids, batch_masks, lengths = encode_texts(texts, tokenizer)
        ids.extend(batch_ids)
        masks.extend(batch_masks)
        hashes.extend(text_sha256(text) for text in texts)
        original_total += sum(lengths)
        effective_total += sum(sum(mask) - special for mask in batch_masks)
        truncated += sum(length > budget for length in lengths)
        longest = max(longest, max(lengths, default=0))
    # Materialize logical row order BEFORE adding columns: selected/shuffled
    # datasets can otherwise apply their old indices to the newly added values.
    # This also detaches memory-mapped input tables for a standalone pickle.
    output = dataset.remove_columns("text")
    output = type(output).from_dict(output.to_dict(), features=output.features)
    for key, values in (("input_ids", ids), ("attention_mask", masks),
                        (TEXT_SHA, hashes), (SOURCE_INDEX, list(range(len(dataset))))):
        output = output.add_column(key, values)
    stats = audit_split(dataset, output)
    stats.update({
        "requested_front_tokens": FRONT_TOKENS, "requested_back_tokens": BACK_TOKENS,
        "effective_front_tokens": front, "effective_back_tokens": back,
        "num_special_tokens": special, "content_budget": budget,
        "truncated_examples": truncated, "max_original_token_length": longest,
        "avg_original_token_length": original_total / len(dataset) if len(dataset) else 0.0,
        "avg_effective_content_tokens": effective_total / len(dataset) if len(dataset) else 0.0,
    })
    return output, stats


def iter_splits(fed):
    for split in CLIENT_SPLITS:
        for client_id, dataset in enumerate(fed[split]):
            yield split, client_id, dataset
    for split in SERVER_SPLITS:
        if split in fed:
            yield split, None, fed[split]


def validate_raw(fed):
    if not isinstance(fed, dict) or fed.get("tokenized"):
        raise ValueError("Expected a raw partition dictionary")
    if not isinstance(fed.get("partition_repair"), dict) or not fed["partition_repair"]:
        raise ValueError("Expected repaired raw input with partition_repair metadata")
    if "server_vali" not in fed:
        raise ValueError("server_vali is required")
    for split in CLIENT_SPLITS:
        if not isinstance(fed.get(split), (list, tuple)) or len(fed[split]) != 10:
            raise ValueError("All train/vali/test splits must contain exactly ten clients")
    if (set(fed) & GENERATED_KEYS) - {"tokenized"}:
        raise ValueError("Raw metadata collides with generated tokenization metadata")


def tokenizer_metadata(tokenizer, model, revision, cache_dir):
    vocab_hash = hashlib.sha256(json_bytes(tokenizer.get_vocab())).hexdigest()
    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_hash = hashlib.sha256(backend.to_str().encode("utf-8")).hexdigest() if backend else None
    special = {name: getattr(tokenizer, name, None) for name in (
        "pad_token", "pad_token_id", "eos_token", "eos_token_id", "bos_token",
        "bos_token_id", "unk_token", "unk_token_id", "all_special_tokens", "all_special_ids",
    )}
    identity = {"class": type(tokenizer).__name__, "vocab_sha256": vocab_hash,
                "backend_sha256": backend_hash, "special_tokens": special,
                "padding_side": tokenizer.padding_side,
                "num_special_tokens": tokenizer.num_special_tokens_to_add(pair=False)}
    versions = {}
    for package in ("transformers", "tokenizers", "datasets"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {
        **identity, "effective_tokenizer_sha256": hashlib.sha256(json_bytes(identity)).hexdigest(),
        "model": str(model), "requested_revision": revision,
        "resolved_revision": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        "name_or_path": str(getattr(tokenizer, "name_or_path", model)),
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "cache_dir": str(cache_dir) if cache_dir else None,
        "local_files_only": True, "trust_remote_code": False,
        "chat_template_applied": False, "versions": versions,
    }


def audit_federation(raw, output):
    if any(len(raw[split]) != len(output[split]) for split in CLIENT_SPLITS):
        raise ValueError("Client counts changed")
    if any((split in raw) != (split in output) for split in SERVER_SPLITS):
        raise ValueError("Server split presence changed")
    for key in raw:
        if key not in (*CLIENT_SPLITS, *SERVER_SPLITS) and key not in GENERATED_KEYS:
            if pickle.dumps(raw[key], protocol=4) != pickle.dumps(output[key], protocol=4):
                raise ValueError("Client/partition metadata changed")
    report = {"metadata_equal": True, "all_equal": True, "splits": {}}
    for split, client_id, dataset in iter_splits(raw):
        other = output[split] if client_id is None else output[split][client_id]
        group = split if client_id is None else f"{split}/client_{client_id}"
        report["splits"][group] = audit_split(dataset, other)
    return report


def prepare_data(input_file, output_file, model, *, stats_file=None, revision=None,
                 cache_dir=None, batch_size=128, source_files=()):
    """Write a new pickle + JSON audit, or fail without replacing existing files."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    source = Path(input_file).expanduser().resolve(strict=True)
    output = Path(output_file).expanduser().absolute()
    stats_path = Path(stats_file).expanduser().absolute() if stats_file else Path(str(output) + ".audit.json")
    destinations = (output, stats_path)
    for path in destinations:
        if os.path.lexists(path):
            raise FileExistsError("Refusing to overwrite an existing output or audit path")
    if output.resolve() == stats_path.resolve():
        raise ValueError("Output pickle and audit JSON must be distinct")

    model_path = Path(model).expanduser()
    if model_path.is_dir():
        model = str(model_path.resolve())
    tokenizer_files = tokenizer_source_files(model, revision, cache_dir)
    protected = [source, Path(__file__).resolve(), *[Path(path).expanduser() for path in source_files]]
    protected.extend(tokenizer_files.values())
    # Keep the lookup path as well as file_state's resolved target: cached HF
    # assets can be symlinks, and retargeting one must invalidate the audit.
    states = {str(path.absolute()): file_state(path) for path in protected}
    tokenizer_hashes = {name: states[str(path.absolute())]["sha256"] for name, path in tokenizer_files.items()}
    with source.open("rb") as handle:
        raw = pickle.load(handle)
    validate_raw(raw)
    assert_files_unchanged(states)
    tokenizer = configure_tokenizer(load_tokenizer(model, revision, cache_dir))
    tok_meta = tokenizer_metadata(tokenizer, model, revision, cache_dir)
    result = dict(raw)
    for split in CLIENT_SPLITS:
        result[split] = []
    split_stats = {}
    for split, client_id, dataset in iter_splits(raw):
        encoded, counts = tokenize_split(dataset, tokenizer, batch_size)
        if client_id is None:
            result[split] = encoded
        else:
            result[split].append(encoded)
        group = split if client_id is None else f"{split}/client_{client_id}"
        split_stats[group] = counts
    for split in CLIENT_SPLITS:
        if isinstance(raw[split], tuple):
            result[split] = tuple(result[split])
    identity_audit = audit_federation(raw, result)
    provenance = {
        "schema_version": 1, "source": states[str(source)], "protected_files": states,
        "tokenized": True, "tokenizer_model": MODEL_ID,
        "padding_side": "right", "pad_token_id": tokenizer.pad_token_id,
        "tokenizer_files_sha256": tokenizer_hashes, "source_sha256": states[str(source)]["sha256"],
        "tokenizer": tok_meta, "max_length": MAX_LENGTH, "strategy": "head_tail",
        "requested_front_tokens": FRONT_TOKENS, "requested_back_tokens": BACK_TOKENS,
        "encode_device": "cpu", "batch_size": batch_size,
        "identity_columns": {"text_sha256": TEXT_SHA, "source_index": SOURCE_INDEX,
                             "index_scope": "zero-based within each original split/client",
                             "text_encoding": "exact UTF-8; no normalization"},
        "split_sizes": {
            **{split: [len(ds) for ds in raw[split]] for split in CLIENT_SPLITS},
            **{split: len(raw[split]) for split in SERVER_SPLITS if split in raw},
        },
        "split_tokenization_stats": split_stats, "audit": identity_audit,
        "protected_files_unchanged": True,
    }
    result.update(tokenized=True, tokenizer_model=MODEL_ID, max_length=MAX_LENGTH,
                  padding_side="right", pad_token_id=tokenizer.pad_token_id,
                  tokenizer_files_sha256=tokenizer_hashes, source_sha256=states[str(source)]["sha256"],
                  tokenization_strategy="head_tail", encode_device="cpu",
                  encode_batch_size=batch_size, opt_preparation=provenance)
    validate_opt_cache(result)
    assert_files_unchanged(states)

    # Stage privately, verify the pickle round trip, then publish with hard links.
    # Unlike replace()/rename(), link() cannot clobber a concurrently created file.
    with ExitStack() as stack:
        staged = []
        for path in destinations:
            path.parent.mkdir(parents=True, exist_ok=True)
            directory = stack.enter_context(tempfile.TemporaryDirectory(prefix=".jl_opt_", dir=path.parent))
            staged.append(Path(directory) / "artifact")
        with staged[0].open("xb") as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with staged[0].open("rb") as handle:
            restored = pickle.load(handle)
        validate_opt_cache(restored, tokenizer_files_sha256=tokenizer_hashes,
                            source_sha256=states[str(source)]["sha256"], pad_token_id=tokenizer.pad_token_id)
        if audit_federation(raw, restored) != identity_audit:
            raise ValueError("Serialized sample identity audit failed")
        del restored
        report = {**provenance, "serialization_identity_equal": True,
                  "output_file": str(output), "output_sha256": file_state(staged[0])["sha256"]}
        with staged[1].open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
            handle.write("\n")
        assert_files_unchanged(states)
        published = []
        try:
            for temporary, destination in zip(staged, destinations):
                os.link(temporary, destination)
                published.append((temporary, destination))
            assert_files_unchanged(states)
        except BaseException:
            for temporary, destination in reversed(published):
                if destination.exists() and os.path.samefile(temporary, destination):
                    destination.unlink()
            raise
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local OPT directory or already-cached model ID")
    parser.add_argument("--input-file", "--input_file", required=True, help="Trusted repaired raw pickle")
    parser.add_argument("--output-file", "--output_file", required=True, help="New tokenized pickle")
    parser.add_argument("--stats-file", "--stats_file", help="New audit JSON (default: OUTPUT.audit.json)")
    parser.add_argument("--revision", help="Locally cached tokenizer revision")
    parser.add_argument("--cache-dir", help="Existing Hugging Face cache location")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--source-file", action="append", default=[], help="Additional read-only file to hash")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        report = prepare_data(args.input_file, args.output_file, args.model,
                              stats_file=args.stats_file, revision=args.revision,
                              cache_dir=args.cache_dir, batch_size=args.batch_size,
                              source_files=args.source_file)
    except Exception as error:
        print(f"Preparation failed ({type(error).__name__}); sample text omitted.", file=sys.stderr)
        return 1
    print(json.dumps({"status": "verified", "output_file": report["output_file"],
                      "output_sha256": report["output_sha256"], "split_sizes": report["split_sizes"],
                      "audit_equal": report["audit"]["all_equal"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
