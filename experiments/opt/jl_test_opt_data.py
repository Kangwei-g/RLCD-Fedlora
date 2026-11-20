"""CPU-only offline tests; no transformers, datasets, model weights, or network."""

from contextlib import redirect_stderr, redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jl_opt_prepare_data as prepare


class FakeDataset:
    """Small immutable-column stand-in for the Dataset methods used here."""

    def __init__(self, columns, features=None):
        self.columns = copy.deepcopy(columns)
        self.column_names = list(columns)
        self.features = features or {key: "feature:" + key for key in columns}
        if len({len(values) for values in columns.values()}) > 1:
            raise ValueError("Unequal column lengths")

    def __len__(self):
        return len(next(iter(self.columns.values())))

    def __getitem__(self, key):
        if isinstance(key, str):
            return copy.deepcopy(self.columns[key])
        return {name: copy.deepcopy(values[key]) for name, values in self.columns.items()}

    def with_format(self, value):
        if value is not None:
            raise AssertionError("Only unformatted Python values are expected")
        return self

    def remove_columns(self, name):
        return FakeDataset({key: value for key, value in self.columns.items() if key != name},
                           {key: value for key, value in self.features.items() if key != name})

    def add_column(self, name, values):
        if name in self.column_names:
            raise ValueError("Duplicate column")
        return FakeDataset({**self.columns, name: values}, {**self.features, name: "feature:" + name})

    def to_dict(self):
        return copy.deepcopy(self.columns)

    @classmethod
    def from_dict(cls, columns, features=None):
        return cls(columns, features)


class FakeTokenizer:
    padding_side = "left"
    eos_token = "<eos>"
    eos_token_id = 9001
    bos_token = "<bos>"
    bos_token_id = 9000
    name_or_path = "fake-opt"
    is_fast = True
    init_kwargs = {"_commit_hash": "offline-test-revision"}

    def __init__(self, special=0, pad="<pad>"):
        self.special = special
        self.pad_token = pad
        self.calls = []

    @property
    def pad_token_id(self):
        return None if self.pad_token is None else (self.eos_token_id if self.pad_token == self.eos_token else 9002)

    def num_special_tokens_to_add(self, pair=False):
        assert pair is False
        return self.special

    def __call__(self, texts, **kwargs):
        assert kwargs == dict(add_special_tokens=False, truncation=False, padding=False)
        self.calls.append(len(texts))
        return {"input_ids": [[int(word) for word in text.split()] for text in texts]}

    def build_inputs_with_special_tokens(self, ids):
        if self.special == 0:
            return list(ids)
        if self.special == 1:
            return list(ids) + [self.eos_token_id]
        return [self.bos_token_id] + list(ids) + [self.eos_token_id] * (self.special - 1)

    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("Chat templates must never be used")

    def get_vocab(self):
        return {str(index): index for index in range(1024)}


def split(texts=("1 2", "3", "1 2"), labels=(1, 0, 0)):
    return FakeDataset({"text": list(texts), "label": list(labels),
                        "ROW_ID": list(range(100, 100 + len(texts))),
                        "SUBJECT_ID": [7] * len(texts),
                        "source_index": list(reversed(range(len(texts))))})


def federation(server_test=True):
    raw = {role: [split() for _ in range(10)] for role in prepare.CLIENT_SPLITS}
    raw.update(server_vali=split(), client_names=[f"client-{index}" for index in range(10)],
               care_units=["unit-a", "unit-b"], client_train_sizes=[3] * 10,
               partition_repair={"seed": 42, "source_sha256": "a" * 64},
               custom_metadata={"nested": ["keep", 7, None]}, tokenized=False)
    if server_test:
        raw["server_test"] = split((), ())
    return raw


class EncodingTests(unittest.TestCase):
    def test_special_budget_and_exact_head_tail(self):
        text = " ".join(str(index) for index in range(700))
        for special, front, back in ((0, 256, 256), (1, 256, 255), (2, 255, 255), (3, 254, 255)):
            with self.subTest(special=special):
                tokenizer = FakeTokenizer(special)
                self.assertEqual(prepare.head_tail_budget(tokenizer), (front, back, special, 512 - special))
                output, stats = prepare.tokenize_split(split((text,), (1,)), tokenizer)
                expected = tokenizer.build_inputs_with_special_tokens(list(range(front)) + list(range(700 - back, 700)))
                self.assertEqual(output["input_ids"][0], expected)
                self.assertEqual(output["attention_mask"][0], [1] * 512)
                self.assertEqual(stats["truncated_examples"], 1)
                self.assertEqual(stats["avg_effective_content_tokens"], 512 - special)

    def test_threshold_empty_short_and_right_padding(self):
        for special in (0, 1, 2):
            tokenizer = FakeTokenizer(special)
            budget = 512 - special
            texts = ("", "1 2", " ".join(map(str, range(budget))), " ".join(map(str, range(budget + 1))))
            output, stats = prepare.tokenize_split(split(texts, (0, 1, 0, 1)), tokenizer, batch_size=2)
            self.assertEqual(tokenizer.padding_side, "right")
            self.assertEqual(tokenizer.calls, [2, 2])
            self.assertEqual(stats["truncated_examples"], 1)
            for index, length in enumerate((special, special + 2, 512, 512)):
                self.assertEqual(len(output["input_ids"][index]), 512)
                self.assertEqual(output["attention_mask"][index], [1] * length + [0] * (512 - length))
                self.assertEqual(output["input_ids"][index][length:], [9002] * (512 - length))

    def test_eos_padding_is_masked_by_position_not_token_value(self):
        tokenizer = FakeTokenizer(1, pad=None)
        output, _ = prepare.tokenize_split(split(("9001",), (1,)), tokenizer)
        self.assertEqual(tokenizer.pad_token, tokenizer.eos_token)
        self.assertEqual(output["input_ids"][0], [9001] * 512)
        self.assertEqual(output["attention_mask"][0], [1, 1] + [0] * 510)
        existing = FakeTokenizer()
        prepare.configure_tokenizer(existing)
        self.assertEqual(existing.pad_token, "<pad>")
        broken = FakeTokenizer(pad=None)
        broken.eos_token = None
        with self.assertRaises(ValueError):
            prepare.configure_tokenizer(broken)

    def test_budget_validation_and_asymmetric_budget(self):
        tokenizer = FakeTokenizer(2)
        self.assertEqual(prepare.head_tail_budget(tokenizer, 512, 512, 0), (510, 0, 2, 510))
        self.assertEqual(prepare.head_tail_budget(tokenizer, 512, 0, 512), (0, 510, 2, 510))
        for length, front, back in ((2, 256, 256), (0, 1, 1), (512, -1, 1), (512, 0, 0)):
            with self.assertRaises(ValueError):
                prepare.head_tail_budget(tokenizer, length, front, back)

    def test_identity_duplicates_labels_extras_and_no_mutation(self):
        raw = split()
        before = pickle.dumps(raw)
        output, audit = prepare.tokenize_split(raw, FakeTokenizer(), batch_size=1)
        self.assertEqual(before, pickle.dumps(raw))
        self.assertNotIn("text", output.column_names)
        for name in ("label", "ROW_ID", "SUBJECT_ID", "source_index"):
            self.assertEqual(raw[name], output[name])
            self.assertEqual(raw.features[name], output.features[name])
        self.assertEqual(output[prepare.SOURCE_INDEX], [0, 1, 2])
        self.assertEqual(output[prepare.TEXT_SHA], [prepare.text_sha256(text) for text in raw["text"]])
        self.assertEqual(audit["label_counts"], {"0": 2, "1": 1})
        self.assertNotEqual(prepare.text_sha256("1 2"), prepare.text_sha256("1 2 "))
        self.assertEqual(prepare.text_sha256("\u00e9"), hashlib.sha256(b"\xc3\xa9").hexdigest())
        for name in ("label", "ROW_ID", prepare.TEXT_SHA, prepare.SOURCE_INDEX):
            damaged = copy.deepcopy(output)
            damaged.columns[name][0], damaged.columns[name][1] = damaged.columns[name][1], damaged.columns[name][0]
            with self.subTest(column=name), self.assertRaises(ValueError):
                prepare.audit_split(raw, damaged)

    def test_empty_dataset_and_invalid_columns(self):
        output, audit = prepare.tokenize_split(split((), ()), FakeTokenizer())
        self.assertEqual(len(output), 0)
        self.assertEqual(audit["avg_original_token_length"], 0.0)
        self.assertIn(prepare.TEXT_SHA, output.column_names)
        for raw in (split().remove_columns("label"), split().add_column(prepare.TEXT_SHA, ["x"] * 3),
                    split((None,), (0,))):
            with self.assertRaises(ValueError):
                prepare.tokenize_split(raw, FakeTokenizer())
        with self.assertRaises(ValueError):
            prepare.tokenize_split(split(), FakeTokenizer(), 0)

    def test_auto_tokenizer_loading_is_local_only(self):
        calls = []

        class GPT2Tokenizer(FakeTokenizer):
            eos_token_id = 2
            @property
            def pad_token_id(self):
                return 1

        def from_pretrained(*args, **kwargs):
            calls.append((args, kwargs))
            return GPT2Tokenizer(pad="<pad>")

        module = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=from_pretrained))
        with patch.dict("sys.modules", {"transformers": module}):
            tokenizer = prepare.load_tokenizer("cached-opt", "revision", "local-cache")
        self.assertEqual(calls, [(("cached-opt",), dict(revision="revision", cache_dir="local-cache",
                                                       local_files_only=True, trust_remote_code=False, use_fast=False))])
        self.assertEqual(tokenizer.padding_side, "right")
        self.assertEqual(tokenizer.pad_token, "<pad>")
        self.assertEqual(tokenizer.pad_token_id, 1)

    def test_opt_requires_eos_even_when_existing_pad_is_present(self):
        class GPT2Tokenizer(FakeTokenizer):
            eos_token = None
            eos_token_id = None

        module = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: GPT2Tokenizer()))
        with patch.dict("sys.modules", {"transformers": module}), self.assertRaisesRegex(ValueError, "EOS"):
            prepare.load_tokenizer("local-opt")

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "transformers not installed")
    def test_real_opt_bos_builder_matches_native_encoding(self):
        from transformers import GPT2Tokenizer

        with tempfile.TemporaryDirectory() as directory:
            vocab = Path(directory) / "vocab.json"
            merges = Path(directory) / "merges.txt"
            vocab.write_text(json.dumps({"<s>": 0, "<pad>": 1, "</s>": 2, "h": 3, "i": 4}))
            merges.write_text("#version: 0.2\n")
            tokenizer = GPT2Tokenizer(vocab_file=str(vocab), merges_file=str(merges),
                                      bos_token="</s>", eos_token="</s>", unk_token="</s>",
                                      pad_token="<pad>", add_bos_token=True)
            ids, masks, _ = prepare.encode_texts(["hi", "hi" * 400], tokenizer)
            native = tokenizer("hi")["input_ids"]
            self.assertEqual(native, [2, 3, 4])
            self.assertEqual(ids[0][:len(native)], native)
            self.assertEqual(ids[0][len(native):], [1] * (512 - len(native)))
            self.assertEqual(ids[1][0], 2)
            self.assertEqual(sum(masks[1]), 512)

    def test_non_opt_tokenizer_is_rejected(self):
        module = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: FakeTokenizer()))
        with patch.dict("sys.modules", {"transformers": module}), self.assertRaisesRegex(ValueError, "OPT"):
            prepare.load_tokenizer("local-roberta")

    def test_cached_tokenizer_files_are_resolved_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            vocabulary = Path(directory) / "tokenizer.json"
            vocabulary.write_text("{}")
            calls = []

            def from_cache(repo, name, **kwargs):
                calls.append((repo, name, kwargs))
                return str(vocabulary) if name == "tokenizer.json" else None

            with patch.dict("sys.modules", {"huggingface_hub": SimpleNamespace(try_to_load_from_cache=from_cache)}):
                files = prepare.tokenizer_source_files(prepare.MODEL_ID, "pinned", "cache")
            self.assertEqual(files, {"tokenizer.json": vocabulary})
            self.assertTrue(all(repo == prepare.MODEL_ID and kwargs == dict(revision="pinned", cache_dir="cache")
                                for repo, _, kwargs in calls))

    def test_tokenizer_errors_do_not_expose_sample_text(self):
        class BrokenTokenizer(FakeTokenizer):
            def __call__(self, texts, **kwargs):
                raise ValueError("sensitive clinical sample " + texts[0])

        with self.assertRaisesRegex(RuntimeError, "sample text omitted") as caught:
            prepare.tokenize_split(split(), BrokenTokenizer())
        self.assertNotIn("sensitive", str(caught.exception))


class FilePreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw_path = self.root / "repaired.pkl"
        self.output = self.root / "opt.pkl"
        self.stats = Path(str(self.output) + ".audit.json")
        self.source = self.root / "original-source.csv"
        self.source.write_bytes(b"source must stay unchanged\n")
        self.model = self.root / "local-model"
        self.model.mkdir()
        (self.model / "tokenizer_config.json").write_text('{"model_type":"opt"}')
        (self.model / "tokenizer.json").write_text('{"vocab":{}}')
        self.raw = federation()
        self.write_raw()
        self.loader = patch.object(prepare, "load_tokenizer", return_value=FakeTokenizer())
        self.mock_loader = self.loader.start()
        self.addCleanup(self.loader.stop)

    def write_raw(self):
        self.raw_path.write_bytes(pickle.dumps(self.raw))

    def run_prepare(self, **kwargs):
        return prepare.prepare_data(self.raw_path, self.output, str(self.model),
                                    source_files=[self.source], batch_size=2, **kwargs)

    def test_end_to_end_metadata_hashes_audit_and_read_only_inputs(self):
        paths = (self.raw_path, self.source, self.model / "tokenizer_config.json", Path(prepare.__file__))
        before = {path: prepare.file_state(path) for path in paths}
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            report = self.run_prepare()
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(before, {path: prepare.file_state(path) for path in paths})
        output = pickle.loads(self.output.read_bytes())
        self.assertEqual(report, json.loads(self.stats.read_text()))
        self.assertEqual(report["output_sha256"], prepare.file_state(self.output)["sha256"])
        self.assertEqual(report["source"]["sha256"], before[self.raw_path]["sha256"])
        self.assertEqual(report["split_sizes"]["train"], [3] * 10)
        self.assertEqual(len(report["audit"]["splits"]), 32)
        self.assertTrue(report["serialization_identity_equal"])
        self.assertTrue(report["protected_files_unchanged"])
        for key in ("client_names", "care_units", "client_train_sizes", "partition_repair", "custom_metadata"):
            self.assertEqual(output[key], self.raw[key])
        for _, _, dataset in prepare.iter_splits(output):
            self.assertNotIn("text", dataset.column_names)
        tokenizer = report["tokenizer"]
        self.assertTrue(tokenizer["local_files_only"])
        self.assertFalse(tokenizer["chat_template_applied"])
        self.assertEqual(tokenizer["padding_side"], "right")
        self.assertEqual(len(tokenizer["effective_tokenizer_sha256"]), 64)
        self.assertIn(str((self.model / "tokenizer_config.json").resolve()), report["protected_files"])
        expected = {"tokenized": True, "tokenizer_model": "facebook/opt-125m", "max_length": 512,
                    "padding_side": "right", "pad_token_id": 9002,
                    "source_sha256": before[self.raw_path]["sha256"],
                    "tokenizer_files_sha256": {name: prepare.file_state(path)["sha256"]
                                                for name, path in prepare.tokenizer_source_files(self.model).items()}}
        for key, value in expected.items():
            self.assertEqual(output[key], value)
            self.assertEqual(report[key], value)
        self.assertTrue(prepare.validate_opt_cache(output))
        self.assertNotIn('"1 2"', self.stats.read_text())
        self.assertEqual(list(self.root.glob(".jl_opt_*")), [])

    def test_preflight_rejects_roberta_legacy_and_mismatched_assets(self):
        self.run_prepare()
        output = pickle.loads(self.output.read_bytes())
        for key, value in (("tokenized", False), ("tokenizer_model", "roberta"),
                           ("padding_side", "left"), ("max_length", 256), ("pad_token_id", None),
                           ("tokenizer_files_sha256", {}), ("source_sha256", "not-a-hash")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                prepare.validate_opt_cache({**output, key: value})
        for key in ("padding_side", "pad_token_id", "tokenizer_files_sha256", "source_sha256"):
            legacy = dict(output)
            del legacy[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                prepare.validate_opt_cache(legacy)
        for expectations in (dict(pad_token_id=0), dict(source_sha256="0" * 64),
                             dict(tokenizer_files_sha256={"other": "0" * 64})):
            with self.assertRaises(ValueError):
                prepare.validate_opt_cache(output, **expectations)

    def test_server_test_optional_tuple_clients_and_no_extra_columns(self):
        del self.raw["server_test"]
        self.raw["train"] = tuple(FakeDataset({"text": ["1"], "label": [1]}) for _ in range(10))
        self.write_raw()
        report = self.run_prepare()
        output = pickle.loads(self.output.read_bytes())
        self.assertNotIn("server_test", output)
        self.assertIsInstance(output["train"], tuple)
        self.assertEqual(len(report["audit"]["splits"]), 31)
        self.assertEqual(output["train"][0][prepare.SOURCE_INDEX], [0])

    def test_existing_pickle_or_stats_or_dangling_link_is_never_overwritten(self):
        for existing in (self.output, self.stats):
            with self.subTest(path=existing):
                existing.write_bytes(b"untouched")
                with self.assertRaises(FileExistsError):
                    self.run_prepare()
                self.assertEqual(existing.read_bytes(), b"untouched")
                self.mock_loader.assert_not_called()
                existing.unlink()
        self.output.symlink_to(self.root / "missing")
        with self.assertRaises(FileExistsError):
            self.run_prepare()
        self.assertTrue(self.output.is_symlink())

    def test_raw_or_source_as_output_and_identical_destinations_refused(self):
        for target in (self.raw_path, self.source):
            before = target.read_bytes()
            with self.assertRaises(FileExistsError):
                prepare.prepare_data(self.raw_path, target, str(self.model))
            self.assertEqual(before, target.read_bytes())
        with self.assertRaises(ValueError):
            self.run_prepare(stats_file=self.output)

    def test_concurrent_destination_creation_never_clobbered(self):
        real_link = os.link

        def race(source, destination):
            if destination == self.stats:
                destination.write_bytes(b"other writer")
            return real_link(source, destination)

        with patch.object(prepare.os, "link", side_effect=race), self.assertRaises(FileExistsError):
            self.run_prepare()
        self.assertFalse(self.output.exists())
        self.assertEqual(self.stats.read_bytes(), b"other writer")

    def test_changes_to_protected_files_abort_without_publishing(self):
        for target in (self.raw_path, self.source, self.model / "tokenizer_config.json"):
            original = target.read_bytes()

            def tamper(*args, **kwargs):
                target.write_bytes(original + b" ")
                return FakeTokenizer()

            with self.subTest(path=target), patch.object(prepare, "load_tokenizer", side_effect=tamper):
                with self.assertRaisesRegex(RuntimeError, "protected"):
                    self.run_prepare()
                self.assertFalse(self.output.exists())
                self.assertFalse(self.stats.exists())
            target.write_bytes(original)

    def test_tokenizer_symlink_retarget_aborts(self):
        asset = self.model / "tokenizer_config.json"
        original = self.root / "original-config.json"
        alternate = self.root / "alternate-config.json"
        original.write_bytes(asset.read_bytes())
        alternate.write_bytes(asset.read_bytes())
        asset.unlink()
        asset.symlink_to(original)

        def retarget(*args, **kwargs):
            asset.unlink()
            asset.symlink_to(alternate)
            return FakeTokenizer()

        with patch.object(prepare, "load_tokenizer", side_effect=retarget):
            with self.assertRaisesRegex(RuntimeError, "protected"):
                self.run_prepare()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.stats.exists())

    def test_invalid_partitions_fail_before_tokenizer_load(self):
        cases = []
        for key in ("partition_repair", "server_vali"):
            raw = federation()
            del raw[key]
            cases.append(raw)
        raw = federation()
        raw["test"].pop()
        cases.append(raw)
        raw = federation()
        raw["tokenized"] = True
        cases.append(raw)
        for raw in cases:
            self.raw_path.write_bytes(pickle.dumps(raw))
            with self.assertRaises(ValueError):
                self.run_prepare()
            self.mock_loader.assert_not_called()
            self.assertFalse(self.output.exists())

    def test_cli_and_error_output_are_text_free(self):
        args = prepare.parse_args(["--model", "local", "--input_file", "raw", "--output_file", "new"])
        self.assertEqual(args.batch_size, 128)
        output, errors = io.StringIO(), io.StringIO()
        with patch.object(prepare, "prepare_data", side_effect=RuntimeError("secret clinical text")):
            with redirect_stdout(output), redirect_stderr(errors):
                code = prepare.main(["--model", "local", "--input-file", "raw", "--output-file", "new"])
        self.assertEqual(code, 1)
        self.assertNotIn("secret", errors.getvalue())
        self.assertEqual(output.getvalue(), "")


@unittest.skipUnless(importlib.util.find_spec("datasets"), "optional real datasets integration")
class HuggingFaceDatasetTests(unittest.TestCase):
    def test_actual_arrow_features_and_indices(self):
        from datasets import ClassLabel, Dataset, Features, Value

        features = Features({"text": Value("string"), "label": ClassLabel(names=["no", "yes"]),
                             "ROW_ID": Value("int64")})
        base = Dataset.from_dict({"text": ["1 2", "3", "4"], "label": [0, 1, 0], "ROW_ID": [11, 12, 13]},
                                 features=features)
        for order in ([2, 0, 1], [2, 0], [2, 2, 0], []):
            with self.subTest(order=order):
                raw = (base.select(order) if order else
                       Dataset.from_dict({key: [] for key in base.column_names}, features=features))
                before = raw.to_dict()
                output, audit = prepare.tokenize_split(raw, FakeTokenizer(), batch_size=2)
                self.assertEqual(list(output["ROW_ID"]), [11 + index for index in order])
                self.assertEqual(list(output["label"]), list(raw["label"]))
                self.assertEqual(list(output[prepare.SOURCE_INDEX]), list(range(len(order))))
                self.assertEqual(list(output[prepare.TEXT_SHA]), [prepare.text_sha256(text) for text in raw["text"]])
                for row, text in zip(output, raw["text"]):
                    tokens = [int(word) for word in text.split()]
                    self.assertEqual(row["input_ids"], tokens + [9002] * (512 - len(tokens)))
                self.assertEqual(output.features["label"], features["label"])
                self.assertEqual(output.cache_files, [])
                self.assertIsNone(output._indices)
                self.assertTrue(audit["sample_order_equal"])
                self.assertEqual(pickle.loads(pickle.dumps(output)).to_dict(), output.to_dict())
                self.assertEqual(raw.to_dict(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
