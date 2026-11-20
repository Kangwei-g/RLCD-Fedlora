import importlib.util
import pickle
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "preprocessing" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cohort_labels_survivors_and_care_units(tmp_path):
    script = load_script("extract_cohort")
    pd.DataFrame(dict(SUBJECT_ID=[1, 2, 3, 4], HADM_ID=[11, 12, 13, 14],
                      DISCHTIME=["2100-01-01 12:00:00"] * 4,
                      HOSPITAL_EXPIRE_FLAG=[0, 0, 1, 0])).to_csv(tmp_path / "ADMISSIONS.csv", index=False)
    pd.DataFrame(dict(SUBJECT_ID=[1, 2, 3, 4], DOD=["2100-01-02", None, "2100-01-02", "2102-01-01"])).to_csv(tmp_path / "PATIENTS.csv", index=False)
    pd.DataFrame(dict(HADM_ID=[11, 11, 12], FIRST_CAREUNIT=["CCU", "MICU", "SICU"])).to_csv(tmp_path / "ICUSTAYS.csv", index=False)
    pd.DataFrame(dict(ROW_ID=[101, 102, 103, 104, 105], SUBJECT_ID=[1, 2, 3, 4, 1],
                      HADM_ID=[11, 12, 13, 14, 11],
                      CATEGORY=["Discharge summary"] * 4 + ["Other"],
                      TEXT=["synthetic note"] * 5)).to_csv(tmp_path / "NOTEEVENTS.csv", index=False)
    out = tmp_path / "cohort.csv"
    assert script.extract(tmp_path, out, chunksize=2) == 3
    result = pd.read_csv(out)
    assert result["1_YEAR_MORTALITY"].tolist() == [1, 0, 0]
    assert result["FIRST_CAREUNIT"].tolist() == ["CCU", "SICU", "Non-ICU"]
    with pytest.raises(FileExistsError):
        script.extract(tmp_path, out)


def test_repair_requires_patient_and_admission_disjointness():
    script = load_script("repair_icu_partition")
    def row(i, patient):
        return dict(ROW_ID=i, SUBJECT_ID=patient, HADM_ID=i, text_key=str(i), token_key=str(i),
                    FIRST_CAREUNIT="CCU", label=1, text="synthetic")
    groups = {"client:CCU_1500:test": [row(1, 10)], "client:CCU_1500:train": [row(2, 10)]}
    fixed, replacements, _ = script.select_replacements(groups, [row(3, 20)],
                                                        lambda text: {"token_key": "new"}, seed=0)
    assert len(replacements) == 1
    assert not script.mark_conflicts(fixed)
    assert fixed["client:CCU_1500:train"][0]["SUBJECT_ID"] == 20
    with pytest.raises(ValueError, match="No unused"):
        script.select_replacements(groups, [], lambda text: {}, seed=0)


def test_complete_preparation_with_synthetic_rows_and_tokenizer(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "preprocessing"))
    script = load_script("prepare_data")
    groups, source_rows = {}, []
    names = ["client:%s_%d:%s" % (u, size, role)
             for u in script.ICU_UNITS for size in (1000, 1500)
             for role in ("train", "vali", "test")]
    names += ["server:server_vali", "server:server_test"]
    for i, name in enumerate(names, 1):
        unit = name.split(":")[1].split("_")[0] if name.startswith("client:") else "CCU"
        row = dict(ROW_ID=i, SUBJECT_ID=i, HADM_ID=i, text="synthetic %d" % i,
                   label=i % 2, FIRST_CAREUNIT=unit)
        groups[name] = pd.DataFrame([row])
        source_rows.append(row)
    source = tmp_path / "source.csv"
    pd.DataFrame(source_rows).rename(columns={"text": "TEXT", "label": "1_YEAR_MORTALITY"}).to_csv(source, index=False)
    monkeypatch.setattr(script, "replay_partition_from_source", lambda *args: groups)

    class Tokenizer:
        def num_special_tokens_to_add(self, **kwargs):
            return 2
        def __call__(self, texts, **kwargs):
            return {"input_ids": [[10 + int(t.split()[-1])] for t in texts]}
        def prepare_for_model(self, ids, max_length, **kwargs):
            content = [0] + ids + [2]
            return dict(input_ids=content + [1] * (max_length - len(content)),
                        attention_mask=[1] * len(content) + [0] * (max_length - len(content)))

    from transformers import RobertaTokenizerFast
    monkeypatch.setattr(RobertaTokenizerFast, "from_pretrained", lambda *a, **k: Tokenizer())
    output = tmp_path / "prepared"
    args = SimpleNamespace(input_file=source, output_dir=output, roberta_model=tmp_path,
                           partition_seed=42, repair_seed=20261106)
    script.build(args)
    with (output / "raw.pkl").open("rb") as stream:
        raw = pickle.load(stream)
    with (output / "roberta.pkl").open("rb") as stream:
        tokenized = pickle.load(stream)
    assert len(raw["train"]) == len(tokenized["train"]) == 10
    assert len(tokenized["train"][0][0]["input_ids"]) == 512
    assert raw["vali"][0]["label"] == tokenized["vali"][0]["label"]
    assert (output / "audit.json").is_file()
    with pytest.raises(FileExistsError):
        script.build(args)
