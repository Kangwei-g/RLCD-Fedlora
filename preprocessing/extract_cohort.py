#!/usr/bin/env python3
"""Extract the private discharge-summary cohort from authorized MIMIC tables."""
import argparse
from pathlib import Path


def table_path(root, name):
    for suffix in (".csv.gz", ".csv", ".CSV.gz", ".CSV"):
        path = root / (name + suffix)
        if path.is_file():
            return path
    raise FileNotFoundError("Missing MIMIC table: " + name)


def extract(root, output, chunksize=100000):
    import pandas as pd

    if output.exists():
        raise FileExistsError(output)
    admissions = pd.read_csv(table_path(root, "ADMISSIONS"), usecols=[
        "SUBJECT_ID", "HADM_ID", "DISCHTIME", "HOSPITAL_EXPIRE_FLAG"])
    patients = pd.read_csv(table_path(root, "PATIENTS"), usecols=["SUBJECT_ID", "DOD"])
    admissions = admissions[admissions["HOSPITAL_EXPIRE_FLAG"] == 0]
    labels = admissions.merge(patients, on="SUBJECT_ID", how="inner", validate="many_to_one")
    labels["DISCHTIME"] = pd.to_datetime(labels["DISCHTIME"]).dt.normalize()
    labels["DOD"] = pd.to_datetime(labels["DOD"])
    labels = labels.dropna(subset=["DISCHTIME"])
    days = (labels["DOD"] - labels["DISCHTIME"]).dt.days
    labels["1_YEAR_MORTALITY"] = ((days > 0) & (days <= 365)).astype(int)
    units = pd.read_csv(table_path(root, "ICUSTAYS"), usecols=["HADM_ID", "FIRST_CAREUNIT"])
    units = units.drop_duplicates("HADM_ID", keep="first")
    labels = labels.merge(units, on="HADM_ID", how="left", validate="many_to_one")
    labels["FIRST_CAREUNIT"] = labels["FIRST_CAREUNIT"].fillna("Non-ICU")
    columns = ["ROW_ID", "SUBJECT_ID", "HADM_ID", "TEXT", "1_YEAR_MORTALITY", "FIRST_CAREUNIT"]
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    # Exclusive creation prevents accidental replacement of an existing cohort.
    with output.open("x", newline="") as stream:
        pd.DataFrame(columns=columns).to_csv(stream, index=False)
        for chunk in pd.read_csv(table_path(root, "NOTEEVENTS"), chunksize=chunksize,
                                 usecols=["ROW_ID", "SUBJECT_ID", "HADM_ID", "CATEGORY", "TEXT"]):
            chunk = chunk[chunk["CATEGORY"] == "Discharge summary"].dropna(subset=["TEXT", "HADM_ID", "SUBJECT_ID"])
            merged = chunk.merge(labels[["SUBJECT_ID", "HADM_ID", "1_YEAR_MORTALITY", "FIRST_CAREUNIT"]],
                                 on=["SUBJECT_ID", "HADM_ID"], how="inner", validate="many_to_one")
            merged[columns].to_csv(stream, header=False, index=False)
            count += len(merged)
    if count == 0:
        raise ValueError("No eligible discharge summaries; inspect the input tables")
    return count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mimic-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    count = extract(args.mimic_dir, args.output)
    print("Wrote %d private cohort rows to %s" % (count, args.output))


if __name__ == "__main__":
    main()
