# Private data preparation

All files must remain under the user's own MIMIC authorization. Do not commit
preprocessing outputs, including identifier-bearing manifests.

## Cohort

`extract_cohort.py` accepts authorized NOTEEVENTS, ADMISSIONS, PATIENTS, and
ICUSTAYS CSV or CSV.GZ files. It retains discharge summaries for admissions
with `HOSPITAL_EXPIRE_FLAG=0`, normalizes discharge dates, and labels documented
death 1..365 days after discharge as positive. Missing or outside-window death
dates map to 0 under this construction; this is not a censoring adjustment.
The first ICU record per admission in the supplied file determines care unit,
matching the source rule. Preserve source ordering for reproducibility.

The prepared CSV contract is `ROW_ID`, `SUBJECT_ID`, `HADM_ID`, `TEXT`,
`1_YEAR_MORTALITY` (0/1), and `FIRST_CAREUNIT`. Existing authorized prepared
cohorts with these fields can be passed directly to `prepare_data.py`.

## Partition

Partition seed 42 assigns a 1,000-example and a 1,500-example client for each
of CCU, CSRU, MICU, SICU, and TSICU, followed by stratified 80/10/10 splits.
The larger five form (5,5); (10,8) uses all ten without changing those splits.

The source procedure reserves 100 records per ICU in a server pool, then
stratifies these 500 records into 100 server validation records and 400 unused
server test records. Only `server_vali` supplies training feedback. The
released runner does not use `server_test`. The final 100 server validation
records need not contain exactly 20 from every ICU after stratification.

The repair pass keeps patients in one client/split and removes duplicate
admissions, normalized texts, and token sequences. Conflicts are replaced by
previously unused patients of the same label and ICU type, preserving split
counts. Repair seed 20261106 is separate from model seeds. Distinct admissions
of one patient may remain in the same split. Exhausted replacement candidates
cause a failure rather than a relaxation of disjointness.

## Outputs

- `raw.pkl`: repaired text/label splits shared by the two models.
- `roberta.pkl`: head/tail token cache, length 512 including special tokens.
- `partition_manifest.csv`: private row provenance for overlap auditing.
- `audit.json`, `preparation.json`: overlap checks and source checksum.

RoBERTa retains up to 255 head and 255 tail tokens plus two special tokens.
OPT preparation uses its own tokenizer and BOS/padding conventions, verifies
sample identity against `raw.pkl`, and writes `opt.pkl.audit.json`. Download
models before tokenization so the offline checks can succeed.

Source ordering and preprocessing library versions affect exact splits.
This release uses datasets 3.1.0 for serialization; regenerate caches rather
than relying on compatibility with an old pickle. Never load untrusted pickles.
