# Data are not distributed

See [preprocessing instructions](../preprocessing/README.md) for authorized
local cohort extraction, disjoint splits, and model-specific token caches.

The cache is a trusted local Python pickle containing `train`, `vali`, and
`test` lists of ten Hugging Face Dataset objects, one `server_vali` Dataset,
and `client_names`. Tokenized rows contain `input_ids`, `attention_mask`, and
binary `label`. The primary setting selects the five names ending in `_1500`;
the supplementary setting retains all ten clients. OPT additionally requires
tokenizer/source provenance and its identity audit.

Server feedback data are separate from the client validation splits used for
final checkpoint/threshold selection. The standard runner evaluates client
test data only after that selection.
