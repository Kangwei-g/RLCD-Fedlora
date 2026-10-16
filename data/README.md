# External Data Interface

RLCD-FedLoRA accepts one record per client with three disjoint splits:

```python
clients[client_id] = {
    "train": train_dataset,
    "validation": validation_dataset,
    "test": test_dataset,
}
```

Each dataset item must provide `input_ids`, `attention_mask`, and a binary
`label`. Raw text may be tokenized outside this repository. Patient identifiers
must never be used as model inputs or committed to this repository.

Required evaluation discipline:

1. Local model parameters are fitted on `train` only.
2. The classification threshold is selected on `validation` only.
3. That fixed threshold is applied to `test` for reporting.
4. Test metrics never affect DDQN rewards, layer scores, factor scores,
   checkpoint selection, or early stopping.

Before training, verify that patient or admission IDs do not overlap across
client splits. This audit should run as part of the dataset preparation
pipeline.
