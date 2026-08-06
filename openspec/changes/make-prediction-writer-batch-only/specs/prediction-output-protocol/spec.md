## ADDED Requirements

### Requirement: Prediction writer SHALL be batch-only and independent from Lightning batch index inference

`LocalPickleWriter` SHALL operate as a batch-only Lightning callback that consumes `predict_step` outputs directly. It SHALL NOT require Lightning `BasePredictionWriter` interval handling, and SHALL NOT depend on inferred `batch_indices` from the dataloader.

#### Scenario: writer handles prediction batch outputs directly
- **WHEN** Lightning finishes a prediction batch
- **THEN** `LocalPickleWriter` MUST consume the batch `ModelOutput` from `on_predict_batch_end`
- **THEN** `LocalPickleWriter` MUST buffer and flush predictions using the existing sample-count `flush_frequency` semantics

#### Scenario: writer does not request dataloader batch indices
- **WHEN** inference uses an `IterableDataset` through `DataloaderWithIterationRetry`
- **THEN** `LocalPickleWriter` MUST NOT trigger Lightning `BasePredictionWriter` batch index inference
- **THEN** inference MUST NOT emit a writer-induced warning about inability to infer batch indices from `DataloaderWithIterationRetry`

#### Scenario: writer exposes no epoch interval configuration
- **WHEN** maintainers inspect inference callback configs
- **THEN** configs MUST NOT declare `write_interval`
- **THEN** `LocalPickleWriter` MUST NOT expose an epoch or batch-and-epoch writing mode

#### Scenario: prediction completion still merges outputs
- **WHEN** prediction completes
- **THEN** rank-local buffered outputs MUST be flushed
- **THEN** rank 0 MUST merge pickle shards into `merged_predictions_tensor.pt`
- **THEN** configured post-processing functions MUST still run on rank 0 after merge
