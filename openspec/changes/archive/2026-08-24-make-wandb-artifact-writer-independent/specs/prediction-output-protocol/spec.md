## ADDED Requirements

### Requirement: WandbArtifactWriter SHALL be an independent prediction writer

`WandbArtifactWriter` SHALL consume `ModelOutput` directly during prediction and SHALL independently flush, merge, post-process, and publish prediction outputs as W&B Artifacts. It SHALL NOT require `LocalPickleWriter`, SHALL NOT read a `source_path` produced by another writer, and SHALL NOT expose a `source_path` configuration parameter.

#### Scenario: W&B writer handles prediction batch outputs directly
- **WHEN** Lightning finishes a prediction batch and `WandbArtifactWriter` is enabled
- **THEN** `WandbArtifactWriter` MUST consume the batch `ModelOutput` from `on_predict_batch_end`
- **THEN** `WandbArtifactWriter` MUST buffer and flush predictions using sample-count `flush_frequency` semantics

#### Scenario: W&B writer merges its own shards
- **WHEN** prediction completes
- **THEN** `WandbArtifactWriter` MUST flush rank-local buffered outputs
- **THEN** rank 0 MUST merge only the shard files in the W&B writer's own `output_dir`
- **THEN** rank 0 MUST save `merged_predictions_tensor.pt` in the W&B writer's own `output_dir`

#### Scenario: W&B writer does not depend on local writer output
- **WHEN** `LocalPickleWriter` is disabled and `WandbArtifactWriter` is enabled
- **THEN** prediction completion MUST NOT require `${paths.output_dir}/pickle/merged_predictions_tensor.pt`
- **THEN** W&B publishing MUST use the bundle produced by `WandbArtifactWriter`

#### Scenario: W&B writer and local writer can coexist
- **WHEN** `LocalPickleWriter` and `WandbArtifactWriter` are both enabled
- **THEN** each writer MUST write temporary shards and merged bundles under its own configured `output_dir`
- **THEN** neither writer MUST read, delete, or post-process the other writer's files

#### Scenario: W&B writer is logger independent
- **WHEN** no W&B logger created an active run
- **THEN** `WandbArtifactWriter` MUST be able to create a W&B run for artifact publishing
- **THEN** `WandbArtifactWriter` MUST only finish runs it created itself
