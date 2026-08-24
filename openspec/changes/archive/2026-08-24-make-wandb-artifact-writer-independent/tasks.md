## 1. Writer Implementation

- [x] 1.1 Refactor `WandbArtifactWriter` to remove `source_path` and accept `output_dir`, `flush_frequency`, and `post_processing_functions`.
- [x] 1.2 Implement W&B writer batch buffering, shard flushing, rank 0 merge, and merged bundle path selection independent of `LocalPickleWriter`.
- [x] 1.3 Run post-processing functions on the W&B writer's own merged output before publishing.
- [x] 1.4 Preserve W&B run reuse/create/finish behavior without requiring W&B logger.

## 2. Configuration Migration

- [x] 2.1 Replace inference `wandb_artifact_writer.source_path` config with independent `output_dir`.
- [x] 2.2 Add W&B writer post-processing config for semantic ID inference artifacts where local writer applies the same processing.
- [x] 2.3 Ensure configs no longer require local pickle writer for W&B artifact publication.

## 3. Tests and Validation

- [x] 3.1 Update W&B artifact writer tests for independent prediction writing and remove old source-path publisher expectations.
- [x] 3.2 Add coverage for using W&B writer without local writer.
- [x] 3.3 Add coverage for local writer and W&B writer coexisting with separate directories.
- [x] 3.4 Run focused writer tests, ruff, and `openspec validate make-wandb-artifact-writer-independent --strict`.
