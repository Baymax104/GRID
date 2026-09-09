## 1. Diagnosis Config Identity

- [x] 1.1 Add `wandb_entity: ${user}` and `wandb_project: ${project}` to the official diagnosis test dataloader config without changing existing path or dataset fields.
- [x] 1.2 Extend Hydra composition coverage to assert that the diagnosis dataloader receives the experiment user/project identity and retains optional embedding semantics.

## 2. Setup-Time Artifact Resolution

- [x] 2.1 Resolve `semantic_id_path` in `DiagnosisDataModule.setup_stage` through the shared resolver with `field_name="semantic_id_path"` and explicit configured identity.
- [x] 2.2 Resolve non-null `embedding_path` with `field_name="embedding_path"`, skip resolution when it is null, and preserve fully qualified URI and local-path behavior.
- [x] 2.3 Pass only the resolved local paths into the existing `DiagnosisDataset` constructor while preserving stage idempotence and the Dataset keyed bundle contract.

## 3. Focused Regression Coverage

- [x] 3.1 Add DataModule unit tests that verify resolver arguments, `semantic_id`/`semantic_embedding` role inference, resolved-path forwarding, local-path bypass, and null embedding behavior.
- [x] 3.2 Add a focused setup-order test proving diagnosis references are registered before `WandbArtifactLineageCallback.setup` records them on the logger-owned run.
- [x] 3.3 Preserve and run existing diagnosis Dataset tests for keyed bundle loading, item-key alignment, SID views, optional embeddings, and Lightning test integration.

## 4. Verification

- [x] 4.1 Run focused pytest coverage for Tail-SID diagnosis, W&B identity composition, Artifact resolution, and W&B lineage callbacks.
- [x] 4.2 Run `uv run ruff check src tests`.
- [x] 4.3 Compose the official diagnosis experiment with local and short W&B-style overrides without executing a full experiment.
- [x] 4.4 Run `openspec validate fix-diagnosis-wandb-artifact-resolution --strict` and confirm all change artifacts remain apply-ready.
