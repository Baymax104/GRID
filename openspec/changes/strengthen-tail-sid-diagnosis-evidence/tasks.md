## 1. Evidence Configuration and Result Contract

- [x] 1.1 Add Hydra configuration for primary Tail ratio, priority multipliers, semantic reference sampling, damage components, bootstrap, frequency bins, pair-output limits, sensitivity grids, optional `recommendation_output_path`, and evidence output directory.
- [x] 1.2 Define typed diagnosis evidence containers for summary/verdict metadata and group, item, prefix, harmful-pair, sensitivity, and optional recommendation rows without introducing pandas/pyarrow dependencies.
- [x] 1.3 Refactor the diagnosis computation path so one complete `DiagnosisEvidence` is built once per official test batch and both scalar metrics and structured output are derived from that object.
- [x] 1.4 Preserve existing W&B structural/semantic/damage/prefix summary keys where their meanings remain compatible and add schema/version metadata for changed or new evidence fields.
- [x] 1.5 Define a domain-neutral structured analysis payload protocol in common infrastructure and a diagnosis-owned adapter from `DiagnosisEvidence`, with no common/data import of the Tail-SID diagnosis package.

## 2. Ungated Group and Frequency-Asymmetric Evidence

- [x] 2.1 Rename and separate ungated `raw_damage` from frequency-aware `priority_score`, recording component normalization, clamp, weight, degeneracy, and group multiplier metadata.
- [x] 2.2 Compute the same collision, strict near-collision, prefix-density, suffix-weakness, last-step-burden, semantic, and raw-damage fields for Head, Mid, Tail, and Tail-Cold.
- [x] 2.3 Add per-group support, average training frequency, mean, and configured quantiles while representing empty groups explicitly rather than as observed zero risk.
- [x] 2.4 Add typed Tail-Head, Tail-Mid, Tail-Tail, and Tail-Cold strict-overlap/full-collision counts plus head-dominated bucket, Tail isolation deficit, and Tail-to-Head pressure fields.
- [x] 2.5 Ensure full item/group aggregates are computed before harmful-pair output truncation and remain independent of output limits.

## 3. Semantic Compatibility Evidence

- [x] 3.1 Implement deterministic global random item-pair cosine sampling and the configurable primary similarity quantile/threshold with complete sample metadata.
- [x] 3.2 Compute `semantic_mismatch_global` for strict deep-overlap pairs and retain the existing within-bucket quantile calculation under a distinct `bucket_relative_semantic_outlier` field.
- [x] 3.3 Separate semantic and harmful-overlap aggregates by partner frequency group and record semantic evidence as unavailable when embeddings are absent.
- [x] 3.4 Add deterministic per-item/global harmful-pair ranking and limits, including candidate, retained, and truncated counts.

## 4. Recommendation Outcome Input and Keyed Alignment

- [x] 4.1 Extend the official diagnosis config, dataloader config, and root script with optional `recommendation_output_path` while preserving `--dry-run`, both notes forms, quoting, and trailing Hydra override precedence.
- [x] 4.2 Resolve non-null recommendation references during `DiagnosisDataModule.setup_stage` with `field_name="recommendation_output_path"`, explicit experiment identity, local-path compatibility, and pre-test W&B lineage registration.
- [x] 4.3 Read testing labels and keyed TIGER generated-SID bundles, join them by user key, and validate duplicate, missing, unknown, and shape cases without relying on row order.
- [x] 4.4 Compute configured user-level hit@K, rank, and NDCG contribution values and aggregate label support and outcomes by label item.
- [x] 4.5 Preserve pure SID diagnosis when the recommendation input is null and mark recommendation evidence/verdict dimensions as unavailable.

## 5. Statistical Evidence, Sensitivity, and Verdict

- [x] 5.1 Implement seeded bootstrap Tail-versus-Head absolute differences, defined ratios, confidence intervals, and support counts for configured primary raw components.
- [x] 5.2 Implement overall and Tail-only raw-risk/outcome association summaries using item label support and deterministic rank correlation calculations.
- [x] 5.3 Implement `log1p(freq_train)` frequency-bin matched low/high-damage outcome comparisons with configurable minimum bin support.
- [x] 5.4 Execute configured Tail-ratio, damage-component, and semantic-threshold sensitivity settings under the same evidence schema while keeping the primary setting explicit.
- [x] 5.5 Produce machine-readable `supported`/`not_supported`/`unavailable` states for tail structural asymmetry, equal-risk Tail vulnerability, generation-risk validity, and cross-setting stability from declared thresholds that never consume `priority_score` as evidence.

## 6. Structured Local and W&B Evidence Writer

- [x] 6.1 Add a common structured analysis output protocol and writer for `Trainer.test` that serializes named JSON/CSV evidence without embedding Tail-SID formulas in the writer.
- [x] 6.2 Write required `summary.json`, `group_metrics.csv`, `item_damage_scores.csv`, `prefix_risk_scores.csv`, and `harmful_overlap_pairs.csv`, plus sensitivity/recommendation tables when enabled, to a staging directory before atomically exposing the completed output.
- [x] 6.3 Add schema version, dataset/task, resolved input roles, primary settings, normalization metadata, pair limits, and verdict to local output and Artifact metadata.
- [x] 6.4 Publish the complete evidence directory as one W&B diagnosis Artifact through the logger-owned run without calling `wandb.init` or `wandb.finish`, while retaining local-only behavior without W&B.
- [x] 6.5 Propagate serialization and configured publication failures and ensure partial output is not marked or published as complete.

## 7. Focused Regression Coverage

- [x] 7.1 Add toy SID/frequency tests proving group schemas are identical, Tail-Cold is distinct, raw damage is gate-free, and priority multipliers cannot change evidence verdicts.
- [x] 7.2 Add deterministic asymmetric-bucket tests covering partner typing, head dominance, Tail isolation, full-versus-near separation, and pre-truncation aggregates.
- [x] 7.3 Add embedding tests for global random-pair thresholds, bucket-relative outlier separation, sampling reproducibility, and missing-embedding availability metadata.
- [x] 7.4 Add keyed recommendation tests for shuffled rows, missing/duplicate/unknown users, generated-SID shapes, hit/rank/NDCG aggregation, and null input compatibility.
- [x] 7.5 Add bootstrap, frequency-bin, sensitivity, minimum-support, and verdict tests including zero denominators, degenerate components, and unavailable evidence.
- [x] 7.6 Add structured writer tests for required schemas, deterministic row identity, pair limits, atomic failure, local-only mode, logger-owned W&B publication, and publication failure propagation.
- [x] 7.7 Add Hydra/DataModule/lineage tests for local, short, and fully qualified recommendation references and confirm all three optional input combinations compose.
- [x] 7.8 Add shell syntax and argument tests covering notes quoting, empty values, recommendation path forms, dry-run, and extra Hydra override precedence.
- [x] 7.9 Add architecture tests that prohibit common/data reverse imports of Tail-SID diagnosis, keep launcher and metric callback free of diagnosis branches, and verify only the lineage callback records input Artifact usage.

## 8. Verification and Evidence Audit

- [x] 8.1 Run focused diagnosis, data resolver/lineage, metric, writer, Hydra composition, and script tests from the repository root with `uv run pytest`.
- [x] 8.2 Run `uv run ruff check` on affected source and test paths and run shell syntax validation for `tail_sid_diagnosis.sh`.
- [x] 8.3 Run local and W&B-style dry-run compositions with and without embedding/recommendation inputs, verifying no full experiment or external service is required by unit tests.
- [x] 8.4 Run `openspec validate strengthen-tail-sid-diagnosis-evidence --strict` and confirm the change remains apply-ready.
- [x] 8.5 Re-run full Beauty diagnosis for the existing RK-Means and R-VQ SID/embedding lineages, using available keyed TIGER inference outputs when present, and audit W&B summaries against every emitted evidence file.
- [x] 8.6 Record the primary and sensitivity verdicts for RK-Means and R-VQ and make the next-method decision strictly from the declared go/no-go dimensions rather than from priority scores or two composite absolute values.
