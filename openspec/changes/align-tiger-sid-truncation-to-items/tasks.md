## 1. Test Coverage

- [x] 1.1 Add focused tests for SID-aware normalization that trims long flattened SID inputs on whole-item boundaries.
- [x] 1.2 Add tests for non-divisible `sequence_length` where retained real tokens are right-padded instead of keeping a partial item SID group.
- [x] 1.3 Add regression coverage that `target_ids` is not mutated and attention masks match the normalized input.
- [x] 1.4 Preserve or add coverage showing default normalization remains token-level when no SID hierarchy width is configured.

## 2. Implementation

- [x] 2.1 Inline fixed-length sequence normalization logic into `normalize_sequence`.
- [x] 2.2 Update `normalize_sequence` to accept the optional SID hierarchy width and validate invalid values.
- [x] 2.3 Update `configs/data/tiger_train.yaml` train and eval `normalize_sequence` entries to pass `sid_hierarchy: ${num_hierarchies}`.

## 3. Verification

- [x] 3.1 Run the focused data preprocessing tests with `uv run pytest`.
- [x] 3.2 Run `uv run ruff check` on touched Python files.
- [x] 3.3 Run OpenSpec validation for `align-tiger-sid-truncation-to-items`.
