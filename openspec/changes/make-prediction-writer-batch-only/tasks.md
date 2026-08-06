## 1. Writer Implementation

- [x] 1.1 Change `BaseBufferedWriter` to inherit from Lightning `Callback` instead of `BasePredictionWriter`.
- [x] 1.2 Replace `write_on_batch_end` with `on_predict_batch_end` while preserving `ModelOutput` buffering and flush behavior.
- [x] 1.3 Remove `write_interval` and epoch writing support from writer constructors and methods.

## 2. Configuration

- [x] 2.1 Remove `write_interval` from all inference callback configs that instantiate `LocalPickleWriter`.

## 3. Verification

- [x] 3.1 Add or update focused tests for batch-only writer behavior and API shape.
- [x] 3.2 Run focused tests and lint checks for touched files.
