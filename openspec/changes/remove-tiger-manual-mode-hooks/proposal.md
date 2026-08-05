## Why

TIGER currently defines custom hooks that manually switch the wrapped encoder and decoder between train and eval mode. These hooks duplicate Lightning loop behavior and also maintain a local `is_training` attribute that is not consumed by the repository.

## What Changes

- Remove TIGER's manual train/eval mode switching helper.
- Remove validation/test/predict hooks that only switch encoder/decoder mode.
- Keep metric reset and metric logging hooks intact.
- Rely on Lightning to manage train/eval mode for training, validation, test, and prediction loops.

## Capabilities

### New Capabilities

### Modified Capabilities
- `self-contained-tiger-generation-model`: Clarify that TIGER mode switching is owned by Lightning lifecycle behavior rather than custom submodule hooks.

## Impact

- Affected code: `src/recommendation/tiger/tiger.py`.
- Affected behavior: internal lifecycle cleanup only; training, validation, test, and prediction semantics should remain equivalent under Lightning Trainer.
- Dependencies: none.
