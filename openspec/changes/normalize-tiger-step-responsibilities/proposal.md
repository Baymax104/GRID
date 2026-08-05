## Why

TIGER currently uses `model_step` as a mode-dependent dispatcher: with labels it computes teacher-forcing loss, without labels it performs autoregressive generation. This makes the model interface shallow and ambiguous because callers must know hidden `label_data` semantics to understand the returned values.

## What Changes

- Clarify `Tiger.forward()` as pure teacher-forcing model computation.
- Replace mode-dependent `model_step` usage with explicit loss computation and generation paths.
- Make `training_step`, `eval_step`, and `predict_step` call the behavior they actually need.
- Preserve validation/test behavior: compute teacher-forcing loss once and run autoregressive generation once for ranking metrics.
- Keep train/eval mode switching under Lightning lifecycle hooks instead of calling `train()` or `eval()` inside step methods.

## Capabilities

### New Capabilities

### Modified Capabilities
- `self-contained-tiger-generation-model`: Clarify TIGER Lightning step responsibilities and require explicit loss/generation paths.

## Impact

- Affected code: `src/recommendation/tiger/tiger.py`.
- Affected behavior: internal method responsibilities only; training, validation, test, and prediction outputs remain behaviorally equivalent.
- Dependencies: none.
