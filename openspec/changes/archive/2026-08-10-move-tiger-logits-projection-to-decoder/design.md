## Context

`TigerDecoder` owns the decoder stack and generation-time projection from hidden states to candidate logits. Training-time loss computation still reaches into `TigerDecoder.decoder_mlp` from `Tiger._compute_loss`, so loss aggregation and logits projection are mixed at the LightningModule layer.

The decoder should expose model outputs as raw logits. `Tiger` should consume those logits to calculate training/evaluation loss using the configured loss function.

## Goals / Non-Goals

**Goals:**
- Keep logits projection inside `TigerDecoder`.
- Make teacher-forcing `TigerDecoder.forward()` return raw logits.
- Keep `_compute_loss()` focused on loss aggregation from logits and target IDs.
- Preserve generation behavior and raw-logit beam-search inputs.

**Non-Goals:**
- Do not output softmax probabilities from `TigerDecoder.forward()`.
- Do not move `loss_function` into `TigerDecoder`.
- Do not change beam-search scoring, prefix validation, or decoder architecture.

## Decisions

1. Return raw logits from `TigerDecoder.forward()`.
   - Rationale: raw logits are the standard input to cross entropy and preserve numerical stability. Softmax belongs to generation/scoring logic when probabilities are required.
   - Alternative considered: return softmax probabilities. Rejected because it is less suitable for training loss and less flexible for generation strategies.

2. Keep `decoder_mlp` owned by `TigerDecoder`.
   - Rationale: projection from decoder hidden states to semantic-ID logits is decoder-side model computation and is already used in generation.
   - Alternative considered: move `decoder_mlp` back to `Tiger`. Rejected because it would expose decoder internals to the LightningModule.

3. Keep loss aggregation in `Tiger`.
   - Rationale: `loss_function` is a training dependency from `TrainingModelConfig`, and `Tiger` owns Lightning training/evaluation lifecycle.
   - Alternative considered: move loss into `TigerDecoder`. Rejected because it would mix training configuration into the decoder module.

## Risks / Trade-offs

- [Risk] Callers expecting hidden states from `Tiger.forward()` will receive logits after this change. -> Mitigation: repository callers use `Tiger.forward()` only for training/evaluation loss, where logits are the desired interface.
- [Risk] Shape expectations must be updated consistently. -> Mitigation: scan and validate TIGER callers after implementation.
