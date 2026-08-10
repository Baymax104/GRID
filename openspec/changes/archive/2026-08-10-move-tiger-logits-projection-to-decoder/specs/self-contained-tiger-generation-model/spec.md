## ADDED Requirements

### Requirement: TIGER decoder SHALL own logits projection
`TigerDecoder` SHALL own projection from decoder hidden states to semantic-ID logits. Teacher-forcing decoder forward SHALL return raw logits rather than hidden states or softmax probabilities, and TIGER loss computation SHALL consume those logits without reaching into decoder projection internals.

#### Scenario: Teacher-forcing forward returns raw logits
- **WHEN** TIGER executes teacher-forcing decoder forward for a labeled batch
- **THEN** `TigerDecoder.forward()` MUST return raw logits with shape `(batch_size, sequence_length, codebook_size)`
- **AND** it MUST NOT apply softmax to those logits

#### Scenario: Loss consumes logits only
- **WHEN** TIGER computes training or evaluation loss
- **THEN** the loss helper MUST consume logits and target semantic IDs
- **AND** it MUST NOT call `TigerDecoder.decoder_mlp` or otherwise access decoder projection internals

#### Scenario: Generation keeps decoder-side projection
- **WHEN** `TigerDecoder.generate()` computes candidate scores for beam search
- **THEN** candidate logits MUST be produced by decoder-owned projection layers
- **AND** probability conversion MUST remain local to beam-search scoring logic
