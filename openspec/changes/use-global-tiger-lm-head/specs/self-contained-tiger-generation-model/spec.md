## ADDED Requirements

### Requirement: TIGER decoder SHALL use a global SID lm head
`TigerDecoder` SHALL project decoder hidden states with one global SID `lm_head` whose output width is `num_hierarchies * codebook_size`. The global logits SHALL align with the same hierarchy-offset SID vocabulary used by the shared SID embedding table.

#### Scenario: Teacher-forcing forward returns global logits
- **WHEN** TIGER executes teacher-forcing decoder forward for a labeled batch
- **THEN** `TigerDecoder.forward()` MUST return raw logits with shape `(batch_size, sequence_length, num_hierarchies * codebook_size)`
- **AND** it MUST NOT apply softmax to those logits
- **AND** it MUST NOT use one projection module per hierarchy

#### Scenario: Loss uses global targets
- **WHEN** TIGER computes training or evaluation loss
- **THEN** local target SID values MUST be converted to hierarchy-offset global SID target values
- **AND** loss MUST be computed against global logits

#### Scenario: Generation slices active hierarchy logits
- **WHEN** `TigerDecoder.generate()` computes candidate scores for hierarchy `h`
- **THEN** it MUST slice global logits to `[h * codebook_size, (h + 1) * codebook_size)`
- **AND** beam search MUST continue to emit local SID values in `[0, codebook_size)`

#### Scenario: Weight tying is not introduced
- **WHEN** maintainers inspect `TigerDecoder`
- **THEN** `lm_head` MUST NOT be tied to `sid_embedding_table.weight` in this change
