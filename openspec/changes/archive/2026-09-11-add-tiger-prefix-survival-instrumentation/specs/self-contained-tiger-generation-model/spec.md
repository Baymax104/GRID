## ADDED Requirements

### Requirement: TIGER decoder SHALL own optional generation tracing
`TigerDecoder` SHALL own derivation of beam parent, target-prefix survival, rank, score, and cutoff observations from its constrained beam-search state. TIGER model orchestration MAY request those observations but MUST NOT reimplement beam state reconstruction outside the decoder.

#### Scenario: Decoder tracing is requested
- **WHEN** TIGER invokes decoder generation with labeled target IDs and tracing enabled
- **THEN** `TigerDecoder` MUST derive trace observations from the same tensors used for constrained beam selection
- **AND** TIGER MUST associate the returned trace with the batch output keys

#### Scenario: Decoder tracing is not requested
- **WHEN** ordinary validation, testing, or prediction runs without tracing
- **THEN** the decoder MUST preserve its existing generation behavior and output compatibility

### Requirement: Teacher-forcing trace statistics SHALL consume raw decoder logits
TIGER SHALL derive target token diagnostic statistics from the raw global logits returned by the existing teacher-forcing path. The diagnostic path MUST NOT modify loss inputs or introduce an additional trainable projection.

#### Scenario: Teacher-forcing trace is computed
- **WHEN** a labeled trace batch is evaluated
- **THEN** the model MUST use the active hierarchy slice of the existing global logits
- **AND** training/evaluation loss MUST continue to consume the original logits unchanged
