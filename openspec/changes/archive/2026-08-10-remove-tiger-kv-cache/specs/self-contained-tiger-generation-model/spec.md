## ADDED Requirements

### Requirement: TIGER generation SHALL NOT depend on decoder KV cache
TIGER generation SHALL compute decoder outputs without maintaining or reordering HuggingFace decoder KV cache state.

#### Scenario: Generation recomputes from current prefix
- **WHEN** TIGER generates semantic IDs for validation, testing, or prediction
- **THEN** decoder invocation MUST receive the current generated semantic ID prefix embeddings or BOS embedding as input
- **AND** decoder invocation MUST NOT receive `past_key_values`
- **AND** decoder invocation MUST NOT request cache output

#### Scenario: Beam search does not mutate cache state
- **WHEN** TIGER beam search selects top-k candidate prefixes
- **THEN** it MUST reorder generated semantic ID prefixes and marginal probabilities only
- **AND** it MUST NOT reorder decoder cache state
