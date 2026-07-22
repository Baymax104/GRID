# tiger-user-identity-output-contract Specification

## Purpose
TBD - created by archiving change remove-user-id-training-path. Update Purpose after archive.
## Requirements
### Requirement: TIGER inference SHALL use user identity only as output keys
TIGER inference SHALL preserve the original user identity as prediction output keys and MUST NOT use that identity as a model generation feature.

#### Scenario: Prediction output preserves user identity
- **WHEN** TIGER `predict_step` receives a batch with `TigerModelInput.output_keys`
- **THEN** it MUST return `ModelOutput.keys` derived from `output_keys`
- **AND** it MUST pair those keys with generated semantic ID predictions

#### Scenario: User identity is not passed to generation
- **WHEN** TIGER inference calls `model_step` with no label data
- **THEN** the model MUST call generation using sequence semantic IDs and attention mask only
- **AND** it MUST NOT pass `user_id` as a generation argument

### Requirement: TIGER model SHALL not expose user embedding training inputs
The TIGER generation model SHALL NOT expose constructor parameters or runtime branches that train or apply user identity embeddings.

#### Scenario: Model constructor has no user embedding configuration
- **WHEN** maintainers inspect TIGER model configuration and constructor parameters
- **THEN** they MUST NOT find `num_user_bins` or equivalent user embedding table configuration

#### Scenario: Encoder does not prepend user embeddings
- **WHEN** TIGER encodes a semantic ID sequence
- **THEN** encoder inputs MUST be derived from item semantic IDs and optional separator tokens
- **AND** encoder inputs MUST NOT prepend embeddings derived from user identity
