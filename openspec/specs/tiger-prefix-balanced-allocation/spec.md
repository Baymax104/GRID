# tiger-prefix-balanced-allocation Specification

## Purpose
TBD - created by archiving change add-prefix-balanced-candidate-allocation-probe. Update Purpose after archive.
## Requirements
### Requirement: Prefix allocation prior SHALL use training-only item frequencies
The probe SHALL derive item frequency from the configured training split and align it to the keyed semantic ID catalog. It MUST NOT consume evaluation/testing targets, recommendation labels, or per-user Head/Tail identities when constructing generation-time allocation priorities.

#### Scenario: Training frequency tensor is constructed
- **WHEN** the probe loads a keyed semantic ID bundle and training sequences
- **THEN** it MUST return one non-negative frequency value for every semantic ID row in key order
- **AND** catalog items absent from training MUST receive frequency zero

#### Scenario: Non-training source is requested
- **WHEN** allocation priority configuration identifies evaluation or testing as its frequency source
- **THEN** configuration or model construction MUST fail before generation
- **AND** the error MUST identify the training-only constraint

### Requirement: Decoder SHALL compute sparse prefix training mass
`TigerDecoder` SHALL aggregate item frequency over catalog entries sharing each SID prefix and SHALL query that mass for legal candidate prefixes without materializing the full prefix vocabulary.

#### Scenario: Multiple items share a prefix
- **WHEN** two or more catalog items share a prefix at hierarchy `h`
- **THEN** the prefix mass MUST equal the sum of their aligned training frequencies

#### Scenario: Legal candidate has no prior entry
- **WHEN** a legal candidate prefix cannot be found in the prefix-mass lookup
- **THEN** generation MUST fail instead of assigning an implicit allocation priority

### Requirement: Allocation SHALL reserve bounded slots inside a model-score shortlist
When enabled, the decoder SHALL construct a shortlist using original cumulative path scores, reserve at most the configured number of slots for lower-mass prefixes, fill remaining slots by original path score, and return original path scores for the retained candidates.

#### Scenario: Allocation is enabled
- **WHEN** `reserved_slots` is positive and smaller than beam width
- **THEN** reserve candidates MUST come only from the configured top-score shortlist
- **AND** retained candidates MUST remain catalog-valid when legal candidates do not fill the configured shortlist capacity
- **AND** the final retained set MUST contain no duplicate candidate path
- **AND** returned scores MUST equal the candidates' unmodified cumulative model path scores

#### Scenario: Allocation parameters are invalid
- **WHEN** reserved slots are negative, are not smaller than beam width, or pool multiplier is less than one
- **THEN** construction MUST fail with the invalid field identified

### Requirement: Disabled allocation SHALL preserve generation compatibility
The existing TIGER generation result SHALL remain unchanged when prefix-balanced allocation is disabled. The feature SHALL NOT add checkpoint parameters or require retraining.

#### Scenario: Probe is disabled
- **WHEN** identical model state, inputs, beam width, and random state run with allocation absent or disabled
- **THEN** generated semantic IDs and marginal path scores MUST be element-wise identical

#### Scenario: Existing checkpoint is loaded
- **WHEN** a checkpoint created before this capability is loaded into a probe-configured model
- **THEN** state-dict loading MUST NOT require new trainable keys

### Requirement: Allocation run identity SHALL be auditable
Probe config and prefix trace metadata SHALL record strategy state, reserved slots, shortlist multiplier, training source split, semantic ID reference, data split, checkpoint reference, beam width, seed, and a summary of aligned training frequencies.

#### Scenario: Intervention trace is produced
- **WHEN** prefix-balanced allocation is enabled during trace inference
- **THEN** the trace MUST identify the allocation configuration and its training-only source
- **AND** it MUST emit target prefix mass, shortlist membership, reserve retention, and actual reserve count per hierarchy
