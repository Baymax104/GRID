## ADDED Requirements

### Requirement: TIGER decoder SHALL own optional candidate allocation
`TigerDecoder` SHALL own candidate shortlist construction, prefix-priority lookup, reserved-slot selection, original-score backfill, and final beam ordering for prefix-balanced generation. TIGER model orchestration MAY provide configuration and aligned frequency inputs but MUST NOT reimplement decoder selection.

#### Scenario: Candidate allocation is enabled
- **WHEN** TIGER invokes generation with a valid prefix allocation configuration
- **THEN** `TigerDecoder` MUST apply allocation after legal-prefix filtering and cumulative path scoring
- **AND** TIGER MUST continue to coordinate encoder execution and decoder invocation through its existing generation path

#### Scenario: Candidate allocation is disabled
- **WHEN** ordinary training, validation, testing, or prediction does not enable the probe
- **THEN** decoder-owned beam selection MUST preserve the existing generation contract
