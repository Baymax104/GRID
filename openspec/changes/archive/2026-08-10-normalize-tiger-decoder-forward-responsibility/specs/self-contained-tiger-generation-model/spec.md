## ADDED Requirements

### Requirement: TIGER decoder forward SHALL be teacher-forcing only
`TigerDecoder.forward()` SHALL be a teacher-forcing decoder computation path. It MUST require future semantic IDs, MUST construct decoder inputs from BOS plus future semantic ID embeddings, and MUST NOT use `future_ids=None` as a generation bootstrap signal.

#### Scenario: Decoder forward requires future semantic IDs
- **WHEN** maintainers inspect `TigerDecoder.forward()`
- **THEN** `future_ids` MUST be treated as a required tensor input
- **AND** the method MUST NOT contain a BOS-only branch for missing future IDs

#### Scenario: Decoder generation owns BOS bootstrap
- **WHEN** `TigerDecoder.generate()` performs the first autoregressive hierarchy step
- **THEN** it MUST construct the BOS-only decoder input inside `generate()`
- **AND** it MUST NOT call `TigerDecoder.forward()` for that BOS-only generation step

#### Scenario: Decoder generation remains inline
- **WHEN** maintainers inspect `TigerDecoder.generate()`
- **THEN** generation decoder input assembly MUST remain in the method body
- **AND** the change MUST NOT introduce separate small helper methods solely for decoder input assembly or wrapped decoder execution
