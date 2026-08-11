## ADDED Requirements

### Requirement: Tail SID diagnosis SHALL use stable damage normalization
Tail-SID diagnosis SHALL compute composite damage scores with a stable normalization strategy that does not amplify near-constant metric components into extremely large scores.

#### Scenario: Degenerate metric component is neutralized
- **WHEN** a metric component has an interquartile range below the configured stability threshold
- **THEN** that component MUST contribute zero to composite `damage`
- **AND** the system MUST NOT divide by a tiny epsilon in a way that produces unbounded scores

#### Scenario: Metric contribution is bounded
- **WHEN** a metric component has non-degenerate spread
- **THEN** its normalized contribution MUST be clamped to a finite configured range before being added to `damage`

#### Scenario: Score metadata is emitted
- **WHEN** diagnosis outputs are written
- **THEN** `summary.json` MUST include score normalization metadata
- **AND** `report.md` MUST describe the normalization method used for `damage`

### Requirement: Tail SID diagnosis SHALL print readable terminal tables
Tail-SID diagnosis SHALL use PrettyTable for human-readable stdout tables while keeping existing machine-readable output files.

#### Scenario: Group metrics are printed with PrettyTable
- **WHEN** diagnosis completes successfully
- **THEN** stdout MUST include a PrettyTable-rendered group metrics table
- **AND** the table MUST include group, item count, full collision rate, strict near-collision rate, local density, and damage columns

#### Scenario: Top risk preview is printed with PrettyTable
- **WHEN** diagnosis result contains item and prefix risk rows
- **THEN** stdout MUST include PrettyTable-rendered top risky item and top risky prefix tables

#### Scenario: Machine-readable outputs are unchanged
- **WHEN** PrettyTable stdout rendering is enabled
- **THEN** `summary.json`, `group_metrics.csv`, `item_damage_scores.csv`, `prefix_risk_scores.csv`, and `report.md` MUST still be written
