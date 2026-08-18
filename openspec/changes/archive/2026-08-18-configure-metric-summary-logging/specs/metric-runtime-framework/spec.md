## ADDED Requirements

### Requirement: Metric callback SHALL support stage logging modes
The metric runtime SHALL allow each stage to choose whether computed metrics are emitted as logger history entries or as run summary values. The default mode SHALL be history logging to preserve existing behavior.

#### Scenario: History mode preserves existing logging
- **WHEN** a stage uses the default metric logging mode
- **THEN** `MetricCallback` MUST log computed metrics through the existing Lightning `log_dict` path
- **AND** existing `on_step`, `on_epoch`, `logger`, `prog_bar`, and `sync_dist` log kwargs MUST continue to apply

#### Scenario: Summary mode writes run-level values
- **WHEN** an epoch-end stage uses summary metric logging mode
- **THEN** `MetricCallback` MUST compute the prefixed stage metrics
- **AND** it MUST write scalar metric values to supported logger run summaries
- **AND** it MUST NOT write those metrics through the logger history path

#### Scenario: Unsupported summary logger does not fail metrics
- **WHEN** summary metric logging mode is active and a configured logger does not expose a supported summary destination
- **THEN** `MetricCallback` MUST NOT fail metric computation or stage reset
- **AND** it MUST make the unsupported summary logging behavior observable through a rank-zero warning or an equivalent testable signal

#### Scenario: Invalid logging mode fails clearly
- **WHEN** metric callback configuration declares an unsupported logging mode
- **THEN** callback construction MUST fail with a clear error naming the invalid mode
