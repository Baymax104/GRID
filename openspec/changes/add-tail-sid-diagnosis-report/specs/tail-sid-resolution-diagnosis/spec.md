## ADDED Requirements

### Requirement: Tail SID diagnosis SHALL emit a Markdown report
The diagnosis SHALL write a human-readable `report.md` that summarizes key diagnosis outputs without replacing the machine-readable CSV and JSON files.

#### Scenario: Markdown report is written
- **WHEN** the diagnosis completes successfully
- **THEN** the output directory MUST contain `report.md`
- **AND** the report MUST include sections for summary, group metrics, top risky items, top risky prefixes, and output files

#### Scenario: Report top-k is configurable
- **WHEN** the diagnosis CLI receives `--top-k-report`
- **THEN** the generated report MUST limit top risky item and prefix sections to that many rows

### Requirement: Tail SID diagnosis SHALL show report location in stdout
The diagnosis CLI SHALL show the Markdown report location and a concise top-risk preview in stdout.

#### Scenario: Report path is displayed
- **WHEN** the diagnosis completes successfully
- **THEN** stdout MUST include the path to `report.md`

#### Scenario: Top risk preview is displayed
- **WHEN** the diagnosis result contains item and prefix risk rows
- **THEN** stdout MUST include the highest-risk item id
- **AND** stdout MUST include the highest-risk prefix
