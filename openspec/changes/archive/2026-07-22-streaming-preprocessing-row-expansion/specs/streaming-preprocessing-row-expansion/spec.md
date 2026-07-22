## ADDED Requirements

### Requirement: SequenceDataset SHALL support streaming row expansion preprocessing
`SequenceDataset` preprocessing execution SHALL support preprocessing functions that return `None`, a single row, or an iterable of rows. Expanded rows MUST be processed through the remaining preprocessing functions as a stream, without requiring the dataset to materialize the full expanded result before continuing.

#### Scenario: preprocessing function expands one row into multiple rows
- **WHEN** a preprocessing function returns an iterable of row dictionaries for one input row
- **THEN** `SequenceDataset` MUST pass each returned row through all remaining preprocessing functions
- **AND** it MUST yield each fully processed row independently

#### Scenario: preprocessing function filters a row
- **WHEN** a preprocessing function returns `None`
- **THEN** `SequenceDataset` MUST drop that row branch
- **AND** it MUST continue processing subsequent source rows

#### Scenario: dict return is treated as one row
- **WHEN** a preprocessing function returns a dictionary row
- **THEN** `SequenceDataset` MUST treat it as a single row
- **AND** it MUST NOT iterate over dictionary keys as expanded rows

### Requirement: Row expansion SHALL be lazy and bounded by the active row branch
Streaming preprocessing SHALL avoid list-pipeline behavior where every preprocessing step receives and re-traverses a fully materialized list of expanded rows.

#### Scenario: expansion returns a generator
- **WHEN** a preprocessing function returns a generator of rows
- **THEN** `SequenceDataset` MUST consume it lazily
- **AND** subsequent preprocessing functions MUST execute per generated row branch

#### Scenario: large source dataset is processed
- **WHEN** the reader yields a large or unbounded dataset
- **THEN** preprocessing row expansion MUST NOT require loading all expanded rows into memory before yielding the first processed row
