# Preprocessed Sequence Normalization

## Purpose

定义 TIGER 序列数据在 preprocessing 阶段将 model input 归一化为固定长度并生成 attention_mask 的协议，使 `collate_fn_sequence` 不再承担 sequence 长度归一化职责。

## Requirements

### Requirement: TIGER preprocessing SHALL normalize model input sequences row-by-row

TIGER train/eval preprocessing SHALL provide a row-level `normalize_sequence` function that consumes a configured input sequence field, pads or trims it to `sequence_length`, writes the normalized tensor back to the configured input field, and generates an `attention_mask` field from the normalized input and `padding_token`.

#### Scenario: short input sequence is padded and masked
- **WHEN** `normalize_sequence` receives a row whose configured input field is shorter than `sequence_length`
- **THEN** it MUST right-pad the input field with `padding_token` to `sequence_length`
- **AND** it MUST generate `attention_mask` with `1` for non-padding tokens and `0` for padding tokens

#### Scenario: long input sequence is trimmed and masked
- **WHEN** `normalize_sequence` receives a row whose configured input field is longer than `sequence_length`
- **THEN** it MUST keep the most recent non-padding tokens according to the existing TIGER normalization semantics
- **AND** it MUST generate `attention_mask` with the same length as the normalized input field

#### Scenario: target labels are not normalized
- **WHEN** `normalize_sequence` processes a row containing `target_ids`
- **THEN** it MUST NOT pad, trim, or otherwise mutate `target_ids`

### Requirement: TIGER normalization SHALL run after label generation

TIGER train/eval preprocessing chains SHALL place `normalize_sequence` after SID expansion and next-k label generation so labels are generated from the unnormalized semantic-ID sequence and model inputs are fixed-length before collate.

#### Scenario: train preprocessing order is inspected
- **WHEN** 维护者检查 `configs/data/tiger_train.yaml` 的 train preprocessing chain
- **THEN** `normalize_sequence` MUST appear after `expand_sid_causal_duplicate_sequences`
- **AND** it MUST appear after `generate_next_k_labels`

#### Scenario: eval preprocessing order is inspected
- **WHEN** 维护者检查 `configs/data/tiger_train.yaml` 的 eval preprocessing chain
- **THEN** `normalize_sequence` MUST appear after `generate_next_k_labels`
- **AND** eval preprocessing MUST NOT apply SID causal duplicate expansion
