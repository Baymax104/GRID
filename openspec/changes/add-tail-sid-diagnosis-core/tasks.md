## 1. Module Structure and CLI

- [x] 1.1 Create `src/quantization/tail_sid_diagnosis/` with public module exports.
- [x] 1.2 Add an argparse CLI entrypoint that accepts data, SID, hierarchy, embedding, and output parameters.
- [x] 1.3 Add root `tail_sid_diagnosis.sh` using `uv run`.

## 2. Input Loading

- [x] 2.1 Load semantic IDs through the keyed prediction bundle loader and preserve item keys.
- [x] 2.2 Split SID predictions into raw SID, model SID, and optional dedup digit views.
- [x] 2.3 Load optional embedding keyed bundle and align embeddings by item key.

## 3. Frequency Groups

- [x] 3.1 Scan `data_dir/training` records and compute `freq_train`.
- [x] 3.2 Assign Head, Mid, Tail, and Tail-Cold groups from training frequency.

## 4. Diagnosis Metrics

- [x] 4.1 Build prefix buckets for all raw SID depths.
- [x] 4.2 Compute full collision, MPOD, strict near-collision, local density, suffix weakness, and last-step burden.
- [x] 4.3 Compute optional embedding-based semantic mismatch and harmful overlap counts.
- [x] 4.4 Compute item damage, tail damage, and prefix risk scores.

## 5. Outputs and Display

- [x] 5.1 Write `summary.json`, `group_metrics.csv`, `item_damage_scores.csv`, and `prefix_risk_scores.csv`.
- [x] 5.2 Print a concise human-readable summary with output paths and group metrics.

## 6. Verification

- [x] 6.1 Add CPU unit tests for SID view splitting and structural metrics.
- [x] 6.2 Add CPU unit tests for frequency grouping and output file generation.
- [x] 6.3 Run focused pytest and ruff checks for the new module.
