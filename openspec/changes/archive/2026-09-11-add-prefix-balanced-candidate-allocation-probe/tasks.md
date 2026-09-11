## 1. Training-only allocation input

- [x] 1.1 Add a data-layer helper that reads training TFRecords, validates keyed semantic IDs, and returns aligned non-negative item frequencies.
- [x] 1.2 Add focused tests for key alignment, zero-frequency catalog items, empty training data, duplicate keys, and training-only source validation.

## 2. Decoder-owned prefix-balanced allocation

- [x] 2.1 Add immutable allocation configuration and sparse encoded-prefix mass lookup with validation and deterministic queries.
- [x] 2.2 Implement shortlist, low-mass reserve selection, score-based backfill, and final original-score ordering inside `TigerDecoder`.
- [x] 2.3 Wire optional aligned frequencies and allocation configuration through `Tiger` without adding checkpoint parameters.
- [x] 2.4 Add decoder/model tests for prefix aggregation, reserve behavior, parameter errors, deterministic ties, original score preservation, and disabled-path element-wise compatibility.

## 3. Trace and paired diagnosis evidence

- [x] 3.1 Extend prefix trace payload and metadata with target prefix mass, shortlist membership, reserve retention, actual reserve counts, and allocation identity.
- [x] 3.2 Add matched baseline/intervention audits for keys, labels, split, checkpoint, SID lineage/shape, seed, equal beam width, and opposite allocation state.
- [x] 3.3 Produce per-user, frequency-group, and per-layer candidate allocation evidence while separating candidate access, Top10 additions/losses, and prefix survival.
- [x] 3.4 Add per-setting gate evaluation and a four-setting cross-run verdict helper using the predeclared recommendation and cost thresholds.
- [x] 3.5 Add tests covering valid pairs, every mismatch class, transition conservation, intermediate-only improvement, cost guardrails, and incomplete cross-setting evidence.

## 4. Config, launcher, and manual execution contract

- [x] 4.1 Add thin TIGER prefix allocation probe experiment/model/logger/callback configs with explicit source and strategy fields.
- [x] 4.2 Add a root launcher supporting required data/split/beam/checkpoint/SID/group/notes, default seed 42, dry-run, allocation parameters, and tail-positioned Hydra overrides.
- [x] 4.3 Extend Tail-SID diagnosis config and launcher inputs for allocation-probe comparison without changing existing search-ranking defaults.
- [x] 4.4 Add config-compose and launcher tests for both flag syntaxes, invalid/empty values, spaced paths, dry-run, notes, source split rejection, and override precedence.

## 5. Reviewable handoff and verification

- [x] 5.1 Document the four-setting manual baseline/intervention/diagnosis execution matrix, artifact identity fields, and advance/stop/inconclusive gates.
- [x] 5.2 Run focused pytest, Ruff, shell syntax checks, Hydra compose, `git diff --check`, and `openspec validate add-prefix-balanced-candidate-allocation-probe --strict` without starting full experiments.

## 6. Same-width diagnosis regression

- [x] 6.1 Keep widened-beam recovery disabled when diagnosis is comparing a same-width allocation baseline/intervention pair.
- [x] 6.2 Add an end-to-end evidence regression test that reaches candidate-allocation analysis with equal beam widths.

## 7. Catalog-valid allocation regression

- [x] 7.1 Audit the real baseline/intervention artifacts for catalog membership and identify intervention-only invalid generated SIDs.
- [x] 7.2 Keep invalid shortlist padding excluded from reserve and score backfill, enforce a retained-candidate invariant, and add focused decoder/diagnosis tests.
- [x] 7.3 Record target shortlist membership by candidate prefix value after score sorting, with a regression test for position/token-ID divergence.
