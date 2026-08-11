## 1. Dependencies

- [x] 1.1 Add `prettytable` to project dependencies using `uv`.
- [x] 1.2 Verify `pyproject.toml` and `uv.lock` reflect the dependency.

## 2. Score Stabilization

- [x] 2.1 Replace epsilon-only robust z-score scaling with degenerate-IQR neutralization.
- [x] 2.2 Clamp non-degenerate per-metric z-score contributions.
- [x] 2.3 Add score normalization metadata to `summary.json` and `report.md`.

## 3. Terminal Display

- [x] 3.1 Use PrettyTable for group metrics stdout display.
- [x] 3.2 Use PrettyTable for top risky item and prefix previews.
- [x] 3.3 Keep CSV/JSON/Markdown file outputs compatible with existing filenames.

## 4. Verification

- [x] 4.1 Add tests for degenerate metric normalization avoiding huge damage values.
- [x] 4.2 Add tests for PrettyTable stdout rendering.
- [x] 4.3 Run focused pytest, ruff, and `openspec validate stabilize-tail-sid-diagnosis-output --strict`.
