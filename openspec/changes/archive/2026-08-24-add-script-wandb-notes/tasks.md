## 1. Logger Config

- [x] 1.1 Add `notes: null` to W&B logger configs for RKMeans, RVQ, RQ-VAE, TIGER, and Tail-SID diagnosis.
- [x] 1.2 Verify Hydra composition accepts `logger.wandb.notes=...` for each affected experiment.

## 2. Script Interface

- [x] 2.1 Convert RKMeans, RVQ, RQ-VAE, and TIGER train scripts from fixed command lines to Bash `ARGS` arrays.
- [x] 2.2 Add shared local parsing behavior in affected scripts for `--notes=value`, `--notes value`, and `--dry-run`.
- [x] 2.3 Preserve pass-through of extra Hydra overrides after default script arguments.
- [x] 2.4 Remove fixed default `--dry-run` from training scripts so dry-run is explicit.
- [x] 2.5 Extend `tail_sid_diagnosis.sh` with the same `--notes`, `--dry-run`, and pass-through behavior.
- [x] 2.6 Fail fast with a clear error when `--notes` is provided without a value.

## 3. Verification

- [x] 3.1 Run shell syntax checks for all modified scripts.
- [x] 3.2 Smoke-check dry-run argument rewriting with at least one train script and the diagnosis script.
- [x] 3.3 Smoke-check notes propagation by composing an affected experiment with `logger.wandb.notes=...`.
- [x] 3.4 Run `openspec validate add-script-wandb-notes --strict`.
