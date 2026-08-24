## 1. Experiment Identity

- [x] 1.1 Replace top-level `wandb_project` with `project` and `group` across official experiment configs.
- [x] 1.2 Set group values to experiment family names.

## 2. Component References

- [x] 2.1 Update W&B logger configs to use `${project}` and `${group}`.
- [x] 2.2 Update W&B artifact/checkpoint writer callback configs to use `${project}` and `${group}`.
- [x] 2.3 Update data/model artifact lookup configs to pass `wandb_project: ${project}`.

## 3. Runtime Default and Validation

- [x] 3.1 Use top-level `project` directly when resolving train/inference checkpoint W&B references.
- [x] 3.2 Add or update tests for project-based short W&B URI defaults.
- [x] 3.3 Run focused tests, ruff, and strict OpenSpec validation.
