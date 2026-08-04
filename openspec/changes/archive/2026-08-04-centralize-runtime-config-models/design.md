## Context

The project uses Hydra `_target_` blocks to instantiate small dataclass objects that describe dataset, dataloader, and model training configuration. Existing data config dataclasses are stored in `src/data/components/config_models.py`, while the recently introduced training dependency grouping lives in `src/common/components/training_components.py`.

The desired ownership is narrower: `common/configs/` should own configuration model classes, organized by configuration domain (`data`, `model`, and later `experiment` or `trainer` if concrete classes exist). Domain packages such as `src/data` should keep runtime data loading logic, readers, collate functions, and batch data models.

## Goals / Non-Goals

**Goals:**

- Make `src/common/configs/` the canonical package for runtime config dataclasses.
- Place data config classes in `src/common/configs/data.py`.
- Place the model training config class in `src/common/configs/model.py`.
- Rename `TrainingComponents` to `TrainingModelConfig`.
- Rename the YAML key and model constructor parameter to `training_model_config`.
- Avoid making `src/common` depend on `src/data` implementation classes.

**Non-Goals:**

- Do not create empty `experiment.py` or `trainer.py` modules until real config dataclasses exist for those domains.
- Do not move OmegaConf `DictConfig` annotations; those are framework config containers, not project runtime config dataclasses.
- Do not alter optimizer, scheduler, loss, dataset, dataloader, or model behavior.
- Do not provide old import-path compatibility shims.

## Decisions

1. Use `src/common/configs/` instead of `src/common/components/`
   - Rationale: config dataclasses describe Hydra configuration shape, not executable runtime components.
   - Alternative considered: keep model config in `common/components`. Rejected because it keeps config models mixed with losses, schedulers, and evaluators.

2. Split by domain files: `data.py` and `model.py`
   - Rationale: this matches the user's requested taxonomy and keeps future `experiment` or `trainer` config models straightforward to add.
   - Alternative considered: one large `config_models.py`. Rejected because it would quickly become a mixed registry without local domain grouping.

3. Avoid importing `src.data` from `src.common.configs.data`
   - Rationale: `common` must remain lower-level than data. The dataset reader factory can be typed as `Callable[..., Any]` without changing runtime behavior.
   - Alternative considered: preserve `BaseDataReader` type import. Rejected because it introduces a common-to-data dependency.

4. Rename both class and parameter/key
   - Rationale: using `TrainingModelConfig` while leaving `training_components` in YAML and constructor signatures would create ambiguous naming.
   - Alternative considered: class-only rename. Rejected because the user asked to rename `training_components` and the class together.

## Risks / Trade-offs

- [Risk] Broad Hydra `_target_` path migration can miss a config file -> Mitigation: residual scan all `configs/`, `src/`, `tests/`, and non-archive specs.
- [Risk] OpenSpec living specs still refer to old data paths -> Mitigation: include explicit modified spec deltas and update living specs during implementation.
- [Risk] Existing unarchived changes mention `TrainingComponents` -> Mitigation: update the unarchived `consolidate-model-training-components` artifacts to the new name before final validation.
