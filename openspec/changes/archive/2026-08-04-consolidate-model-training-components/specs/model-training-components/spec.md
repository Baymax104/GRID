## ADDED Requirements

### Requirement: Train model configs SHALL group training dependencies
Official train model configs SHALL pass training-only model dependencies through a single `training_model_config` object under `model.root`. The group SHALL include the primary loss function, optimizer factory, optional scheduler factory, and any model-specific auxiliary training loss such as RQVAE reconstruction loss.

#### Scenario: Quantization train configs use training_model_config
- **WHEN** a maintainer inspects `configs/model/rkmeans_train.yaml`, `configs/model/rvq_train.yaml`, or `configs/model/rqvae_train.yaml`
- **THEN** `root` MUST pass `training_model_config: ${model.training_model_config}`
- **THEN** `root` MUST NOT pass separate `loss_function`, `optimizer`, `scheduler`, or `reconstruction_loss_function` fields
- **THEN** `training_model_config` MUST contain the loss and optimizer definitions used by that model

#### Scenario: TIGER train config uses training_model_config
- **WHEN** a maintainer inspects `configs/model/tiger_train.yaml`
- **THEN** `root` MUST pass `training_model_config: ${model.training_model_config}`
- **THEN** `root` MUST NOT pass separate `loss_function`, `optimizer`, or `scheduler` fields
- **THEN** `training_model_config` MUST contain the cross-entropy loss, optimizer factory, and scheduler factory used by TIGER training

#### Scenario: No-scheduler train models preserve null scheduler
- **WHEN** a train model does not use a scheduler
- **THEN** its `training_model_config.scheduler` field MUST be `null`
- **THEN** model optimizer configuration MUST continue returning an optimizer without a Lightning scheduler entry

### Requirement: TrainingModelConfig SHALL be a passive runtime container
`TrainingModelConfig` SHALL be a passive runtime container for instantiated training dependencies and factories. It MUST NOT implement training logic, optimizer stepping, scheduler stepping, or model-family-specific behavior.

#### Scenario: Model constructors receive one training dependency object
- **WHEN** a train-capable model is constructed from official config
- **THEN** the model constructor MUST receive a `training_model_config` object
- **THEN** the model constructor MUST NOT require separate `loss_function`, `optimizer`, `scheduler`, or `reconstruction_loss_function` constructor arguments

#### Scenario: Existing training behavior remains model-owned
- **WHEN** a model executes training or `configure_optimizers`
- **THEN** loss computation MUST remain in the model implementation
- **THEN** optimizer and scheduler construction MUST remain coordinated by the model's `configure_optimizers`
- **THEN** `TrainingModelConfig` MUST NOT call model parameters or Lightning trainer state directly

### Requirement: Inference model configs SHALL not gain training_model_config
Official inference model configs SHALL not add `training_model_config`, because inference model construction does not require loss, optimizer, scheduler, or reconstruction loss dependencies.

#### Scenario: Inference configs remain training-dependency free
- **WHEN** a maintainer inspects `configs/model/*_inference.yaml`
- **THEN** those configs MUST NOT add `training_model_config`
- **THEN** those configs MUST NOT add loss, optimizer, scheduler, or reconstruction loss dependencies solely for structural consistency with train configs
