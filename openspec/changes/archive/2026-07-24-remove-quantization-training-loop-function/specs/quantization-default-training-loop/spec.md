## ADDED Requirements

### Requirement: Quantization models SHALL use Lightning automatic optimization
RKMeans、RVQ、RQVAE SHALL use Lightning default automatic optimization and SHALL NOT require a configurable `training_loop_function` for initialization or normal training.

#### Scenario: Maintainer checks quantization model constructors
- **WHEN** 维护者查看 `ResidualKMeans`、`ResidualVectorQuantization`、`ResidualQuantizationVAE` 的 constructor
- **THEN** constructors MUST NOT accept `training_loop_function`
- **AND** models MUST NOT set `automatic_optimization = False` for quantization initialization

#### Scenario: Maintainer checks quantization training configs
- **WHEN** 维护者查看 RKMeans、RVQ、RQVAE training model configs
- **THEN** configs MUST NOT contain `training_loop_function`
- **AND** configs MUST NOT reference `scale_loss_by_world_size_for_initialization_training_loop`

### Requirement: Quantization initialization SHALL assign centroids directly
Quantization models SHALL initialize centroids by directly assigning computed centroid tensors to model parameters, not by using an initialization loss and a custom training loop to move parameters.

#### Scenario: RKMeans initializes a layer
- **WHEN** RKMeans has collected enough residual points for the current layer
- **THEN** rank 0 MUST compute initial centroids
- **AND** all ranks MUST receive the same centroid tensor before assignment
- **AND** the layer MUST mark itself initialized without requiring a special optimizer step

#### Scenario: RVQ initializes a layer
- **WHEN** RVQ has collected enough residual points for a layer
- **THEN** rank 0 MUST compute K-Means++ centroids
- **AND** all ranks MUST receive the same centroid tensor before assignment
- **AND** the layer MUST mark itself initialized without requiring `is_initial_step_list`

#### Scenario: RQVAE initializes a layer
- **WHEN** RQVAE has collected enough residual points for a layer
- **THEN** rank 0 MUST compute K-Means++ centroids and run convergence refinement
- **AND** all ranks MUST receive the same final centroid tensor before assignment
- **AND** the layer MUST mark itself initialized without requiring `is_initial_step_list`

### Requirement: Custom quantization training loop SHALL be removed
The active codebase SHALL NOT contain `scale_loss_by_world_size_for_initialization_training_loop` or an active `src/quantization/training_loop_functions.py` module required by configs.

#### Scenario: Maintainer searches active code
- **WHEN** 维护者搜索 active source and config files
- **THEN** `scale_loss_by_world_size_for_initialization_training_loop` MUST NOT be referenced
- **AND** no active config MUST reference `src.quantization.training_loop_functions`
