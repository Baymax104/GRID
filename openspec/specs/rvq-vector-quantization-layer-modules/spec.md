# rvq-vector-quantization-layer-modules Specification

## Purpose
TBD - created by archiving change extract-rvq-layer. Update Purpose after archive.
## Requirements
### Requirement: RVQ SHALL use independent vector quantization layer modules
ResidualVectorQuantization SHALL represent each residual hierarchy with a distinct `nn.Module` owned by an `nn.ModuleList`. Each layer module SHALL own the VQ codebook parameter and per-layer runtime initialization state for exactly one hierarchy.

#### Scenario: one module per residual hierarchy
- **WHEN** `ResidualVectorQuantization` is constructed with `n_layers = N`
- **THEN** the model MUST create N vector quantization layer modules
- **THEN** each module MUST own exactly one centroid parameter with shape `(n_clusters, n_features)`

#### Scenario: no parallel centroid parameter list remains
- **WHEN** maintainers inspect `ResidualVectorQuantization`
- **THEN** centroid parameters MUST be owned by the layer modules rather than by a model-level `centroids_list`
- **THEN** per-layer initialization buffers and initialized flags MUST be owned by the layer modules rather than parallel model-level lists

### Requirement: RVQ layer module SHALL own single-layer VQ-STE behavior
Each RVQ layer module SHALL encapsulate single-layer K-Means++ initialization, VQ-STE training quantization inputs, and prediction behavior. ResidualVectorQuantization SHALL delegate single-layer train and predict operations to the corresponding layer module while retaining quantization loss calculation in the parent model.

#### Scenario: training a selected layer
- **WHEN** `ResidualVectorQuantization` trains the current residual hierarchy
- **THEN** it MUST call the selected layer module's training operation with the current residuals
- **THEN** the layer module MUST return layer ids, selected embeddings for residual traversal, and selected codebook embeddings for parent-owned quantization loss calculation when available
- **THEN** `ResidualVectorQuantization` MUST calculate the quantization loss for the selected layer

#### Scenario: predicting a non-selected layer
- **WHEN** `ResidualVectorQuantization` traverses a residual hierarchy that is not currently being trained
- **THEN** it MUST call that layer module's prediction operation
- **THEN** prediction MUST return layer ids and selected centroid embeddings without updating runtime initialization buffers

### Requirement: ResidualVectorQuantization SHALL retain residual orchestration
ResidualVectorQuantization SHALL remain responsible for residual traversal, optional residual normalization, current-layer schedule, Lightning lifecycle hooks, metrics, eval/test/predict steps, optimizer configuration, and checkpoint metadata.

#### Scenario: residual traversal remains model-owned
- **WHEN** `ResidualVectorQuantization.forward()` processes an embedding batch
- **THEN** it MUST iterate through the layer modules in hierarchy order
- **THEN** it MUST apply existing residual normalization behavior before each layer when configured
- **THEN** it MUST subtract each layer embedding from the current residual and stack residuals in the existing output position

#### Scenario: layer schedule remains model-owned
- **WHEN** training reaches a layer step boundary and the current layer has initialized
- **THEN** `ResidualVectorQuantization` MUST advance to the next layer without requiring layer modules to know global step budgets

### Requirement: RVQ checkpoint structure SHALL follow moduleized layer ownership
New RVQ checkpoints SHALL store centroid parameters under the RVQ layer module paths. The implementation SHALL NOT provide backward compatibility for old `centroids_list.*` checkpoint keys.

#### Scenario: new checkpoint parameter paths
- **WHEN** a new RVQ checkpoint is saved after this change
- **THEN** centroid parameters MUST be stored under per-layer module keys such as `layers.<index>.centroids`

#### Scenario: old checkpoint compatibility is not required
- **WHEN** a checkpoint using old `centroids_list.*` keys is loaded
- **THEN** the system is not required to remap those keys to the new layer module paths

