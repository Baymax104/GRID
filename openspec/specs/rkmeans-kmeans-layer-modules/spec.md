# RKMeans K-Means Layer Modules

## Purpose

定义 ResidualKMeans 使用独立 KMeans layer module 的协议，使单层 K-Means 聚类参数与行为封装到同目录下的专用 layer module 中。

## Requirements

### Requirement: RKMeans SHALL use independent KMeans layer modules
ResidualKMeans SHALL represent each residual hierarchy with a distinct `nn.Module` owned by an `nn.ModuleList`. Each layer module SHALL own the K-Means centroid parameter and per-layer K-Means runtime state for exactly one hierarchy.

#### Scenario: one module per residual hierarchy
- **WHEN** `ResidualKMeans` is constructed with `n_layers = N`
- **THEN** the model MUST create N KMeans layer modules
- **THEN** each module MUST own exactly one centroid parameter with shape `(n_clusters, n_features)`

#### Scenario: no parallel centroid parameter list remains
- **WHEN** maintainers inspect `ResidualKMeans`
- **THEN** centroid parameters MUST be owned by the KMeans layer modules rather than by a model-level `centroids_list`

### Requirement: KMeans layer module SHALL own single-layer training behavior
Each KMeans layer module SHALL encapsulate the single-layer K-Means initialization, mini-batch KMeans update, and prediction behavior. ResidualKMeans SHALL delegate single-layer train and predict operations to the corresponding layer module.

#### Scenario: training a selected layer
- **WHEN** `ResidualKMeans` trains the current residual hierarchy
- **THEN** it MUST call the selected layer module's training operation with the current residuals
- **THEN** the layer module MUST return layer ids, selected centroid embeddings, and a quantization loss

#### Scenario: predicting a non-selected layer
- **WHEN** `ResidualKMeans` traverses a residual hierarchy that is not currently being trained
- **THEN** it MUST call that layer module's prediction operation
- **THEN** prediction MUST return layer ids and selected centroid embeddings without updating centroid parameters

### Requirement: ResidualKMeans SHALL retain layer-wise residual orchestration
ResidualKMeans SHALL remain responsible for residual traversal, current-layer schedule, Lightning lifecycle hooks, metrics, eval/test/predict steps, optimizer configuration, and the existing manual optimization entrypoint.

#### Scenario: residual traversal remains model-owned
- **WHEN** `ResidualKMeans.forward()` processes an embedding batch
- **THEN** it MUST iterate through the layer modules in hierarchy order
- **THEN** it MUST apply existing residual normalization behavior before each layer when configured
- **THEN** it MUST subtract each layer embedding from the current residual and stack residuals in the existing output position

#### Scenario: layer schedule remains model-owned
- **WHEN** training reaches a layer step boundary
- **THEN** `ResidualKMeans` MUST advance its current layer schedule without requiring the KMeans layer modules to know global step budgets

### Requirement: RKMeans checkpoint structure SHALL follow moduleized layer ownership
New RKMeans checkpoints SHALL store centroid parameters under the KMeans layer module paths. The implementation SHALL NOT provide backward compatibility for old `centroids_list.*` checkpoint keys.

#### Scenario: new checkpoint parameter paths
- **WHEN** a new RKMeans checkpoint is saved after this change
- **THEN** centroid parameters MUST be stored under per-layer module keys such as `layers.<index>.centroids`

#### Scenario: old checkpoint compatibility is not required
- **WHEN** a checkpoint using old `centroids_list.*` keys is loaded
- **THEN** the system is not required to remap those keys to the new layer module paths

### Requirement: KMeans layer runtime initialization state SHALL not become persistent model artifact state
KMeans initialization buffers and temporary initialization targets SHALL remain runtime training state rather than persistent checkpoint contract. Cluster-count state used for online updates SHALL be reset at training start and SHALL NOT be required for inference checkpoint loading.

#### Scenario: training state reset
- **WHEN** `ResidualKMeans.on_train_start()` runs
- **THEN** it MUST reset each KMeans layer module's initialization buffer and cluster counts on the active device

#### Scenario: inference uses centroids only
- **WHEN** RKMeans runs inference from a new checkpoint
- **THEN** prediction MUST rely on saved centroid parameters and MUST NOT require persisted initialization buffers
