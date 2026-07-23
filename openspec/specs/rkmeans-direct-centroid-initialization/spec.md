# RKMeans Direct Centroid Initialization

## Purpose

定义 RKMeans KMeans layer 直接初始化 centroids 的协议，消除 `is_initial_step` 过渡状态。

## Requirements

### Requirement: RKMeans KMeans layer SHALL use direct centroid initialization
`KMeansLayer` SHALL initialize its centroid parameter directly when its initialization buffer contains enough samples. The layer SHALL NOT require a separate initialization-loss optimizer step or an `is_initial_step` transition state to become initialized.

#### Scenario: initialization buffer fills
- **WHEN** a KMeans layer receives training residuals and its `init_buffer` reaches `init_buffer_size`
- **THEN** the layer MUST run K-Means++ initialization for that layer
- **THEN** the layer MUST directly write the initialized centroid values into `layer.centroids`
- **THEN** the layer MUST mark itself initialized in the same train step

#### Scenario: initialization buffer is not full
- **WHEN** a KMeans layer receives training residuals and its `init_buffer` has fewer than `init_buffer_size` samples
- **THEN** the layer MUST keep collecting initialization samples
- **THEN** the layer MUST NOT mark itself initialized
- **THEN** the layer MUST NOT update centroid values from partial initialization data

### Requirement: RKMeans KMeans layer initialization SHALL use rank-zero broadcast in distributed training
When torch distributed training is initialized, RKMeans KMeans layer initialization SHALL compute initial centroids on rank zero and broadcast those centroids to every rank before the layer is marked initialized.

#### Scenario: distributed centroid initialization
- **WHEN** torch distributed is initialized and a KMeans layer initialization buffer fills
- **THEN** rank zero MUST compute the K-Means++ centroid tensor
- **THEN** all ranks MUST participate in a broadcast from rank zero
- **THEN** every rank MUST copy the broadcast centroid tensor into its local `layer.centroids`
- **THEN** every rank MUST mark the layer initialized only after broadcast has completed

#### Scenario: single-process centroid initialization
- **WHEN** torch distributed is not initialized and a KMeans layer initialization buffer fills
- **THEN** the current process MUST compute K-Means++ centroids locally
- **THEN** the layer MUST copy those centroids directly into `layer.centroids`

### Requirement: RKMeans layer initialization state SHALL be single-state
RKMeans KMeans layer initialization state SHALL be represented by a single initialized state exposed to `ResidualKMeans`. Temporary initialization targets SHALL NOT be kept as cross-step layer state.

#### Scenario: maintainers inspect layer state
- **WHEN** maintainers inspect `KMeansLayer`
- **THEN** the layer MUST NOT define an `is_initial_step` state
- **THEN** the layer MUST NOT require `init_centroids` to persist across train steps
- **THEN** `ResidualKMeans` MUST determine schedule readiness using only the layer's initialized state

#### Scenario: checkpoint initialized state
- **WHEN** RKMeans saves a checkpoint
- **THEN** the checkpoint MUST preserve each layer's initialized boolean state
- **THEN** the checkpoint MUST NOT persist `init_buffer` or temporary initialization centroid targets
