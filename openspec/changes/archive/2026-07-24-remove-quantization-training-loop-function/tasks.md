## 1. Config Cleanup

- [x] 1.1 Remove `training_loop_function: ${model.training_loop_function}` from `configs/model/rkmeans_train.yaml`.
- [x] 1.2 Remove the `training_loop_function` `_target_` block from `configs/model/rkmeans_train.yaml`.
- [x] 1.3 Remove `training_loop_function` root field and `_target_` block from `configs/model/rvq_train.yaml`.
- [x] 1.4 Remove `training_loop_function` root field and `_target_` block from `configs/model/rqvae_train.yaml`.

## 2. Shared Distributed Initialization Utilities

- [x] 2.1 Identify whether RKMeans broadcast helpers should be reused, copied, or moved for RVQ/RQVAE without creating a cross-model abstraction that violates independent model ownership.
- [x] 2.2 Ensure every direct initialization path handles single-process mode and initialized `torch.distributed` mode.

## 3. RKMeans Cleanup

- [x] 3.1 Remove `training_loop_function` from `ResidualKMeans.__init__()` and stop storing it.
- [x] 3.2 Remove `automatic_optimization = False` setup from RKMeans.
- [x] 3.3 Remove RKMeans `training_step()` custom loop invocation and let Lightning automatic optimization handle optimizer/scheduler stepping.

## 4. RVQ Direct Initialization

- [x] 4.1 Remove `training_loop_function` from `ResidualVectorQuantization.__init__()` and stop storing it.
- [x] 4.2 Remove `automatic_optimization = False` setup from RVQ.
- [x] 4.3 Replace RVQ init-loss parameter transfer with rank-zero K-Means++ compute, broadcast, and direct `centroids_list[layer_idx].copy_(...)`.
- [x] 4.4 Remove RVQ initialization transition state that is no longer needed (`is_initial_step_list`, `init_centroids_list`, `init_loss_function`) and update forward unlock logic accordingly.
- [x] 4.5 Remove RVQ `training_step()` custom loop invocation and rely on automatic optimization.

## 5. RQVAE Direct Initialization

- [x] 5.1 Remove `training_loop_function` from `ResidualQuantizationVAE.__init__()` and stop storing it.
- [x] 5.2 Remove `automatic_optimization = False` setup from RQVAE.
- [x] 5.3 Keep RQVAE K-Means++ + convergence refinement, but return/broadcast final centroids and direct `centroids_list[layer_idx].copy_(...)`.
- [x] 5.4 Remove RQVAE initialization transition state that is no longer needed (`is_initial_step_list`, `init_centroids_list`, `init_loss_function`) and update progressive layer unlock logic accordingly.
- [x] 5.5 Remove RQVAE `training_step()` custom loop invocation and rely on automatic optimization.

## 6. Remove Custom Loop Module

- [x] 6.1 Delete `src/quantization/training_loop_functions.py` after all active references are removed.
- [x] 6.2 Confirm no active Python/config files reference `scale_loss_by_world_size_for_initialization_training_loop` or `src.quantization.training_loop_functions`.

## 7. Tests and Verification

- [x] 7.1 Update RKMeans tests to assert no `training_loop_function` constructor/config exposure and no manual optimization flag.
- [x] 7.2 Add/update RVQ tests covering same-step direct initialization and absence of `is_initial_step_list` / `init_centroids_list`.
- [x] 7.3 Add/update RQVAE tests covering same-step direct initialization after convergence and absence of `is_initial_step_list` / `init_centroids_list`.
- [x] 7.4 Add/update distributed broadcast tests for at least one RVQ/RQVAE direct initialization path.
- [x] 7.5 Run targeted quantization unit tests.
- [x] 7.6 Run Hydra target/import smoke check for RKMeans, RVQ, and RQVAE.
- [x] 7.7 Run final search confirming active code/config has no `training_loop_function`, `scale_loss_by_world_size_for_initialization_training_loop`, `automatic_optimization = False`, or `manual_backward` residue for quantization.
