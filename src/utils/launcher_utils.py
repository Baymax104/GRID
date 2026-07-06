from contextlib import contextmanager
from dataclasses import dataclass

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, open_dict

from src.utils.file_utils import (
    get_last_modified_file,
    has_no_extension,
    list_subfolders,
)
from src.utils.instantiators import instantiate_callbacks, instantiate_loggers
from src.utils.logging_utils import DryRunLogger, finalize_loggers, log_hyperparameters
from src.utils.pylogger import RankedLogger
from src.utils.utils import has_class_object_inside_list

command_line_logger = RankedLogger(__name__, rank_zero_only=True)

DRY_RUN_DISABLED_CALLBACK_TARGETS = {
    "lightning.pytorch.callbacks.ModelCheckpoint",
    "lightning.pytorch.callbacks.EarlyStopping",
    "src.utils.inference_utils.LocalPickleWriter",
}

DRY_RUN_DISABLED_LOGGER_TARGETS = {
    "lightning.pytorch.loggers.csv_logs.CSVLogger",
    "lightning.pytorch.loggers.wandb.WandbLogger",
}


@dataclass
class PipelineModules:
    cfg: DictConfig
    datamodule: LightningDataModule
    model: LightningModule
    # We use the plural form to match the names used by lightning
    callbacks: list[Callback]
    loggers: list[Logger]
    trainer: Trainer


def update_cfg_with_most_recent_checkpoint_path(cfg: DictConfig) -> DictConfig:
    """
    Updates the configuration with the most recent checkpoint path when ``ckpt_path``
    points to a checkpoint directory.

    This keeps the convenience of resolving a checkpoint folder to its newest
    checkpoint file without coupling the default pipeline to any restart metadata.

    Args:
        cfg (DictConfig): The configuration dictionary containing training parameters.

    Returns:
        DictConfig: The updated configuration dictionary with the most recent checkpoint path.
    """

    ckpt_path = cfg.get("ckpt_path", None)

    if ckpt_path is not None and has_no_extension(ckpt_path) and cfg.get("should_retrieve_latest_ckpt_path", False):
        # If a path to a folder is passed, we assume it contains folders with versions of checkpoints.
        # We expect those folders to be named using a timestamp.
        checkpoint_folders = list_subfolders(ckpt_path)
        if len(checkpoint_folders) > 0:
            # We sort them in reverse order to get the most recent one.
            checkpoint_folders.sort(reverse=True)
            # We take the first one, which is the most recent one.
            latest_ckpt_folder = checkpoint_folders[0]
            last_modified = get_last_modified_file(folder_path=latest_ckpt_folder, suffix="*.ckpt")
            if len(last_modified) > 0:
                ckpt_path = last_modified
                command_line_logger.info(
                    f"Found most recent checkpoint path: {ckpt_path}. Starting job from this checkpoint."
                )

    cfg.ckpt_path = ckpt_path
    return cfg


def apply_dry_run_overrides(cfg: DictConfig) -> DictConfig:
    """Apply minimal-run and no-result-writing overrides for dry run mode."""
    if not cfg.get("dry_run", False):
        return cfg

    command_line_logger.info("Applying dry run overrides: minimal execution without business result writes.")

    with open_dict(cfg):
        callback_definitions = cfg.get("components", {}).get("callbacks")
        if callback_definitions:
            for name, cb_conf in callback_definitions.items():
                if isinstance(cb_conf, DictConfig) and cb_conf.get("_target_") in DRY_RUN_DISABLED_CALLBACK_TARGETS:
                    command_line_logger.info(f"Disabling callback for dry run: {name} <{cb_conf.get('_target_')}>")
                    callback_definitions[name] = None

        logger_definitions = cfg.get("components", {}).get("logger")
        if logger_definitions:
            for name, lg_conf in logger_definitions.items():
                if isinstance(lg_conf, DictConfig) and lg_conf.get("_target_") in DRY_RUN_DISABLED_LOGGER_TARGETS:
                    command_line_logger.info(f"Disabling logger for dry run: {name} <{lg_conf.get('_target_')}>")
                    logger_definitions[name] = None

        cfg.trainer.log_every_n_steps = 1
        cfg.trainer.max_epochs = 1
        cfg.trainer.limit_predict_batches = 1

        if cfg.get("run_mode") == "train":
            cfg.trainer.max_steps = 1
            cfg.trainer.limit_train_batches = 1
            cfg.trainer.limit_val_batches = 0
            cfg.trainer.limit_test_batches = 0
            cfg.trainer.num_sanity_val_steps = 0

            if cfg.get("model") and "train_layer_wise" in cfg.model:
                cfg.model.train_layer_wise = False

        if "run_test_after_training" in cfg:
            cfg.run_test_after_training = False

    return cfg


def initialize_pipeline_modules(cfg: DictConfig) -> PipelineModules:
    """
    Initialize and instantiate various objects required for running pipelines.

    Python-side top-level instantiation entrypoints are read from ``cfg.components``.
    Parameter domains such as ``data_loading``, ``model``, and ``trainer`` remain
    available for shared values, overrides, and hyperparameter logging.

    Args:
        cfg (DictConfig): Configuration object containing component entrypoints and parameter domains.

    Returns:
        PipelineModules: A dataclass containing the instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    cfg = update_cfg_with_most_recent_checkpoint_path(cfg)
    cfg = apply_dry_run_overrides(cfg)

    command_line_logger.info(f"Instantiating datamodule <{cfg.components.data_loading.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.components.data_loading.datamodule)

    command_line_logger.info(f"Instantiating model <{cfg.components.model.root._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.components.model.root)

    command_line_logger.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("components", {}).get("callbacks"))

    command_line_logger.info("Instantiating loggers...")
    loggers: list[Logger] = instantiate_loggers(cfg.get("components", {}).get("logger"))
    if cfg.get("dry_run", False) and len(loggers) == 0:
        command_line_logger.info("Using DryRunLogger to satisfy Lightning logging without writing business results.")
        loggers = [DryRunLogger()]

    command_line_logger.info(f"Instantiating trainer <{cfg.components.trainer.root._target_}>")

    enable_checkpointing = has_class_object_inside_list(callbacks, ModelCheckpoint)
    enable_model_summary = has_class_object_inside_list(callbacks, ModelSummary)
    trainer: Trainer = hydra.utils.instantiate(
        cfg.components.trainer.root,
        callbacks=callbacks,
        logger=loggers,
        # The default behavior for lightning it to set `enable_checkpointing` and
        # `enable_model_summary` to True, which might be misleading when we are trying to
        # debug. We change the default to False, but this can be overridden by either
        # setting the parameters in the config file or passing the callbacks as part
        # of the callbacks YAML.
        enable_checkpointing=cfg.components.trainer.root.get("enable_checkpointing", enable_checkpointing),
        enable_model_summary=cfg.components.trainer.root.get("enable_model_summary", enable_model_summary),
    )

    pipeline_modules = PipelineModules(
        cfg=cfg,
        datamodule=datamodule,
        model=model,
        callbacks=callbacks,
        loggers=loggers,
        trainer=trainer,
    )

    return pipeline_modules


@contextmanager
def pipeline_launcher(cfg: DictConfig):
    """
    Launches the pipeline with the given configuration and logger.
    Args:
        cfg (DictConfig): Configuration object containing pipeline settings.
    Yields:
        PipelineModules: A dataclass containing the instantiated objects.
    Raises:
        Exception: Propagates any exception that occurs during pipeline initialization.
    Notes:
        - If the configuration contains a logger, hyperparameters will be logged.
        - Ensures that loggers are finalized and profiler output is saved even if the task fails.
    """

    pipeline_modules: PipelineModules | None = None
    try:
        pipeline_modules: PipelineModules = initialize_pipeline_modules(cfg)
        # Log hyperparameters if loggers are present
        if len(pipeline_modules.loggers) > 0:
            command_line_logger.info("Logging hyperparameters!")
            log_hyperparameters(cfg, pipeline_modules.model, pipeline_modules.trainer)
        yield pipeline_modules
    except Exception as ex:
        raise ex
    finally:
        # We add the try catch to make sure the loggers are finalized even if the task fails.
        if pipeline_modules:
            finalize_loggers(pipeline_modules.trainer)
