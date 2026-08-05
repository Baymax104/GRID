from contextlib import contextmanager
from dataclasses import dataclass

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.callbacks.progress import ProgressBar
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, open_dict

import src.utils.logging as logging_utils
from src.common.metrics import MetricCallback
from src.utils.file import (
    get_last_modified_file,
    has_no_extension,
    list_subfolders,
)
from src.utils.logging import DryRunLogger, finalize_loggers
from src.utils.pylogger import RankedLogger
from src.utils.rich import StepBasedRichProgressBar

logger = RankedLogger(__name__, rank_zero_only=True)

DRY_RUN_DISABLED_CALLBACK_TARGETS = {
    "lightning.pytorch.callbacks.ModelCheckpoint",
    "lightning.pytorch.callbacks.EarlyStopping",
    "src.inference.prediction_writers.LocalPickleWriter",
}

DRY_RUN_DISABLED_LOGGER_TARGETS = {
    "lightning.pytorch.loggers.csv_logs.CSVLogger",
    "lightning.pytorch.loggers.wandb.WandbLogger",
}


def ensure_training_progress_bar(callbacks: list[Callback], cfg: DictConfig) -> list[Callback]:
    """Attach the default step-based progress bar for training runs if none is configured."""
    if cfg.get("run_mode") != "train":
        return callbacks

    if any(isinstance(callback, ProgressBar) for callback in callbacks):
        return callbacks

    logger.info("Attaching default step-based Rich training progress bar.")
    callbacks.append(StepBasedRichProgressBar())
    return callbacks


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
            if last_modified:
                ckpt_path = last_modified
                logger.info(f"Found most recent checkpoint path: {ckpt_path}. Starting job from this checkpoint.")

    cfg.ckpt_path = ckpt_path
    return cfg


def apply_dry_run_overrides(cfg: DictConfig) -> DictConfig:
    """Apply minimal-run and no-result-writing overrides for dry run mode."""
    if not cfg.get("dry_run", False):
        return cfg

    logger.info("Applying dry run overrides: minimal execution without business result writes.")

    with open_dict(cfg):
        callback_definitions = cfg.get("callbacks")
        if callback_definitions:
            for name, cb_conf in callback_definitions.items():
                if isinstance(cb_conf, DictConfig) and cb_conf.get("_target_") in DRY_RUN_DISABLED_CALLBACK_TARGETS:
                    logger.info(f"Disabling callback for dry run: {name} <{cb_conf.get('_target_')}>")
                    callback_definitions[name] = None

        logger_definitions = cfg.get("logger")
        if logger_definitions:
            for name, lg_conf in logger_definitions.items():
                if isinstance(lg_conf, DictConfig) and lg_conf.get("_target_") in DRY_RUN_DISABLED_LOGGER_TARGETS:
                    logger.info(f"Disabling logger for dry run: {name} <{lg_conf.get('_target_')}>")
                    logger_definitions[name] = None

        cfg.trainer.root.log_every_n_steps = 1
        cfg.trainer.root.max_epochs = 1
        cfg.trainer.root.limit_predict_batches = 1

        if cfg.get("run_mode") == "train":
            cfg.trainer.root.max_steps = 1
            cfg.trainer.root.limit_train_batches = 1
            cfg.trainer.root.limit_val_batches = 0
            cfg.trainer.root.limit_test_batches = 0
            cfg.trainer.root.num_sanity_val_steps = 0

        if "run_test_after_training" in cfg:
            cfg.run_test_after_training = False

    return cfg


def instantiate_callbacks(callbacks_cfg: DictConfig) -> list[Callback]:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: list[Callback] = []

    if not callbacks_cfg:
        logger.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            logger.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> list[Logger]:
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    loggers: list[Logger] = []

    if not logger_cfg:
        logger.warning("No logger configs found! Skipping...")
        return loggers

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for name, lg_conf in logger_cfg.items():
        if name == "wandb":
            logger.info("Authenticating to W&B!")
            logging_utils.login_wandb()

        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            logger.info(f"Instantiating logger <{lg_conf._target_}>")
            loggers.append(hydra.utils.instantiate(lg_conf))

    return loggers


def attach_metric_callback(callbacks: list[Callback], cfg: DictConfig) -> list[Callback]:
    """Attach config-declared model metrics through the shared metric callback."""
    metrics_cfg = cfg.get("model", {}).get("metrics")
    if not metrics_cfg:
        return callbacks

    logger.info("Attaching config-declared metric callback.")
    metric_engine = hydra.utils.instantiate(metrics_cfg, _recursive_=False)
    callbacks.append(MetricCallback(engine=metric_engine))
    return callbacks


def initialize_pipeline_modules(cfg: DictConfig) -> PipelineModules:
    """
    Initialize and instantiate various objects required for running pipelines.

    Python-side top-level instantiation entrypoints are read from top-level component
    groups such as ``cfg.data``, ``cfg.model``, ``cfg.trainer``, ``cfg.callbacks``,
    and ``cfg.logger``.

    Args:
        cfg (DictConfig): Configuration object containing top-level component entrypoints.

    Returns:
        PipelineModules: A dataclass containing the instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    cfg = update_cfg_with_most_recent_checkpoint_path(cfg)
    cfg = apply_dry_run_overrides(cfg)

    logger.info(f"Instantiating datamodule <{cfg.data.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data.datamodule)

    logger.info(f"Instantiating model <{cfg.model.root._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model.root)

    logger.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    callbacks = attach_metric_callback(callbacks, cfg)
    callbacks = ensure_training_progress_bar(callbacks, cfg)

    logger.info("Instantiating loggers...")
    loggers: list[Logger] = instantiate_loggers(cfg.get("logger"))
    if cfg.get("dry_run", False) and len(loggers) == 0:
        logger.info("Using DryRunLogger to satisfy Lightning logging without writing business results.")
        loggers = [DryRunLogger()]

    logger.info(f"Instantiating trainer <{cfg.trainer.root._target_}>")

    enable_checkpointing = any(isinstance(cb, ModelCheckpoint) for cb in callbacks)
    enable_model_summary = any(isinstance(cb, ModelSummary) for cb in callbacks)
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer.root,
        callbacks=callbacks,
        logger=loggers,
        # The default behavior for lightning it to set `enable_checkpointing` and
        # `enable_model_summary` to True, which might be misleading when we are trying to
        # debug. We change the default to False, but this can be overridden by either
        # setting the parameters in the config file or passing the callbacks as part
        # of the callbacks YAML.
        enable_checkpointing=cfg.trainer.root.get("enable_checkpointing", enable_checkpointing),
        enable_model_summary=cfg.trainer.root.get("enable_model_summary", enable_model_summary),
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
            logger.info("Logging hyperparameters!")
        yield pipeline_modules
    except Exception as ex:
        raise ex
    finally:
        # We add the try catch to make sure the loggers are finalized even if the task fails.
        if pipeline_modules:
            finalize_loggers(pipeline_modules.trainer)
