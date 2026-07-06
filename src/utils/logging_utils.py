import json
import os
from importlib.util import find_spec
from typing import Any, Optional

from dotenv import load_dotenv
from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf

from src.utils.pylogger import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)

# logging constants
END_RUN = "end_run"


class DryRunLogger(Logger):
    """Minimal no-op logger used to satisfy Lightning logging during dry runs."""

    @property
    def name(self) -> str:
        return "dry_run"

    @property
    def version(self) -> str:
        return "0"

    @property
    def save_dir(self) -> str:
        return ""

    @property
    def experiment(self) -> "DryRunLogger":
        return self

    def log_hyperparams(self, params, *args, **kwargs):
        return None

    def log_metrics(self, metrics: dict[str, float], step=None) -> None:
        pass

    def finalize(self, status: str):
        return None


def convert_dict_to_json_string(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


@rank_zero_only
def login_wandb():
    """
    If WANDB_API_KEY is set in the environment, login to wandb.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Now you can access the WANDB_API_KEY
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        import wandb

        wandb.login(key=wandb_api_key, relogin=True)


@rank_zero_only
def finalize_loggers(trainer: Any, status=END_RUN) -> None:
    """
    Finalize loggers after training is done.

    :param trainer: The Lightning trainer.
    :param status: The status of the trainer.

    """
    for logger in trainer.loggers:
        if hasattr(logger, "finalize"):
            logger.finalize(status)

    if find_spec("wandb"):  # check if wandb is installed. If so, close connection to wandb.
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()


@rank_zero_only
def log_hyperparameters(cfg: DictConfig, model: LightningModule, trainer: Trainer) -> None:
    """
    Controls which config parts are saved by Lightning loggers.

    Args:
        cfg: A DictConfig object containing the main config.
        model: The Lightning model.
        trainer: The Lightning trainer.

    Additional saves:
        - Number of model parameters
    """
    hparams = {}
    # We resolve the configs to get the actual paths for logging.
    cfg = OmegaConf.to_container(cfg, resolve=True)

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["paths"] = cfg["paths"]
    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams["model/params/non_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
