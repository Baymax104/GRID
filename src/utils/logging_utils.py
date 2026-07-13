import os
from importlib.util import find_spec
from typing import Any

from dotenv import load_dotenv
from lightning.pytorch.loggers import Logger
from lightning_utilities.core.rank_zero import rank_zero_only

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

    def log_metrics(self, metrics: dict[str, float], step=None):
        pass

    def finalize(self, status: str):
        return None


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
def finalize_loggers(trainer: Any, status=END_RUN):
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
