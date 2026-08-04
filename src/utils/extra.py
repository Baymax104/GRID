import warnings
from time import sleep

from omegaconf import DictConfig

from src.utils.pylogger import RankedLogger
from src.utils.rich import print_config_tree

logger = RankedLogger(__name__, rank_zero_only=True)


def print_warnings_for_missing_configs(cfg: DictConfig):
    _DEFAULT_CONFIGS = ["data", "model"]
    has_warnings = False
    for config in _DEFAULT_CONFIGS:
        if not cfg.get(config):
            logger.warning(f"Config {config} was not found in the config tree. Make sure this is expected.")
            has_warnings = True
    if has_warnings:
        sleep(3)  # wait for 3 seconds to let the user read the warning


def extras(cfg: DictConfig):
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        logger.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        logger.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    if cfg.extras.get("print_config_warnings"):
        print_warnings_for_missing_configs(cfg)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        logger.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        print_config_tree(cfg, resolve=True, save_to_file=True)
