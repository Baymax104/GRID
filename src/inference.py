import os

import hydra
import rootutils
import torch


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from src.utils import RankedLogger, extras
from src.utils.custom_hydra_resolvers import *
from src.utils.launcher_utils import pipeline_launcher


console_logger = RankedLogger(__name__, rank_zero_only=True)

torch.set_float32_matmul_precision("medium")


def inference(cfg: DictConfig):
    """
    Runs inference using a pre-trained model.

    Args:
        cfg: A DictConfig configuration composed by Hydra.

    Returns:
        A dict with all instantiated objects.
    """

    with pipeline_launcher(cfg) as pipeline_modules:
        console_logger.info("Starting inference!")
        ckpt_path = pipeline_modules.cfg.get("ckpt_path", None)
        if not ckpt_path:
            console_logger.warning(
                "No ckpt_path was provided. If using a model you trained, this is mandatory. Only leave ckpt_path=None if using a pre-trained model."
            )

        pipeline_modules.trainer.predict(
            model=pipeline_modules.model,
            datamodule=pipeline_modules.datamodule,
            ckpt_path=ckpt_path,
            return_predictions=False,
        )


@hydra.main(version_base="1.3", config_path="../configs", config_name="inference.yaml")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for inference.

    Args:
        cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    extras(cfg)

    # run inference
    inference(cfg)


if __name__ == "__main__":
    main()
