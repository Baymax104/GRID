import os

import hydra
import rootutils
import torch


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from src.utils import RankedLogger, extras
from src.utils.custom_hydra_resolvers import *
from src.utils.launcher_utils import pipeline_launcher
from src.utils.restart_job import LocalJobLauncher


console_logger = RankedLogger(__name__, rank_zero_only=True)

torch.set_float32_matmul_precision("medium")


def train(cfg: DictConfig):
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # Pipeline launcher initializes the modules needed for the pipeline to run.
    # It also serves as a context manager, so all resources are properly closed after the pipeline is done.
    with pipeline_launcher(cfg) as pipeline_modules:

        if cfg.get("train"):
            console_logger.info("Starting training!")
            pipeline_modules.trainer.fit(
                model=pipeline_modules.model,
                datamodule=pipeline_modules.datamodule,
                ckpt_path=cfg.get("ckpt_path"),
            )
        train_metrics = pipeline_modules.trainer.callback_metrics

        if cfg.get("test"):
            console_logger.info("Starting testing!")
            ckpt_path = None
            # Check if a checkpoint callback is available and if it has a best model path.
            # Note that if multiple checkpoint callbacks are used, only the first one will be used
            # to determine the best model path for testing.
            checkpoint_callback = getattr(pipeline_modules.trainer, "checkpoint_callback", None)
            if checkpoint_callback:
                ckpt_path = getattr(checkpoint_callback, "best_model_path", None)
                if ckpt_path == "":
                    ckpt_path = None
            if not ckpt_path:
                console_logger.warning("Best checkpoint not found! Using current weights for testing...")
            pipeline_modules.trainer.test(
                model=pipeline_modules.model,
                datamodule=pipeline_modules.datamodule,
                ckpt_path=ckpt_path,
            )
            console_logger.info(f"Best ckpt path: {ckpt_path}")

        test_metrics = pipeline_modules.trainer.callback_metrics

        # merge train and test metrics
        metric_dict = {**train_metrics, **test_metrics}

        console_logger.info(f"Metrics: {metric_dict}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)
    job_launcher = LocalJobLauncher(cfg=cfg)
    job_launcher.launch(function_to_run=train)


if __name__ == "__main__":
    main()
