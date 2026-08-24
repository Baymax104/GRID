import sys

import hydra
import rootutils
import torch
from omegaconf import DictConfig, open_dict

import src.utils.hydra_resolvers  # noqa: F401  — registers now_tz OmegaConf resolver
from src.data.components.artifacts import resolve_checkpoint_path
from src.utils.cli import rewrite_dry_run_flag
from src.utils.extra import extras
from src.utils.launcher import pipeline_launcher
from src.utils.pylogger import RankedLogger

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

logger = RankedLogger(__name__, rank_zero_only=True)

torch.set_float32_matmul_precision("medium")


def run_training(cfg: DictConfig):
    with open_dict(cfg):
        cfg.ckpt_path = resolve_checkpoint_path(cfg.get("ckpt_path"), default_project=cfg.get("project", None))
    with pipeline_launcher(cfg) as pipeline_modules:
        logger.info("Starting training!")
        pipeline_modules.trainer.fit(
            model=pipeline_modules.model,
            datamodule=pipeline_modules.datamodule,
            ckpt_path=pipeline_modules.cfg.get("ckpt_path"),
        )

        train_metrics = pipeline_modules.trainer.callback_metrics

        if cfg.get("run_test_after_training", False):
            logger.info("Starting testing!")
            ckpt_path = None
            checkpoint_callback = getattr(pipeline_modules.trainer, "checkpoint_callback", None)
            if checkpoint_callback:
                ckpt_path = getattr(checkpoint_callback, "best_model_path", None)
                if ckpt_path == "":
                    ckpt_path = None
            if not ckpt_path:
                logger.warning("Best checkpoint not found! Using current weights for testing...")
            pipeline_modules.trainer.test(
                model=pipeline_modules.model,
                datamodule=pipeline_modules.datamodule,
                ckpt_path=ckpt_path,
            )
            logger.info(f"Best ckpt path: {ckpt_path}")

        test_metrics = pipeline_modules.trainer.callback_metrics
        metric_dict = {**train_metrics, **test_metrics}
        logger.info(f"Metrics: {metric_dict}")


def run_inference(cfg: DictConfig):
    with open_dict(cfg):
        cfg.ckpt_path = resolve_checkpoint_path(cfg.get("ckpt_path"), default_project=cfg.get("project", None))
    with pipeline_launcher(cfg) as pipeline_modules:
        logger.info("Starting inference!")
        ckpt_path = pipeline_modules.cfg.get("ckpt_path", None)
        if not ckpt_path:
            logger.warning(
                "No ckpt_path was provided. If using a model you trained, this is mandatory. "
                "Only leave ckpt_path=None if using a pre-trained model."
            )

        pipeline_modules.trainer.predict(
            model=pipeline_modules.model,
            datamodule=pipeline_modules.datamodule,
            ckpt_path=ckpt_path,
            return_predictions=False,
        )


def run_analysis(cfg: DictConfig):
    with pipeline_launcher(cfg) as pipeline_modules:
        logger.info("Starting analysis!")
        pipeline_modules.trainer.test(
            model=pipeline_modules.model,
            datamodule=pipeline_modules.datamodule,
            ckpt_path=None,
        )


def run(cfg: DictConfig):
    run_mode = cfg.get("run_mode")
    if run_mode == "train":
        run_training(cfg)
        return
    if run_mode == "inference":
        run_inference(cfg)
        return
    if run_mode == "analysis":
        run_analysis(cfg)
        return

    raise ValueError(
        f"Unsupported run_mode={run_mode!r}. Official experiments must declare run_mode: train|inference|analysis."
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="main.yaml")
def main(cfg: DictConfig):
    extras(cfg)
    run(cfg)


if __name__ == "__main__":
    sys.argv = rewrite_dry_run_flag()
    main()
