"""将源码留档作为真实文件上传，独立于业务预测产物 writer。"""

from pathlib import Path

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only


@rank_zero_only
def publish_source_snapshot(loggers: list, snapshot: dict) -> None:
    for experiment_logger in loggers:
        if not isinstance(experiment_logger, WandbLogger):
            continue
        import wandb

        run = experiment_logger.experiment
        if not isinstance(run.id, str) or not run.id:
            raise RuntimeError("Source snapshot publication requires a real W&B run with a non-empty string id.")
        artifact = wandb.Artifact(
            name=f"grid-source-{run.id}",
            type="code",
            metadata={
                "role": "source_snapshot",
                "source_sha256": snapshot["source_sha256"],
                "manifest_sha256": snapshot["manifest_sha256"],
                "file_count": snapshot["file_count"],
                "origin": snapshot["origin"],
            },
        )
        for filename in ("source.tar.gz", "manifest.json", "runtime.json"):
            artifact.add_file(str(Path(snapshot["directory"]) / filename), name=filename)
        run.log_artifact(artifact)
