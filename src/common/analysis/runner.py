from __future__ import annotations

from typing import Protocol

import hydra
from omegaconf import DictConfig

from src.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class AnalysisRunner(Protocol):
    """Protocol for Hydra-instantiated offline analysis runners."""

    def run(self) -> None:
        """Run the analysis job."""


def run_analysis_runner(cfg: DictConfig) -> None:
    """Instantiate and execute the configured analysis runner."""
    analysis_cfg = cfg.get("analysis")
    if not analysis_cfg or not analysis_cfg.get("runner"):
        raise ValueError("Analysis experiments must configure cfg.analysis.runner.")

    runner_target = analysis_cfg.runner.get("_target_", "<missing>")
    logger.info(f"Instantiating analysis runner <{runner_target}>")
    runner: AnalysisRunner = hydra.utils.instantiate(analysis_cfg.runner)
    if not callable(getattr(runner, "run", None)):
        raise TypeError(f"Analysis runner <{runner_target}> must expose a callable run() method.")

    logger.info("Starting analysis!")
    runner.run()
