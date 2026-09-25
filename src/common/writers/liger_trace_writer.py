"""复用辅助产物发布，追加候选覆盖汇总。"""

import json
from pathlib import Path

import torch

from src.common.writers.auxiliary_tensor_writer import AuxiliaryTensorWriter
from src.data.components.liger_trace import summarize_liger_trace


class LigerTraceWriter(AuxiliaryTensorWriter):
    def on_predict_end(self, trainer, pl_module):
        super().on_predict_end(trainer, pl_module)
        if self.global_rank != 0:
            return
        bundle = torch.load(self.merged_output_path, map_location="cpu", weights_only=True)
        summary = summarize_liger_trace(bundle)
        Path(self.output_dir, "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        for logger in trainer.loggers:
            logger.log_metrics({"candidate_trace/" + k: v for k, v in summary.items()})
