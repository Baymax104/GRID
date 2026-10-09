"""复用共享序列化协议，在完整 test epoch 后写入一次分析产物。"""

from src.common.writers.structured_analysis_writer import StructuredAnalysisWriter


class EpochStructuredAnalysisWriter(StructuredAnalysisWriter):
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        pass

    def on_test_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        payload = pl_module.structured_analysis_output()
        # 共享 writer 负责类型校验、原子目录、bundle、manifest 及可选发布。
        super().on_test_batch_end(trainer, pl_module, {self.output_key: payload}, None, 0)
