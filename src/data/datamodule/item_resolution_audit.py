"""在既有推理数据流外包裹有界的确定性用户样本。"""

from src.data.components.item_resolution_audit import select_audit_rows
from src.data.datamodule.item_resolution import ItemResolutionDataModule


class ItemResolutionAuditDataModule(ItemResolutionDataModule):
    def __init__(self, audit_users, audit_sampling_seed, **kwargs):
        super().__init__(**kwargs)
        # StageDataModule 的唯一配置源是 stage_to_config。
        from lightning.pytorch.trainer.states import TrainerFn

        config = self.stage_to_config[TrainerFn.PREDICTING]
        if self.data_split != "evaluation" or config.num_workers != 0 or config.batch_size_per_device != 1:
            raise ValueError("Audit requires evaluation, batch1 and num_workers=0 for paired sampling.")
        if not 1 <= audit_users <= 512 or audit_sampling_seed < 0:
            raise ValueError("Invalid audit sample size/seed.")
        self.audit_users, self.audit_sampling_seed = audit_users, audit_sampling_seed

    def _build_dataset(self, stage, curr_config):
        if self.trainer.world_size != 1:
            raise ValueError("Audit requires a single process/GPU.")
        source = super()._build_dataset(stage, curr_config)
        return select_audit_rows(source, self.audit_users, self.audit_sampling_seed)
