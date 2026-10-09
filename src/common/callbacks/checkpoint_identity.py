"""在消费模型 checkpoint 前核对传入的文件身份。"""

from lightning.pytorch.callbacks import Callback

from src.data.components.artifacts import load_audited_checkpoint


class CheckpointIdentityCallback(Callback):
    def __init__(self, reference, sha256, wandb_entity=None, wandb_project=None):
        super().__init__()
        self.reference, self.sha256 = reference, sha256
        self.entity, self.project = wandb_entity, wandb_project

    def setup(self, trainer, pl_module, stage):
        load_audited_checkpoint(self.reference, self.sha256, self.entity, self.project)
