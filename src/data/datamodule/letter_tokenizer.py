"""LETTER keyed tensor输入；导出必须在完整目录上执行碰撞处理。"""

from lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset


class LetterTokenizerDataModule(LightningDataModule):
    def __init__(self, embeddings, batch_size=1024, num_workers=0):
        super().__init__()
        self.embeddings, self.batch_size, self.num_workers = embeddings, batch_size, num_workers
        self.dataset = TensorDataset(embeddings["keys"], embeddings["content"], embeddings["cf"])

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=len(self.dataset), num_workers=0)

    def predict_dataloader(self):
        return self.val_dataloader()
