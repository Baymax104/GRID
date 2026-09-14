"""区分训练集标定和评价预测的独立数据装配。"""

from pathlib import PurePosixPath

import hydra

from src.data.datamodule.file import FileDataModule


class ItemResolutionDataModule(FileDataModule):
    def __init__(self, data_split, inference_policy, predict_dataloader_config):
        if inference_policy == "calibrate":
            if data_split != "training":
                raise ValueError("Entropy calibration is restricted to training split.")
        elif data_split not in ("evaluation", "testing"):
            raise ValueError("Recommendation inference requires explicit evaluation/testing split.")
        folder = str(predict_dataloader_config.data_folder).replace("\\", "/")
        if PurePosixPath(folder).name != data_split:
            raise ValueError("Prediction folder does not match the declared data_split.")
        self.data_split = data_split
        super().__init__(predict_dataloader_config=hydra.utils.instantiate(predict_dataloader_config))
