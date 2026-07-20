import torch


class ModelOutput:
    """
    推理结果写入的字段规范层，直接持有 keys + predictions。

    Attributes:
        keys: 每条预测对应的业务主键（如 item_id、user_id）。
        predictions: 模型预测值（如 embedding、cluster_ids、semantic_ids）。
    """

    def __init__(self, keys: torch.Tensor, predictions: torch.Tensor):
        self.keys = keys  # (n,)
        self.predictions = predictions  # (n, *)
