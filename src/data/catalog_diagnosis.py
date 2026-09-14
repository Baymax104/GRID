"""为 CGBS 和 Hybrid 显式选择标签划分，复用原 diagnosis 证据逻辑。"""

from typing import Any

from src.data.datasets import DiagnosisDataset


class CatalogDiagnosisDataset(DiagnosisDataset):
    def __init__(self, data_split: str, **kwargs: Any):
        if data_split not in {"evaluation", "testing"}:
            raise ValueError("Catalog diagnosis data_split must be evaluation or testing.")
        self.label_data_split = data_split
        super().__init__(**kwargs)

    def _trace_data_split(self) -> str:
        for trace in (self.fixed_prefix_trace, self.widened_prefix_trace):
            if trace is not None and trace.metadata["data_split"] != self.label_data_split:
                raise ValueError("Catalog diagnosis explicit data_split disagrees with Prefix Trace metadata.")
        return self.label_data_split
