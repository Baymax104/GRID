from __future__ import annotations

from typing import Any

from lightning import LightningModule

from src.data.components.data_models import DiagnosisBatch
from src.quantization.tail_sid_diagnosis.metrics import build_prefix_buckets


class TailSIDDiagnosisModule(LightningModule):
    """Test-only LightningModule for Tail-SID Resolution Damage diagnosis."""

    def __init__(self, max_neighbors_per_bucket: int = 512):
        super().__init__()
        self.max_neighbors_per_bucket = max_neighbors_per_bucket

    def test_step(self, batch: DiagnosisBatch, batch_idx: int) -> dict[str, Any]:
        raw_sid = batch.sid_views.raw_sid
        sid_length = raw_sid.size(1)
        item_ids = [int(item_id) for item_id in batch.sid_views.item_ids.tolist()]
        return {
            "sid_views": batch.sid_views,
            "frequencies": batch.frequencies,
            "groups_by_item": batch.groups_by_item,
            "embeddings": batch.embeddings,
            "item_ids": item_ids,
            "groups_by_index": [batch.groups_by_item[item_id] for item_id in item_ids],
            "buckets": build_prefix_buckets(raw_sid),
            "strict_depth": max(1, sid_length - 1),
            "sid_length": sid_length,
        }
