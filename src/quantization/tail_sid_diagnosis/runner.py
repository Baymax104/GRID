from __future__ import annotations

from dataclasses import dataclass

from src.quantization.tail_sid_diagnosis.data import (
    compute_train_frequencies,
    iter_training_rows,
    load_embeddings_for_items,
    load_sid_views,
)
from src.quantization.tail_sid_diagnosis.metrics import TailSIDDiagnosisMetric, assign_frequency_groups
from src.quantization.tail_sid_diagnosis.reporting import print_summary, write_outputs


@dataclass(frozen=True)
class TailSIDDiagnosisRunner:
    """Hydra-instantiated runner for Tail-SID Resolution Damage diagnosis."""

    data_dir: str
    semantic_id_path: str
    raw_num_hierarchies: int
    output_dir: str
    embedding_path: str | None = None
    head_ratio: float = 0.2
    tail_ratio: float = 0.2
    max_neighbors_per_bucket: int = 512
    top_k_report: int = 10

    def run(self) -> None:
        sid_views = load_sid_views(self.semantic_id_path, self.raw_num_hierarchies)
        embeddings = load_embeddings_for_items(self.embedding_path, sid_views.item_ids)
        frequencies = compute_train_frequencies(iter_training_rows(self.data_dir), sid_views.item_ids)
        groups = assign_frequency_groups(sid_views.item_ids, frequencies, self.head_ratio, self.tail_ratio)

        metric = TailSIDDiagnosisMetric(max_neighbors_per_bucket=self.max_neighbors_per_bucket)
        metric.update(
            sid_views=sid_views,
            frequencies=frequencies,
            groups_by_item=groups,
            embeddings=embeddings,
        )
        result = metric.compute()

        write_outputs(result, self.output_dir, top_k_report=self.top_k_report)
        print_summary(result, self.output_dir)
