"""CoPMRec v2：仅监督联合目录视图，所有阶段均保留历史商品。"""

import torch

from src.recommendation.copmrec.catalog import FullCatalogObjective


class CoPMRec(FullCatalogObjective):
    copmrec_version = "v2"

    @property
    def formal_release(self):
        return dict(
            protocol="copmrec-formal-joint-only-v2-v1",
            implementation_version=self.copmrec_version,
            evidence_phase="formal",
            initialization_seed=self.initialization_seed,
            native_view_ce=False,
            training_history_exclusion=False,
            validation_history_exclusion=False,
            inference_history_exclusion=False,
            development_checkpoints_eligible=False,
        )

    @property
    def unified_scratch_contract(self):
        return {**super().unified_scratch_contract, "deployment": "raw_full_catalog_dense_without_history_exclusion"}

    @property
    def history_eligibility_contract(self):
        return dict(
            protocol="copmrec-final-history-exclusion-off-v1",
            history_exclusion=False,
            history_representation="unchanged",
            score_rule="unchanged_full_catalog_logits",
            rank_ties="stable_catalog_row",
            cold_items_eligible=True,
            single_process=True,
        )

    def _joint_losses(self, target_ids, encoded, mask, query, rows, *, training=False, content_logits=None):
        if training and content_logits is not None:
            raise ValueError("Joint-only training requires its own shared catalog projection.")
        return super()._joint_losses(
            target_ids, encoded, mask, query, rows, training=training, content_logits=content_logits
        )

    @torch.no_grad()
    def retrieve(self, model_input, mode=None, target_ids=None):
        if mode not in (None, "dense") or target_ids is not None:
            raise ValueError("CoPMRec v2 requires dense prediction without target traces.")
        _, _, query = self.encode(model_input.input_ids, model_input.attention_mask)
        scores = self.dense_logits(query)
        rows = scores.argsort(dim=-1, descending=True, stable=True)[:, : self.top_k]
        return self.semantic_ids[rows], scores.gather(1, rows)

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["copmrec_formal_release"] = self.formal_release

    def on_load_checkpoint(self, checkpoint):
        release = checkpoint.get("copmrec_formal_release")
        expected = self.formal_release
        if (
            not isinstance(release, dict)
            or set(release) != set(expected)
            or any(type(release[key]) is not type(value) or release[key] != value for key, value in expected.items())
        ):
            raise ValueError("CoPMRec v2 requires its own joint-only formal checkpoint.")
        super().on_load_checkpoint(checkpoint)
