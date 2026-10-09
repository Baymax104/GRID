"""M3 单卡 Trainer.test 观测，不更新来源模型或更改正式部署。"""

import hashlib
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch
from lightning import LightningModule
from omegaconf import OmegaConf
from torch.nn import functional as F

from src.common.writers.structured_analysis import StructuredAnalysisOutput
from src.data.utils import gather_predictions_by_keys
from src.recommendation.copmrec.ablation import M3_PROTOCOL, CoPMRecAblation
from src.recommendation.copmrec.module import CoPMRec
from src.recommendation.liger.candidate_guidance import mix_log_probs


def source_model(*, source_variant, **kwargs):
    """用同一组件参数装配 Full 或具有独立身份的消融来源。"""
    return CoPMRec(**kwargs) if source_variant == "full" else CoPMRecAblation(variant=source_variant, **kwargs)


def retrieval_values(predictions, labels):
    """根据实际列表位置独立计算单目标四指标。"""
    hits = (predictions == labels[:, None]).all(-1)
    result = {}
    for k in (5, 10):
        valid = hits[:, :k]
        rank = valid.long().argmax(-1) + 1
        result[f"recall@{k}"] = valid.any(-1).float()
        result[f"ndcg@{k}"] = valid.any(-1).float() / torch.log2(rank.float() + 1)
    return result


def prefix_observations(model, x, labels, *, mixed=True):
    """真实前缀上的合法条件概率，训练同定义的未排除历史目录支持。"""
    encoded, mask, query = model.encode(x.input_ids, x.attention_mask)
    tokens = labels.long() + model.offsets
    decoded = model.transformer(encoder_outputs=encoded, attention_mask=mask, labels=tokens, use_cache=False)
    processor = model.training_candidate_processor(model.dense_logits(query))
    prefix = tokens.new_full((len(tokens), 1), model.transformer.config.decoder_start_token_id)
    observations = []
    for depth in range(model.num_hierarchies):
        _, gen, mass, valid = processor.distributions(prefix, decoded.logits[:, depth].float())
        target = labels[:, depth, None].long()
        if not valid.gather(1, target).all():
            raise ValueError("Diagnosis target prefix is not legal.")
        pg, pm = gen.exp(), mass.exp()
        for probability in (pg, pm):
            if not torch.allclose(probability.sum(-1), torch.ones(len(tokens), device=pg.device), atol=2e-5):
                raise ValueError("Legal prefix probabilities do not normalize.")
        average = (pg + pm) / 2
        log_average = average.clamp_min(torch.finfo(pg.dtype).tiny).log()
        # 避免非法位置的 0 * -inf，熵/JS 都在同一合法支持计算。
        safe_gen, safe_mass = gen.masked_fill(~valid, 0), mass.masked_fill(~valid, 0)
        row = dict(
            gen_probability=pg.gather(1, target).squeeze(1),
            mass_probability=pm.gather(1, target).squeeze(1),
            gen_nll=-gen.gather(1, target).squeeze(1),
            mass_nll=-mass.gather(1, target).squeeze(1),
            gen_entropy=-(pg * safe_gen).sum(-1),
            mass_entropy=-(pm * safe_mass).sum(-1),
            js_divergence=((pg * (safe_gen - log_average)).sum(-1) + (pm * (safe_mass - log_average)).sum(-1)) / 2,
            argmax_agree=pg.argmax(-1) == pm.argmax(-1),
        )
        indices = torch.arange(model.codebook_size, device=pg.device)[None]
        for name, log_probability in (("gen", gen), ("mass", mass)):
            chosen = log_probability.gather(1, target)
            row[f"{name}_rank"] = 1 + (
                valid & ((log_probability > chosen) | ((log_probability == chosen) & (indices < target)))
            ).sum(-1)
        if mixed:
            distribution = mix_log_probs(gen, mass, model.dynamic_gate.bias)
            probability = distribution.exp()
            row.update(
                mixed_probability=probability.gather(1, target).squeeze(1),
                mixed_nll=-distribution.gather(1, target).squeeze(1),
                mixed_entropy=-(probability * distribution.masked_fill(~valid, 0)).sum(-1),
                mixed_rank=1
                + (
                    valid
                    & (
                        (distribution > distribution.gather(1, target))
                        | ((distribution == distribution.gather(1, target)) & (indices < target))
                    )
                ).sum(-1),
            )
        observations.append(row)
        prefix = torch.cat((prefix, tokens[:, depth, None]), dim=1)
    return observations


class CoPMRecDiagnosis(LightningModule):
    def __init__(
        self,
        *,
        backbone,
        analysis,
        frequencies,
        checkpoint=None,
        prediction_bundles=None,
        source_reference=None,
        source_sha256=None,
        input_references=None,
        bootstrap_repetitions=2000,
        num_hierarchies=None,
        exclude_history=False,
        evaluation_protocol=None,
        method_version=None,
    ):
        super().__init__()
        if exclude_history is not False:
            raise ValueError("Diagnosis exclude_history must be boolean.")
        self.exclude_history = exclude_history
        if num_hierarchies is not None and num_hierarchies != backbone.num_hierarchies:
            raise ValueError("Diagnosis preprocessing and backbone SID depths differ.")
        if analysis not in {"hits", "residual", "prefix"}:
            raise ValueError("Unknown M3 analysis.")
        if analysis != "hits" and checkpoint is None:
            raise ValueError("Checkpoint diagnosis requires its audited own-best.")
        if analysis == "residual" and (not prediction_bundles or set(prediction_bundles) != {"full"}):
            raise ValueError("M2 requires the existing Full Testing output for V11 reproduction.")
        if analysis == "hits" and (
            not prediction_bundles or set(prediction_bundles) != {"full", "no_mixture", "no_residual", "no_joint_ce"}
        ):
            raise ValueError("M1 requires Full and all five keyed prediction bundles.")
        self.backbone, self.analysis = backbone, analysis
        self.prediction_bundles = prediction_bundles or {}
        self.frequencies = torch.as_tensor(frequencies).cpu()
        if len(self.frequencies) != len(backbone.item_keys) or not torch.equal(
            self.frequencies > 0, backbone.seen_mask.cpu()
        ):
            raise ValueError("Training frequencies must match the source catalog seen mask.")
        self.boundaries = torch.quantile(
            self.frequencies[self.frequencies > 0].float(), torch.tensor([0.33, 0.67])
        ).tolist()
        if checkpoint is not None:
            backbone.on_load_checkpoint(checkpoint)
            backbone.load_state_dict(checkpoint["state_dict"], strict=True)
        self.backbone.requires_grad_(False)
        self.rows, self.keys, self.predictions = [], [], defaultdict(list)
        self.catalog_norms = []
        self.bootstrap_repetitions = bootstrap_repetitions
        if bootstrap_repetitions < 1:
            raise ValueError("Bootstrap repetitions must be positive.")
        self.metadata = dict(
            protocol=M3_PROTOCOL,
            evaluation_protocol=evaluation_protocol or M3_PROTOCOL,
            method_version=method_version,
            ranking_history_exclusion=exclude_history,
            analysis=analysis,
            source_reference=source_reference,
            source_sha256=source_sha256,
            input_references=(
                OmegaConf.to_container(input_references, resolve=True)
                if OmegaConf.is_config(input_references)
                else input_references
            ),
            catalog_sha256=backbone.catalog_sha256,
            frequency_boundaries=self.boundaries,
            training_seed=backbone.initialization_seed,
            scope="single_dataset_single_seed",
            bootstrap=dict(rng="PCG64", seed=42, repetitions=bootstrap_repetitions, confidence=0.95),
        )

    def on_test_start(self):
        if self.trainer.world_size != 1:
            raise ValueError("M3 diagnosis requires one process.")
        self.backbone.eval()
        self.rows.clear()
        self.keys.clear()
        self.predictions.clear()
        self.catalog_norms.clear()

    @contextmanager
    def _history_residual(self, enabled):
        model = self.backbone
        # 实例方法临时替换仅包围单次 encode；退出后恢复类分派及原有方法。
        old = model.__dict__.get("item_content_residual")
        if not enabled:
            model.item_content_residual = lambda rows: None
        try:
            yield
        finally:
            if not enabled:
                if old is None:
                    del model.item_content_residual
                else:
                    model.item_content_residual = old

    def _base_rows(self, x, labels):
        model = self.backbone
        targets = model.lookup_rows(labels).cpu()
        lengths = x.attention_mask.long().sum(-1).cpu() // model.num_hierarchies
        result = []
        for key, target, length in zip(
            x.output_keys.reshape(-1).tolist(), targets.tolist(), lengths.tolist(), strict=True
        ):
            frequency = int(self.frequencies[target])
            result.append(
                dict(
                    user_key=int(key),
                    target_row=target,
                    target_seen=bool(model.seen_mask[target]),
                    frequency=frequency,
                    frequency_group="cold"
                    if frequency == 0
                    else "low"
                    if frequency <= self.boundaries[0]
                    else "middle"
                    if frequency <= self.boundaries[1]
                    else "high",
                    history_length=length,
                    history_group="0" if length == 0 else "1-5" if length <= 5 else "6-10" if length <= 10 else "11-20",
                )
            )
        return result

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        x, label = batch
        if label is None or x.output_keys is None:
            raise ValueError("M3 requires labels and business user keys.")
        labels, model = label.target_ids, self.backbone
        bases = self._base_rows(x, labels)
        self.keys.append(x.output_keys.reshape(-1).cpu())
        if self.analysis == "hits":
            predictions = {
                name: gather_predictions_by_keys(bundle, x.output_keys.cpu()).to(labels.device)
                for name, bundle in self.prediction_bundles.items()
            }
            metrics = {name: retrieval_values(pred, labels) for name, pred in predictions.items()}
            for name, pred in predictions.items():
                if pred.shape[1:] != (10, model.num_hierarchies):
                    raise ValueError("M1 requires legal unique Top10 complete SIDs.")
                rows = model.lookup_rows(pred.reshape(-1, model.num_hierarchies)).reshape(len(pred), 10)
                if any(len(row.unique()) != 10 for row in rows):
                    raise ValueError("M1 prediction has duplicate items.")
                self.predictions[name].append(pred.cpu())
                for i, base in enumerate(bases):
                    values = {key: float(value[i]) for key, value in metrics[name].items()}
                    record = {**base, "variant": name, **values}
                    for k in (5, 10):
                        current, reference = metrics[name][f"recall@{k}"][i], metrics["full"][f"recall@{k}"][i]
                        record[f"hit_class@{k}"] = (
                            "both"
                            if current and reference
                            else "variant_only"
                            if current
                            else "full_only"
                            if reference
                            else "neither"
                        )
                        delta = float(metrics[name][f"ndcg@{k}"][i] - metrics["full"][f"ndcg@{k}"][i])
                        for category in ("variant_only", "full_only", "both"):
                            record[f"contribution_{category}@{k}"] = (
                                delta if record[f"hit_class@{k}"] == category else 0.0
                            )
                    self.rows.append(record)
        elif self.analysis == "residual":
            queries = {}
            for h in (1, 0):
                with self._history_residual(h):
                    queries[h] = model.encode(x.input_ids, x.attention_mask)[2]
            projected = torch.cat(
                [model.content_projection(part) for part in model.content_bank.split(model.catalog_chunk_size)]
            )
            residual = model.item_content_residual(torch.arange(len(model.item_keys), device=projected.device))
            norms = residual.norm(dim=-1) / projected.norm(dim=-1).clamp_min(1e-12)
            if not self.catalog_norms:
                for row, (key, frequency, norm) in enumerate(
                    zip(model.item_keys.tolist(), self.frequencies.tolist(), norms.tolist(), strict=True)
                ):
                    group = (
                        "cold"
                        if frequency == 0
                        else "low"
                        if frequency <= self.boundaries[0]
                        else "middle"
                        if frequency <= self.boundaries[1]
                        else "high"
                    )
                    self.catalog_norms.append(
                        dict(
                            item_key=key,
                            catalog_row=row,
                            training_frequency=frequency,
                            frequency_group=group,
                            residual_projection_norm_ratio=norm,
                        )
                    )
            reference = gather_predictions_by_keys(self.prediction_bundles["full"], x.output_keys.cpu()).to(
                labels.device
            )
            fixed_query_logits = {}
            for h, c in ((1, 1), (1, 0), (0, 1), (0, 0)):
                raw = (
                    F.normalize(queries[h], dim=-1)
                    @ F.normalize(projected + (residual if c else 0), dim=-1).T
                    / model.temperature
                )
                if c:
                    fixed_query_logits[h] = raw
                else:
                    torch.testing.assert_close(
                        raw[:, ~model.seen_mask], fixed_query_logits[h][:, ~model.seen_mask], rtol=0, atol=0
                    )
                scores = raw
                order = scores.argsort(dim=-1, descending=True, stable=True)
                pred = model.semantic_ids[order[:, :10]]
                if h == c == 1 and not torch.equal(pred, reference):
                    raise ValueError("M2 V11 does not reproduce the existing Full Testing output.")
                overlap = (pred[:, :, None] == reference[:, None]).all(-1).any(-1).float().mean(-1)
                self.predictions[f"v{h}{c}"].append(pred.cpu())
                values = retrieval_values(pred, labels)
                for i, base in enumerate(bases):
                    target = base["target_row"]
                    eligible = bool(torch.isfinite(scores[i, target]))
                    competing = scores[i].clone()
                    competing[target] = -torch.inf
                    rank = int((order[i] == target).nonzero()[0, 0]) + 1 if eligible else None
                    self.rows.append(
                        {
                            **base,
                            "variant": f"v{h}{c}",
                            **{key: float(value[i]) for key, value in values.items()},
                            "target_eligible": eligible,
                            "target_rank": rank,
                            "target_margin": float(scores[i, target] - competing.max()) if eligible else None,
                            "target_raw_logit": float(raw[i, target]),
                            "top10_overlap_full": float(overlap[i]),
                            "residual_projection_norm_ratio": float(norms[target]),
                        }
                    )
        else:
            is_full = not hasattr(model, "variant")
            measured = prefix_observations(model, x, labels, mixed=is_full)
            variant = getattr(model, "variant", "full")
            for depth, values in enumerate(measured):
                for i, base in enumerate(bases):
                    row = {
                        **base,
                        "variant": variant,
                        "depth": depth + 1,
                        **{key: float(value[i]) for key, value in values.items()},
                        "alpha": float(model.dynamic_gate.bias.sigmoid()) if is_full else None,
                    }
                    for field in ("mixed_probability", "mixed_nll", "mixed_entropy", "mixed_rank"):
                        row.setdefault(field, None)
                    row["mass_gen_target_difference"] = row["mass_probability"] - row["gen_probability"]
                    difference = row["mass_gen_target_difference"]
                    row["difference_bin"] = (
                        "[-1,-0.1)"
                        if difference < -0.1
                        else "[-0.1,0)"
                        if difference < 0
                        else "[0,0.1)"
                        if difference < 0.1
                        else "[0.1,1]"
                    )
                    self.rows.append(row)
        return {}

    def structured_analysis_output(self):
        keys = torch.cat(self.keys)
        if keys.unique().numel() != len(keys):
            raise ValueError("Diagnosis contains duplicate user keys.")
        if self.prediction_bundles and any(
            set(bundle.keys.tolist()) != set(keys.tolist()) for bundle in self.prediction_bundles.values()
        ):
            raise ValueError("M1 outputs and testing user population differ.")
        summary, slices = [], defaultdict(list)
        slice_groups = {
            "all": ["all"],
            "seen": ["True", "False"],
            "frequency": ["cold", "low", "middle", "high"],
            "history": ["0", "1-5", "6-10", "11-20"],
            "seen_history": [
                f"{seen}/{history}" for seen in ("True", "False") for history in ("0", "1-5", "6-10", "11-20")
            ],
        }
        if self.analysis == "prefix":
            slice_groups["difference_bin"] = ["[-1,-0.1)", "[-0.1,0)", "[0,0.1)", "[0.1,1]"]
        for variant, depth in {(r["variant"], r.get("depth", 0)) for r in self.rows}:
            for kind, groups in slice_groups.items():
                for group in groups:
                    slices[(variant, depth, kind, group)]
        for row in self.rows:
            depth = row.get("depth", 0)
            for name, value in (
                ("all", "all"),
                ("seen", str(row["target_seen"])),
                ("frequency", row["frequency_group"]),
                ("history", row["history_group"]),
                ("seen_history", str(row["target_seen"]) + "/" + row["history_group"]),
            ):
                slices[(row["variant"], depth, name, value)].append(row)
            if "difference_bin" in row:
                slices[(row["variant"], depth, "difference_bin", row["difference_bin"])].append(row)
        for (variant, depth, kind, group), rows in sorted(slices.items()):
            result = dict(
                variant=variant, depth=depth, slice=kind, group=group, n=len(rows), fraction=len(rows) / len(keys)
            )
            prototype = next(r for r in self.rows if r["variant"] == variant and r.get("depth", 0) == depth)
            for field in prototype:
                vals = [r[field] for r in rows if isinstance(r[field], (int, float)) and not isinstance(r[field], bool)]
                if field not in {"user_key", "target_row", "depth"} and (
                    isinstance(prototype[field], (int, float)) or field.startswith("mixed_") or field == "alpha"
                ):
                    result[field + "_mean"] = float(np.mean(vals)) if vals else None
                    if self.analysis == "prefix":
                        for q in (0.1, 0.5, 0.9):
                            result[f"{field}_q{q}"] = float(np.quantile(vals, q)) if vals else None
            summary.append(result)
        pairs, cases = [], []
        by_variant = defaultdict(dict)
        for row in self.rows:
            if "ndcg@10" in row:
                by_variant[row["variant"]][row["user_key"]] = row
        reference = "full" if self.analysis == "hits" else "v11"
        if reference in by_variant:
            comparison_pairs = [(variant, reference) for variant in by_variant]
            for variant, reference in comparison_pairs:
                rows = by_variant[variant]
                for kind, groups in slice_groups.items():
                    for group in groups:
                        selected_keys = [
                            key
                            for key, row in sorted(rows.items())
                            if kind == "all"
                            or (
                                str(row["target_seen"])
                                if kind == "seen"
                                else row["frequency_group"]
                                if kind == "frequency"
                                else row["history_group"]
                                if kind == "history"
                                else str(row["target_seen"]) + "/" + row["history_group"]
                            )
                            == group
                        ]
                        metric_names = ("recall@5", "recall@10", "ndcg@5", "ndcg@10")
                        delta = np.array(
                            [
                                [rows[key][metric] - by_variant[reference][key][metric] for metric in metric_names]
                                for key in selected_keys
                            ]
                        )
                        rng = np.random.Generator(np.random.PCG64(42))
                        bootstrap = (
                            np.array(
                                [
                                    delta[rng.integers(len(delta), size=len(delta))].mean(0)
                                    for _ in range(self.bootstrap_repetitions)
                                ]
                            )
                            if len(delta)
                            else None
                        )
                        for j, metric in enumerate(metric_names):
                            interval = (
                                np.quantile(bootstrap[:, j], [0.025, 0.975]).tolist()
                                if bootstrap is not None
                                else [None, None]
                            )
                            pairs.append(
                                dict(
                                    variant=variant,
                                    reference=reference,
                                    metric=metric,
                                    slice=kind,
                                    group=group,
                                    n=len(selected_keys),
                                    signed_difference=float(delta[:, j].mean()) if len(delta) else None,
                                    ci_low=interval[0],
                                    ci_high=interval[1],
                                )
                            )
                if self.analysis == "hits" and reference == "full":
                    for k in (5, 10):
                        for category in ("variant_only", "full_only", "both", "neither"):
                            selected = [r for r in rows.values() if r[f"hit_class@{k}"] == category]
                            pairs.append(
                                dict(
                                    variant=variant,
                                    metric=f"hit_class@{k}",
                                    category=category,
                                    n=len(rows),
                                    count=len(selected),
                                    fraction=len(selected) / len(rows),
                                )
                            )
                            selected.sort(key=lambda r: hashlib.sha256(str(r["user_key"]).encode()).hexdigest())
                            cases.extend({**r, "case_k": k} for r in selected[:3])
                        signed = np.mean(
                            [r[f"ndcg@{k}"] - by_variant[reference][r["user_key"]][f"ndcg@{k}"] for r in rows.values()]
                        )
                        contributions = {
                            category: float(np.mean([r[f"contribution_{category}@{k}"] for r in rows.values()]))
                            for category in ("variant_only", "full_only", "both")
                        }
                        if not np.isclose(sum(contributions.values()), signed, atol=1e-8):
                            raise ValueError("M1 signed NDCG decomposition does not sum to its observed difference.")
                        pairs.append(
                            dict(
                                variant=variant,
                                metric=f"ndcg@{k}",
                                signed_difference=float(signed),
                                **{f"contribution_{c}": v for c, v in contributions.items()},
                            )
                        )
        if self.analysis == "residual":
            for metric in ("recall@5", "recall@10", "ndcg@5", "ndcg@10"):
                means = {v: np.mean([r[metric] for r in rows.values()]) for v, rows in by_variant.items()}
                pairs.append(
                    dict(metric=metric, interaction=float(means["v11"] - means["v10"] - means["v01"] + means["v00"]))
                )
        metadata = {**self.metadata, "user_count": len(keys)}
        catalog_summary = []
        if self.analysis == "residual":
            for group in ("all", "cold", "low", "middle", "high"):
                values = [
                    r["residual_projection_norm_ratio"]
                    for r in self.catalog_norms
                    if group == "all" or r["frequency_group"] == group
                ]
                catalog_summary.append(
                    dict(
                        group=group,
                        n=len(values),
                        mean=float(np.mean(values)) if values else None,
                        median=float(np.median(values)) if values else None,
                    )
                )
        return StructuredAnalysisOutput(
            documents={"summary.json": dict(metadata=metadata, observations=summary, comparisons=pairs)},
            tables={
                "users.csv": self.rows,
                "slices.csv": summary,
                "comparisons.csv": pairs,
                "cases.csv": cases,
                "catalog_norms.csv": self.catalog_norms,
                "catalog_norm_slices.csv": catalog_summary,
            },
            bundles={
                f"{name}.pt": dict(keys=keys, predictions=torch.cat(values))
                for name, values in self.predictions.items()
            },
            metadata=metadata,
        )
