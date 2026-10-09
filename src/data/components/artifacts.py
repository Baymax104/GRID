"""Data artifact loading helpers and W&B artifact references."""

from __future__ import annotations

import hashlib
import io
import re
from contextvars import ContextVar

import torch

from src.data.components.data_models import ModelOutput, PrefixTraceBundle
from src.data.components.prefix_trace import PREFIX_TRACE_FILENAME, load_prefix_trace
from src.utils.file import open_local_or_remote
from src.utils.pylogger import RankedLogger
from src.utils.wandb import (
    ResolvedArtifactReference,
    is_wandb_reference,
    parse_wandb_uri,
    resolve_wandb_artifact,
)

logger = RankedLogger(__name__, rank_zero_only=True)


DEFAULT_BUNDLE_FILE = "merged_predictions_tensor.pt"


def load_m3_checkpoint(reference, sha256, analysis, **kwargs):
    """无前向的列表分析不读取 checkpoint，其余诊断必须核对身份。"""
    return None if analysis == "hits" else load_audited_checkpoint(reference, sha256, **kwargs)


def load_m3_prediction_bundles(paths, analysis, **kwargs):
    """统一解析诊断使用的 keyed bundle，保留 Artifact 消费记录。"""
    if analysis == "prefix":
        return None
    names = list(paths) if analysis == "hits" else ["full"]
    return {name: load_model_output(paths[name], field_name=f"prediction_paths.{name}", **kwargs) for name in names}


def load_audited_checkpoint(reference: str, sha256: str, wandb_entity=None, wandb_project=None) -> dict:
    """通过统一引用入口读取并校验明确的 checkpoint 文件身份。"""
    if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise ValueError("Checkpoint requires an audited lowercase SHA256.")
    path = resolve_checkpoint_path(reference, default_entity=wandb_entity, default_project=wandb_project)
    if not path:
        raise ValueError("Checkpoint reference is required.")
    with open_local_or_remote(path, "rb") as stream:
        raw = stream.read()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("Checkpoint SHA256 mismatch.")
    checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a mapping.")
    return checkpoint


DEFAULT_ROLE_BY_FIELD = {
    "quantizer_checkpoint_path": "checkpoint",
    "ckpt_path": "checkpoint",
    "pretrained_checkpoint_path": "checkpoint",
    "embedding_path": "semantic_embedding",
    "cf_embedding_path": "collaborative_embedding",
    "model_output_path": "recommendation_output",
    "recommendation_output_path": "recommendation_output",
    "widened_recommendation_output_path": "recommendation_output",
    "baseline_recommendation_output_path": "recommendation_output",
    "intervention_recommendation_output_path": "recommendation_output",
    "semantic_id_path": "semantic_id",
    "fixed_prefix_trace_path": "prefix_trace",
    "widened_prefix_trace_path": "prefix_trace",
    "baseline_prefix_trace_path": "prefix_trace",
    "intervention_prefix_trace_path": "prefix_trace",
}


class ResolvedArtifactRegistry:
    """Track W&B artifact references resolved while loading data artifacts."""

    def __init__(self):
        self._references: list[ResolvedArtifactReference] = []

    def add(self, reference: ResolvedArtifactReference):
        self._references.append(reference)

    def records(self) -> tuple[ResolvedArtifactReference, ...]:
        return tuple(self._references)

    def clear(self):
        self._references.clear()


_DEFAULT_RESOLVED_ARTIFACT_REGISTRY: ContextVar[ResolvedArtifactRegistry | None] = ContextVar(
    "default_resolved_artifact_registry",
    default=None,
)


def resolve_reference(
    reference: str,
    field_name: str,
    *,
    default_entity: str | None = None,
    default_project: str | None = None,
    cache_dir: str | None = None,
    default_file: str | None = None,
    registry: ResolvedArtifactRegistry | None = None,
) -> str:
    if not is_wandb_reference(reference):
        return reference

    uri = parse_wandb_uri(reference)
    entity = uri.entity or default_entity
    project = uri.project or default_project
    if not entity or not project:
        raise ValueError(
            f"{field_name}={reference!r} uses a short W&B URI. Provide entity/project in the URI "
            "or configure experiment user/project defaults."
        )

    role = uri.role or DEFAULT_ROLE_BY_FIELD.get(field_name)
    if not role:
        raise ValueError(f"Cannot infer W&B artifact role for field {field_name!r}; add ?role=<role> to the URI.")

    target_file = uri.file or default_file or _default_file_for_role(role)
    resolved = resolve_wandb_artifact(
        uri=uri,
        field_name=field_name,
        entity=entity,
        project=project,
        role=role,
        target_file=target_file,
        cache_dir=cache_dir,
    )
    (registry or get_resolved_artifact_registry()).add(resolved)
    return resolved.resolved_path


def resolve_checkpoint_path(
    reference: str | None,
    *,
    default_entity: str | None = None,
    default_project: str | None = None,
    cache_dir: str | None = None,
) -> str | None:
    if reference is None:
        return None
    return resolve_reference(
        reference,
        field_name="ckpt_path",
        default_entity=default_entity,
        default_project=default_project,
        cache_dir=cache_dir,
    )


def get_resolved_artifact_registry() -> ResolvedArtifactRegistry:
    registry = _DEFAULT_RESOLVED_ARTIFACT_REGISTRY.get()
    if registry is None:
        registry = ResolvedArtifactRegistry()
        _DEFAULT_RESOLVED_ARTIFACT_REGISTRY.set(registry)
    return registry




def load_model_output(
    file_path: str,
    *,
    field_name: str = "model_output_path",
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    wandb_cache_dir: str | None = None,
) -> ModelOutput:
    """Load a model output from disk, sorted by key for binary search lookup."""
    resolved_path = resolve_reference(
        file_path,
        field_name=field_name,
        default_entity=wandb_entity,
        default_project=wandb_project,
        cache_dir=wandb_cache_dir,
        default_file=DEFAULT_BUNDLE_FILE,
    )
    bundle: dict[str, torch.Tensor] = torch.load(open_local_or_remote(resolved_path, mode="rb"), weights_only=False)
    if not isinstance(bundle, dict):
        raise TypeError(f"Expected model output dict at {resolved_path}, got {type(bundle).__name__}.")

    _validate_model_output(bundle)

    keys = bundle["keys"]
    if keys.unique().numel() != keys.numel():
        raise ValueError("Duplicate keys detected in model output.")

    sort_idx = keys.argsort()
    model_output = ModelOutput(keys=keys[sort_idx], predictions=bundle["predictions"][sort_idx])
    return model_output


def load_semantic_id_tensor(
    file_path: str,
    *,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    wandb_cache_dir: str | None = None,
) -> torch.Tensor:
    """Load semantic IDs from a keyed model output bundle for model-side prefix checks."""
    semantic_ids = load_model_output(
        file_path,
        field_name="semantic_id_path",
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        wandb_cache_dir=wandb_cache_dir,
    ).predictions
    if semantic_ids.ndim != 2:
        raise ValueError(
            "Semantic ID tensor must be 2-D with shape "
            f"(num_items, num_hierarchies), got shape {tuple(semantic_ids.shape)}."
        )
    return semantic_ids.long()


def load_rkmeans_codebooks(
    quantizer_checkpoint_path: str,
    expected_sha256: str,
    num_layers: int,
    codebook_size: int,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
) -> torch.Tensor:
    """读取明确身份的已训练RKMeans码本，沿用公共解析及lineage记录。"""
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("expected_sha256 must be a 64-character SHA256 digest.")
    if num_layers < 1 or codebook_size < 2:
        raise ValueError("Invalid RKMeans layer count or codebook size.")
    path = resolve_reference(
        quantizer_checkpoint_path,
        field_name="quantizer_checkpoint_path",
        default_entity=wandb_entity,
        default_project=wandb_project,
    )
    with open_local_or_remote(path, mode="rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected_sha256.lower():
        raise ValueError("RKMeans checkpoint SHA256 mismatch.")
    with open_local_or_remote(path, mode="rb") as stream:
        checkpoint = torch.load(stream, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", {}) if isinstance(checkpoint, dict) else {}
    names = {f"layers.{level}.centroids" for level in range(num_layers)}
    actual = {name for name in state if re.fullmatch(r"layers\.\d+\.centroids", name)}
    initialized = checkpoint.get("layers_initialized") if isinstance(checkpoint, dict) else None
    if actual != names or not isinstance(initialized, (list, tuple)):
        raise ValueError("RKMeans checkpoint layer structure mismatch.")
    if len(initialized) != num_layers or not all(value is True for value in initialized):
        raise ValueError("RKMeans layers must all be initialized.")
    rows = [state[f"layers.{level}.centroids"] for level in range(num_layers)]
    if any(
        not isinstance(row, torch.Tensor)
        or row.ndim != 2
        or row.shape[0] != codebook_size
        or row.shape[1] < 1
        or row.shape != rows[0].shape
        or not row.isfinite().all()
        for row in rows
    ):
        raise ValueError("RKMeans centroids must have matching finite codebook-by-feature shapes.")
    return torch.stack(rows).detach().float().contiguous()


def load_prefix_trace_artifact(
    file_path: str,
    *,
    field_name: str = "fixed_prefix_trace_path",
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    wandb_cache_dir: str | None = None,
) -> PrefixTraceBundle:
    """Resolve and load a Prefix Trace Artifact by its explicit input role."""
    resolved_path = resolve_reference(
        file_path,
        field_name=field_name,
        default_entity=wandb_entity,
        default_project=wandb_project,
        cache_dir=wandb_cache_dir,
        default_file=PREFIX_TRACE_FILENAME,
    )
    return load_prefix_trace(resolved_path)


def _default_file_for_role(role: str) -> str:
    if role in {"checkpoint", "checkpoint_last"}:
        return "*.ckpt"
    if role == "prefix_trace":
        return PREFIX_TRACE_FILENAME
    return DEFAULT_BUNDLE_FILE


def _validate_model_output(bundle: dict):
    if "keys" not in bundle or "predictions" not in bundle:
        raise ValueError("Model output must contain 'keys' and 'predictions'.")

    keys = bundle["keys"]
    predictions = bundle["predictions"]
    if not isinstance(keys, torch.Tensor) or not isinstance(predictions, torch.Tensor):
        raise TypeError("Model output fields 'keys' and 'predictions' must both be torch.Tensor.")

    if keys.ndim != 1:
        raise ValueError(f"Model output 'keys' must be 1-D, got shape {tuple(keys.shape)}.")

    if predictions.ndim == 0:
        raise ValueError("Model output 'predictions' must have at least 1 dimension.")

    if keys.size(0) != predictions.size(0):
        raise ValueError(
            f"Model output first dimension mismatch: len(keys)={keys.size(0)} vs predictions={predictions.size(0)}."
        )
