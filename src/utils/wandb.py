"""Small W&B Artifact helpers shared by data and writer modules."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from src.utils.distributed import distributed_barrier, get_distributed_rank, is_distributed_initialized


@dataclass(frozen=True)
class WandbArtifactURI:
    original_uri: str
    run_id: str
    entity: str | None = None
    project: str | None = None
    role: str | None = None
    alias: str | None = None
    file: str | None = None


@dataclass(frozen=True)
class ResolvedArtifactReference:
    field_name: str
    original_uri: str
    resolved_path: str
    producer_run_id: str
    entity: str
    project: str
    artifact_name: str
    artifact_version: str | None
    artifact_type: str | None
    artifact_path: str
    role: str
    file: str
    metadata: dict = field(default_factory=dict)


def is_wandb_reference(value: str | os.PathLike | None) -> bool:
    return isinstance(value, str) and value.startswith("wandb://")


def parse_wandb_uri(uri: str) -> WandbArtifactURI:
    parsed = urlparse(uri)
    if parsed.scheme != "wandb":
        raise ValueError(f"Expected a wandb:// URI, got {uri!r}.")

    parts = [parsed.netloc, *[part for part in parsed.path.split("/") if part]]
    if len(parts) == 1:
        entity = None
        project = None
        run_id = parts[0]
    elif len(parts) == 3:
        entity, project, run_id = parts
    else:
        raise ValueError(
            "W&B URI must be wandb://<run-id> or wandb://<entity>/<project>/<run-id>, "
            f"got {uri!r}."
        )

    if not run_id:
        raise ValueError(f"W&B URI is missing a run id: {uri!r}.")

    query = parse_qs(parsed.query)
    return WandbArtifactURI(
        original_uri=uri,
        entity=entity,
        project=project,
        run_id=run_id,
        role=_single_query_value(query, "role"),
        alias=_single_query_value(query, "alias"),
        file=_single_query_value(query, "file"),
    )


def require_wandb_logger_run(trainer: Any, purpose: str) -> Any:
    """Return the run owned by the configured Lightning WandbLogger."""
    for experiment_logger in _trainer_loggers(trainer):
        if not _is_wandb_logger(experiment_logger):
            continue
        run = getattr(experiment_logger, "experiment", None)
        if run is not None:
            return run

    raise RuntimeError(
        f"{purpose} requires a configured Lightning WandbLogger with an active W&B run. "
        "Configure logger.wandb for this experiment."
    )


def _trainer_loggers(trainer: Any) -> list[Any]:
    loggers = getattr(trainer, "loggers", None)
    if loggers is not None:
        return list(loggers)

    single_logger = getattr(trainer, "logger", None)
    if single_logger is None:
        return []
    if isinstance(single_logger, (list, tuple)):
        return list(single_logger)
    return [single_logger]


def _is_wandb_logger(experiment_logger: Any) -> bool:
    logger_type = type(experiment_logger)
    return logger_type.__name__ == "WandbLogger" or logger_type.__module__.endswith(".wandb")


def resolve_wandb_artifact(
    *,
    uri: WandbArtifactURI,
    field_name: str,
    entity: str,
    project: str,
    role: str,
    target_file: str,
    cache_dir: str | None,
) -> ResolvedArtifactReference:
    import wandb

    api = wandb.Api()
    run_path = f"{entity}/{project}/{uri.run_id}"
    run = api.run(run_path)
    artifact = select_output_artifact(run=run, role=role, alias=uri.alias, target_file=target_file)

    download_root = wandb_artifact_download_root(cache_dir, uri.run_id, role)
    local_dir = download_artifact_once_per_distributed_run(artifact, download_root)
    resolved_path = resolve_downloaded_artifact_file(local_dir, target_file)
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"W&B artifact {getattr(artifact, 'name', '<unknown>')} for run {run_path} "
            f"does not contain file {target_file!r} after download to {local_dir}."
        )

    artifact_name = getattr(artifact, "name", "")
    artifact_version = artifact_version_name(artifact)
    artifact_path = artifact_path_name(entity=entity, project=project, artifact=artifact)
    metadata = dict(getattr(artifact, "metadata", {}) or {})
    return ResolvedArtifactReference(
        field_name=field_name,
        original_uri=uri.original_uri,
        resolved_path=str(resolved_path),
        producer_run_id=uri.run_id,
        entity=entity,
        project=project,
        artifact_name=artifact_name,
        artifact_version=artifact_version,
        artifact_type=getattr(artifact, "type", None),
        artifact_path=artifact_path,
        role=role,
        file=target_file,
        metadata=metadata,
    )


def download_artifact_once_per_distributed_run(artifact, download_root: Path) -> Path:
    if not is_distributed_initialized():
        return Path(artifact.download(root=str(download_root)))

    if get_distributed_rank() == 0:
        local_dir = Path(artifact.download(root=str(download_root)))
    else:
        local_dir = download_root

    distributed_barrier()
    return local_dir


def select_output_artifact(run, role: str, alias: str | None, target_file: str):
    artifacts = list(run.logged_artifacts())
    matches = []
    for artifact in artifacts:
        if not artifact_matches_role(artifact, role):
            continue
        if alias and alias not in set(getattr(artifact, "aliases", []) or []):
            continue
        if target_file and not artifact_contains_file(artifact, target_file):
            continue
        matches.append(artifact)

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            f"No W&B output artifact matched role={role!r}, alias={alias!r}, file={target_file!r}. "
            "Check the producer run or add explicit URI query parameters."
        )
    names = [getattr(artifact, "name", "<unknown>") for artifact in matches]
    raise ValueError(
        f"Multiple W&B output artifacts matched role={role!r}, alias={alias!r}, file={target_file!r}: {names}. "
        "Add ?alias=, ?role=, or ?file= to make the reference deterministic."
    )


def artifact_matches_role(artifact, role: str) -> bool:
    metadata = getattr(artifact, "metadata", {}) or {}
    return metadata.get("role") == role or getattr(artifact, "type", None) == role


def artifact_contains_file(artifact, target_file: str) -> bool:
    if "*" in target_file:
        return True
    try:
        files = artifact.files()
    except Exception:
        return True
    return any(getattr(file, "name", None) == target_file for file in files)


def artifact_version_name(artifact) -> str | None:
    version = getattr(artifact, "version", None)
    if version:
        return version
    name = getattr(artifact, "name", "")
    if ":" in name:
        return name.rsplit(":", 1)[1]
    return None


def artifact_path_name(*, entity: str, project: str, artifact) -> str:
    qualified = getattr(artifact, "qualified_name", None)
    if qualified:
        return qualified
    name = getattr(artifact, "name", "")
    if "/" in name:
        return name
    return f"{entity}/{project}/{name}"


def wandb_artifact_download_root(cache_dir: str | None, run_id: str, role: str) -> Path:
    base = Path(cache_dir or os.getenv("GRID_WANDB_ARTIFACT_CACHE", "logs/wandb_artifacts"))
    return base / run_id / role


def resolve_downloaded_artifact_file(local_dir: Path, target_file: str) -> Path:
    if "*" not in target_file:
        return local_dir / target_file

    matches = sorted(local_dir.glob(target_file))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        return local_dir / target_file
    raise ValueError(f"Downloaded artifact contains multiple files matching {target_file!r}: {matches}.")


def _single_query_value(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    if not values:
        return None
    if len(values) > 1:
        raise ValueError(f"W&B URI query parameter {key!r} must be specified once.")
    return values[0]
