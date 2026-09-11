"""Stable tensor schema and loading helpers for TIGER Prefix Trace bundles."""

from __future__ import annotations

from typing import Any

import torch

from src.data.components.data_models import PrefixTraceBundle
from src.utils.file import open_local_or_remote

PREFIX_TRACE_PAYLOAD_NAME = "prefix_trace"
PREFIX_TRACE_SCHEMA_VERSION = "tiger_prefix_trace_v1"
PREFIX_TRACE_FILENAME = "prefix_trace.pt"

LAYER_FLOAT_FIELDS = (
    "teacher_target_probability",
    "teacher_target_vs_best_legal_margin",
    "target_path_score",
    "beam_cutoff_score",
    "cutoff_margin",
)
OPTIONAL_LAYER_FLOAT_FIELDS = ("target_prefix_training_mass",)
LAYER_INTEGER_FIELDS = (
    "teacher_legal_rank",
    "target_beam_rank",
    "target_parent_beam_rank",
    "legal_candidate_count",
)
OPTIONAL_LAYER_INTEGER_FIELDS = ("allocation_reserved_count",)
LAYER_BOOLEAN_FIELDS = ("target_prefix_survived",)
OPTIONAL_LAYER_BOOLEAN_FIELDS = (
    "target_allocation_shortlisted",
    "target_selected_by_reserve",
)
SCALAR_INTEGER_FIELDS = ("first_failure_depth",)
REQUIRED_TRACE_FIELDS = (
    *LAYER_FLOAT_FIELDS,
    *LAYER_INTEGER_FIELDS,
    *LAYER_BOOLEAN_FIELDS,
    *SCALAR_INTEGER_FIELDS,
)
TRACE_FIELDS = (
    *REQUIRED_TRACE_FIELDS,
    *OPTIONAL_LAYER_FLOAT_FIELDS,
    *OPTIONAL_LAYER_INTEGER_FIELDS,
    *OPTIONAL_LAYER_BOOLEAN_FIELDS,
)
RANK_FIELDS = (
    "teacher_legal_rank",
    "target_beam_rank",
    "target_parent_beam_rank",
)
REQUIRED_METADATA_FIELDS = (
    "data_split",
    "beam_width",
    "num_hierarchies",
    "codebook_size",
    "trace_mode",
    "checkpoint_reference",
    "semantic_id_reference",
)


def validate_prefix_trace_bundle(bundle: dict[str, Any]) -> None:
    """Normalize legacy tensor-compatible fields and validate the bundle in place."""
    required = {"schema_version", "keys", "labels", "trace", "metadata"}
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError(f"Prefix Trace bundle is missing fields: {missing}.")
    if bundle["schema_version"] != PREFIX_TRACE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported Prefix Trace schema version: "
            f"{bundle['schema_version']!r}; expected {PREFIX_TRACE_SCHEMA_VERSION!r}."
        )

    keys = _coerce_tensor(bundle["keys"], "keys")
    labels = _coerce_tensor(bundle["labels"], "labels")
    trace = bundle["trace"]
    if isinstance(trace, dict):
        trace = {
            field_name: _coerce_tensor(value, field_name)
            for field_name, value in trace.items()
        }
    bundle["keys"] = keys
    bundle["labels"] = labels
    bundle["trace"] = trace
    metadata = bundle["metadata"]
    if not isinstance(keys, torch.Tensor) or keys.ndim != 1:
        raise ValueError("Prefix Trace keys must be a 1-D torch.Tensor.")
    if keys.unique().numel() != keys.numel():
        raise ValueError("Duplicate keys detected in Prefix Trace bundle.")
    if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
        raise ValueError("Prefix Trace labels must be a 2-D torch.Tensor.")
    if labels.size(0) != keys.size(0):
        raise ValueError("Prefix Trace keys and labels must have the same first dimension.")
    if not isinstance(trace, dict):
        raise TypeError("Prefix Trace trace field must be a dictionary of tensors.")
    if not isinstance(metadata, dict):
        raise TypeError("Prefix Trace metadata must be a dictionary.")

    missing_trace = sorted(set(REQUIRED_TRACE_FIELDS) - set(trace))
    if missing_trace:
        raise ValueError(f"Prefix Trace is missing tensors: {missing_trace}.")
    unexpected_trace = sorted(set(trace) - set(TRACE_FIELDS))
    if unexpected_trace:
        raise ValueError(f"Prefix Trace contains unsupported tensors: {unexpected_trace}.")
    missing_metadata = sorted(set(REQUIRED_METADATA_FIELDS) - set(metadata))
    if missing_metadata:
        raise ValueError(f"Prefix Trace metadata is missing fields: {missing_metadata}.")

    num_rows, num_hierarchies = labels.shape
    if int(metadata["num_hierarchies"]) != num_hierarchies:
        raise ValueError(
            "Prefix Trace hierarchy width does not match metadata: "
            f"labels={num_hierarchies}, metadata={metadata['num_hierarchies']}."
        )
    if metadata["data_split"] not in {"evaluation", "testing"}:
        raise ValueError("Prefix Trace data_split must be 'evaluation' or 'testing'.")
    if int(metadata["beam_width"]) <= 0 or int(metadata["codebook_size"]) <= 0:
        raise ValueError("Prefix Trace beam_width and codebook_size must be positive.")

    present_optional_layer_fields = tuple(
        field_name
        for field_name in (
            *OPTIONAL_LAYER_FLOAT_FIELDS,
            *OPTIONAL_LAYER_INTEGER_FIELDS,
            *OPTIONAL_LAYER_BOOLEAN_FIELDS,
        )
        if field_name in trace
    )
    for field_name in (
        *LAYER_FLOAT_FIELDS,
        *LAYER_INTEGER_FIELDS,
        *LAYER_BOOLEAN_FIELDS,
        *present_optional_layer_fields,
    ):
        value = trace[field_name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (num_rows, num_hierarchies):
            shape = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
            raise ValueError(
                f"Prefix Trace tensor {field_name!r} must have shape "
                f"({num_rows}, {num_hierarchies}), got {shape}."
            )
    first_failure = trace["first_failure_depth"]
    if not isinstance(first_failure, torch.Tensor) or tuple(first_failure.shape) != (num_rows,):
        raise ValueError(
            f"Prefix Trace tensor 'first_failure_depth' must have shape ({num_rows},)."
        )

    for field_name in LAYER_FLOAT_FIELDS:
        if not trace[field_name].dtype.is_floating_point:
            raise TypeError(f"Prefix Trace tensor {field_name!r} must have floating dtype.")
    for field_name in OPTIONAL_LAYER_FLOAT_FIELDS:
        if field_name in trace and not trace[field_name].dtype.is_floating_point:
            raise TypeError(f"Prefix Trace tensor {field_name!r} must have floating dtype.")
    for field_name in (*LAYER_INTEGER_FIELDS, *SCALAR_INTEGER_FIELDS):
        if trace[field_name].dtype not in {torch.int8, torch.int16, torch.int32, torch.int64}:
            raise TypeError(f"Prefix Trace tensor {field_name!r} must have integer dtype.")
    for field_name in OPTIONAL_LAYER_INTEGER_FIELDS:
        if field_name in trace and trace[field_name].dtype not in {torch.int8, torch.int16, torch.int32, torch.int64}:
            raise TypeError(f"Prefix Trace tensor {field_name!r} must have integer dtype.")
    for field_name in LAYER_BOOLEAN_FIELDS:
        if trace[field_name].dtype != torch.bool:
            raise TypeError(f"Prefix Trace tensor {field_name!r} must have bool dtype.")
    for field_name in OPTIONAL_LAYER_BOOLEAN_FIELDS:
        if field_name in trace and trace[field_name].dtype != torch.bool:
            raise TypeError(f"Prefix Trace tensor {field_name!r} must have bool dtype.")

    for field_name in RANK_FIELDS:
        values = trace[field_name]
        if torch.any((values != -1) & (values < 1)):
            raise ValueError(f"Prefix Trace rank {field_name!r} must be 1-based or -1.")
    if torch.any(trace["legal_candidate_count"] < 0):
        raise ValueError("Prefix Trace legal_candidate_count must be non-negative.")
    if torch.any((first_failure != -1) & ((first_failure < 1) | (first_failure > num_hierarchies))):
        raise ValueError("Prefix Trace first_failure_depth must be 1-based or -1.")

    survived = trace["target_prefix_survived"]
    beam_rank = trace["target_beam_rank"]
    if torch.any(survived & (beam_rank < 1)) or torch.any((~survived) & (beam_rank != -1)):
        raise ValueError("Prefix Trace survival and target_beam_rank sentinels are inconsistent.")


def _coerce_tensor(value: Any, field_name: str) -> torch.Tensor:
    """Canonicalize legacy list payloads while retaining strict shape/dtype checks."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (str, bytes, dict)):
        raise ValueError(f"Prefix Trace field {field_name!r} must be tensor-compatible.")
    try:
        return torch.as_tensor(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"Prefix Trace field {field_name!r} cannot be converted to a tensor.") from error


def load_prefix_trace(file_path: str) -> PrefixTraceBundle:
    """Load and validate one Prefix Trace bundle from a resolved local path."""
    serialized = torch.load(open_local_or_remote(file_path, mode="rb"), weights_only=False)
    if not isinstance(serialized, dict):
        raise TypeError(f"Expected Prefix Trace dict at {file_path}, got {type(serialized).__name__}.")
    validate_prefix_trace_bundle(serialized)
    sort_idx = serialized["keys"].argsort()
    return PrefixTraceBundle(
        schema_version=serialized["schema_version"],
        keys=serialized["keys"][sort_idx].long(),
        labels=serialized["labels"][sort_idx].long(),
        trace={name: value[sort_idx] for name, value in serialized["trace"].items()},
        metadata=dict(serialized["metadata"]),
    )
