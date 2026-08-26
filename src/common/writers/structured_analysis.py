"""Domain-neutral structured analysis output contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StructuredAnalysisOutput:
    """Named JSON documents and tabular rows ready for serialization."""

    documents: dict[str, dict[str, Any]] = field(default_factory=dict)
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
