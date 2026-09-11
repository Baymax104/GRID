"""Training-only prefix priorities for bounded TIGER beam allocation probes."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PrefixAllocationConfig:
    enabled: bool = False
    reserved_slots: int = 0
    pool_multiplier: int = 1
    source_split: str = "training"
    strategy: str = "prefix_training_mass"

    def validate(self, beam_width: int) -> None:
        if beam_width <= 0:
            raise ValueError("beam_width must be positive.")
        if self.source_split != "training":
            raise ValueError(
                "Prefix allocation source_split must be 'training'; "
                f"got {self.source_split!r}."
            )
        if self.strategy != "prefix_training_mass":
            raise ValueError(f"Unsupported prefix allocation strategy: {self.strategy!r}.")
        if self.reserved_slots < 0:
            raise ValueError("reserved_slots must be non-negative.")
        if self.reserved_slots >= beam_width and self.reserved_slots > 0:
            raise ValueError("reserved_slots must be smaller than beam width.")
        if self.pool_multiplier < 1:
            raise ValueError("pool_multiplier must be at least 1.")
        if self.enabled:
            if self.reserved_slots <= 0:
                raise ValueError("reserved_slots must be positive when prefix allocation is enabled.")
        elif self.reserved_slots != 0:
            raise ValueError("reserved_slots must be zero when prefix allocation is disabled.")


class PrefixMassLookup:
    """Sparse per-depth lookup from encoded SID prefix to training mass."""

    def __init__(
        self,
        semantic_ids: torch.Tensor,
        item_frequencies: torch.Tensor,
        *,
        codebook_size: int,
        num_hierarchies: int,
    ) -> None:
        semantic_ids = torch.as_tensor(semantic_ids, dtype=torch.long)
        item_frequencies = torch.as_tensor(item_frequencies, dtype=torch.long).reshape(-1)
        if semantic_ids.ndim != 2:
            raise ValueError("semantic_ids must be a 2-D tensor for prefix allocation.")
        if semantic_ids.size(0) != item_frequencies.numel():
            raise ValueError(
                "item_frequencies must have one value per semantic ID row: "
                f"rows={semantic_ids.size(0)}, frequencies={item_frequencies.numel()}."
            )
        if semantic_ids.size(1) < num_hierarchies:
            raise ValueError("semantic_ids do not contain the configured allocation hierarchies.")
        if codebook_size <= 0 or num_hierarchies <= 0:
            raise ValueError("codebook_size and num_hierarchies must be positive.")
        semantic_ids = semantic_ids[:, :num_hierarchies].cpu()
        item_frequencies = item_frequencies.cpu()
        if torch.any((semantic_ids < 0) | (semantic_ids >= codebook_size)):
            raise ValueError("semantic_ids contain a token outside the hierarchy-local vocabulary.")
        if torch.any(item_frequencies < 0):
            raise ValueError("item_frequencies must be non-negative.")

        self.codebook_size = codebook_size
        self.num_hierarchies = num_hierarchies
        self._keys: list[torch.Tensor] = []
        self._masses: list[torch.Tensor] = []
        encoded = torch.zeros(semantic_ids.size(0), dtype=torch.long)
        for hierarchy in range(num_hierarchies):
            encoded = encoded * codebook_size + semantic_ids[:, hierarchy]
            unique_keys, inverse = torch.unique(encoded, sorted=True, return_inverse=True)
            masses = torch.zeros(unique_keys.numel(), dtype=torch.long)
            masses.scatter_add_(0, inverse, item_frequencies)
            self._keys.append(unique_keys)
            self._masses.append(masses)

        self.summary = {
            "num_catalog_items": int(semantic_ids.size(0)),
            "total_training_interactions": int(item_frequencies.sum().item()),
            "zero_frequency_items": int((item_frequencies == 0).sum().item()),
            "unique_prefixes_by_depth": [int(keys.numel()) for keys in self._keys],
        }

    def encode(self, prefixes: torch.Tensor) -> torch.Tensor:
        prefixes = torch.as_tensor(prefixes, dtype=torch.long)
        if prefixes.ndim != 2 or not 1 <= prefixes.size(1) <= self.num_hierarchies:
            raise ValueError("prefixes must be 2-D with a supported hierarchy width.")
        if torch.any((prefixes < 0) | (prefixes >= self.codebook_size)):
            raise ValueError("prefixes contain a token outside the hierarchy-local vocabulary.")
        encoded = torch.zeros(prefixes.size(0), dtype=torch.long, device=prefixes.device)
        for hierarchy in range(prefixes.size(1)):
            encoded = encoded * self.codebook_size + prefixes[:, hierarchy]
        return encoded

    def query(self, prefixes: torch.Tensor) -> torch.Tensor:
        masses, found = self.query_with_found(prefixes)
        if not bool(found.all()):
            missing = prefixes[~found][:5].tolist()
            raise ValueError(f"Legal candidate prefixes are missing allocation prior entries: {missing}.")
        return masses

    def query_with_found(self, prefixes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return masses and catalog membership without treating invalid candidates as errors."""
        encoded = self.encode(prefixes)
        keys = self._keys[prefixes.size(1) - 1].to(encoded.device)
        masses = self._masses[prefixes.size(1) - 1].to(encoded.device)
        indices = torch.searchsorted(keys, encoded)
        safe_indices = indices.clamp(max=keys.numel() - 1)
        found = (indices < keys.numel()) & (keys[safe_indices] == encoded)
        return masses[safe_indices], found
