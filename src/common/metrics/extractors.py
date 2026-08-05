from typing import Any


class MappingMetricExtractor:
    """Extract metric payloads from Lightning step outputs."""

    def extract(
        self,
        stage: str,
        outputs: Any,
        batch: Any,
        pl_module: Any,
    ) -> dict[str, Any]:
        if outputs is None:
            return {}
        if isinstance(outputs, dict):
            return outputs
        return {"output": outputs}
