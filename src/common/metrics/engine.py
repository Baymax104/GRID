from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
from torch import nn


class MetricEngine(nn.Module):
    """Stage-isolated runtime for torchmetrics-style metrics."""

    def __init__(self, stages: Mapping[str, Any] | None = None):
        super().__init__()
        self._input_specs: dict[str, dict[str, dict[str, Any]]] = {}
        self._output_names: dict[str, dict[str, str]] = {}
        self._stage_keys: dict[str, str] = {}
        self._last_updated_metric_keys: dict[str, set[str]] = {}
        self.metrics = nn.ModuleDict()

        for stage, stage_definitions in _to_plain_container(stages or {}).items():
            self._add_stage(stage, stage_definitions or {})

    def has_stage(self, stage: str) -> bool:
        stage_key = self._stage_keys.get(stage, "")
        return stage_key in self.metrics and len(self.metrics[stage_key]) > 0

    def update(self, stage: str, payload: Mapping[str, Any]) -> None:
        if not self.has_stage(stage):
            return

        stage_key = self._stage_keys[stage]
        self._last_updated_metric_keys[stage] = set()
        for metric_key, metric in self.metrics[stage_key].items():
            input_spec = self._input_specs[stage][metric_key]
            if hasattr(metric, "update_from_payload"):
                metric.update_from_payload(payload)
                self._last_updated_metric_keys[stage].add(metric_key)
                continue

            if "kwargs" in input_spec:
                kwargs = {
                    arg_name: _resolve_input(payload, arg_spec)
                    for arg_name, arg_spec in input_spec.get("kwargs", {}).items()
                }
                metric.update(**kwargs)
                self._last_updated_metric_keys[stage].add(metric_key)
                continue

            if "args" in input_spec:
                args = [_resolve_input(payload, arg_spec) for arg_spec in input_spec.get("args", [])]
                metric.update(*args)
                self._last_updated_metric_keys[stage].add(metric_key)
                continue

            metric.update(_resolve_input(payload, input_spec))
            self._last_updated_metric_keys[stage].add(metric_key)

    def compute(self, stage: str, only_updated: bool = False) -> dict[str, Any]:
        if not self.has_stage(stage):
            return {}

        stage_key = self._stage_keys[stage]
        computed = {}
        for metric_key, metric in self.metrics[stage_key].items():
            if only_updated and metric_key not in self._last_updated_metric_keys.get(stage, set()):
                continue
            output_name = self._output_names[stage][metric_key]
            value = metric.compute()
            if isinstance(value, dict):
                if output_name:
                    computed.update({f"{output_name}/{name}": nested_value for name, nested_value in value.items()})
                else:
                    computed.update(value)
            else:
                computed[output_name] = value
        return computed

    def reset(self, stage: str | None = None) -> None:
        stages = [stage] if stage is not None else list(self._stage_keys.keys())
        for stage_name in stages:
            if not self.has_stage(stage_name):
                continue
            stage_key = self._stage_keys[stage_name]
            for metric in self.metrics[stage_key].values():
                metric.reset()

    def log(self, pl_module: Any, stage: str, only_updated: bool = False, **log_kwargs: Any) -> dict[str, Any]:
        metrics = {
            _prefix_stage(stage, name): value for name, value in self.compute(stage, only_updated=only_updated).items()
        }
        if metrics:
            pl_module.log_dict(metrics, **log_kwargs)
        return metrics

    def _add_stage(self, stage: str, definitions: Mapping[str, Any]) -> None:
        stage_metrics = nn.ModuleDict()
        stage_input_specs: dict[str, dict[str, Any]] = {}
        stage_output_names: dict[str, str] = {}

        for definition_index, (definition_name, definition) in enumerate(definitions.items()):
            for output_name, metric, input_spec in _expand_definition(definition_name, definition):
                metric_key = _safe_module_key(f"{definition_index}_{definition_name}_{output_name}")
                stage_metrics[metric_key] = metric
                stage_input_specs[metric_key] = input_spec
                stage_output_names[metric_key] = output_name

        stage_key = _safe_module_key(f"stage_{stage}")
        self.metrics[stage_key] = stage_metrics
        self._stage_keys[stage] = stage_key
        self._input_specs[stage] = stage_input_specs
        self._output_names[stage] = stage_output_names
        self._last_updated_metric_keys[stage] = set()


def _expand_definition(name: str, definition: Any) -> list[tuple[str, nn.Module, dict[str, Any]]]:
    definition = _to_plain_container(definition)
    if not isinstance(definition, dict):
        raise TypeError(f"Metric definition '{name}' must be a mapping.")

    if "repeat" not in definition:
        output_name = definition.get("name", name)
        return [(output_name, _instantiate_metric(definition["metric"]), definition.get("input", {"key": name}))]

    repeat_definition = definition["repeat"]
    count = int(repeat_definition["count"])
    index_name = repeat_definition.get("index_name", "idx")
    name_template = repeat_definition["name_template"]
    expanded = []

    for idx in range(count):
        variables = {index_name: idx}
        output_name = name_template.format(**variables)
        metric_definition = _replace_template_values(deepcopy(repeat_definition["metric"]), variables)
        input_spec = _replace_template_values(deepcopy(repeat_definition.get("input", {"key": name})), variables)
        expanded.append((output_name, _instantiate_metric(metric_definition), input_spec))

    return expanded


def _instantiate_metric(metric_definition: Any) -> nn.Module:
    metric_definition = _to_plain_container(metric_definition)
    if isinstance(metric_definition, nn.Module):
        return metric_definition
    if isinstance(metric_definition, type):
        metric = metric_definition()
    elif isinstance(metric_definition, dict) and "_target_" in metric_definition:
        metric = hydra.utils.instantiate(metric_definition)
    else:
        raise TypeError(f"Unsupported metric definition: {metric_definition!r}")

    if not isinstance(metric, nn.Module):
        raise TypeError(f"Metric must be a torch.nn.Module, got {type(metric).__name__}.")
    return metric


def _resolve_input(payload: Mapping[str, Any], input_spec: Any) -> Any:
    input_spec = _to_plain_container(input_spec)
    if isinstance(input_spec, str):
        input_spec = {"key": input_spec}
    if not isinstance(input_spec, dict):
        raise TypeError(f"Metric input spec must be a mapping or key string, got {input_spec!r}.")

    key = input_spec["key"]
    value = payload[key]
    if "index" in input_spec:
        value = value[input_spec["index"]]
    return value


def _replace_template_values(value: Any, variables: Mapping[str, int]) -> Any:
    if isinstance(value, str):
        try:
            rendered = value.format(**variables)
        except KeyError:
            return value
        if rendered != value and rendered.isdigit():
            return int(rendered)
        return rendered
    if isinstance(value, list):
        return [_replace_template_values(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: _replace_template_values(item, variables) for key, item in value.items()}
    return value


def _to_plain_container(value: Any) -> Any:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _safe_module_key(name: str) -> str:
    return name.replace(".", "_dot_").replace("/", "_slash_")


def _prefix_stage(stage: str, name: str) -> str:
    if name.startswith(f"{stage}/"):
        return name
    return f"{stage}/{name}"
