from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from prettytable import PrettyTable

from src.quantization.tail_sid_diagnosis.metrics import DiagnosisResult


def write_outputs(result: DiagnosisResult, output_dir: str, top_k_report: int = 10) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written = [
        _write_json(output_path / "summary.json", result.summary),
        _write_csv(output_path / "group_metrics.csv", result.group_rows),
        _write_csv(output_path / "item_damage_scores.csv", result.item_rows),
        _write_csv(output_path / "prefix_risk_scores.csv", result.prefix_rows),
        _write_report(output_path / "report.md", result, top_k_report=top_k_report),
    ]
    return written


def print_summary(result: DiagnosisResult, output_dir: str) -> None:
    print(f"Tail-SID diagnosis complete. Output directory: {output_dir}")
    print("")
    print("Group metrics:")
    headers = ["group", "num_items", "full_collision_rate", "near_collision_rate_strict", "avg_local_density", "avg_damage"]
    print(_pretty_table(result.group_rows, headers))
    print("")
    print("Output files:")
    for filename in ["summary.json", "group_metrics.csv", "item_damage_scores.csv", "prefix_risk_scores.csv", "report.md"]:
        print(f"- {Path(output_dir) / filename}")
    if result.item_rows:
        top_item = max(result.item_rows, key=lambda row: float(row["tail_damage"]))
        print("")
        print("Top risky item:")
        print(
            _pretty_table(
                [top_item],
                ["item_id", "group", "freq_train", "full_collision_size", "near_collision_count_strict", "damage", "tail_damage"],
            )
        )
    if result.prefix_rows:
        top_prefix = result.prefix_rows[0]
        print("")
        print("Top risky prefix:")
        print(_pretty_table([top_prefix], ["prefix_depth", "prefix", "bucket_size", "tail_ratio", "prefix_risk"]))


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_report(path: Path, result: DiagnosisResult, top_k_report: int) -> Path:
    top_k = max(0, top_k_report)
    top_items = sorted(result.item_rows, key=lambda row: float(row["tail_damage"]), reverse=True)[:top_k]
    top_prefixes = result.prefix_rows[:top_k]
    lines = [
        "# Tail-SID Resolution Damage Report",
        "",
        "## Summary",
        "",
        _markdown_table([result.summary], preferred_fields=list(result.summary)),
        "",
        "## Score Normalization",
        "",
        _markdown_table(
            [result.summary],
            preferred_fields=[
                "score_normalization_version",
                "score_normalization_method",
                "score_iqr_stability_threshold",
                "score_component_clamp_min",
                "score_component_clamp_max",
                "score_components",
                "score_degenerate_components",
            ],
        ),
        "",
        "## Group Metrics",
        "",
        _markdown_table(
            result.group_rows,
            preferred_fields=[
                "group",
                "num_items",
                "full_collision_rate",
                "near_collision_rate_strict",
                "avg_local_density",
                "avg_semantic_mismatch",
                "avg_damage",
                "avg_tail_damage",
            ],
        ),
        "",
        "## Top Risky Items",
        "",
        _markdown_table(
            top_items,
            preferred_fields=[
                "item_id",
                "group",
                "freq_train",
                "raw_sid",
                "full_collision_flag",
                "near_collision_count_strict",
                "local_density",
                "semantic_mismatch",
                "tail_damage",
            ],
        ),
        "",
        "## Top Risky Prefixes",
        "",
        _markdown_table(
            top_prefixes,
            preferred_fields=[
                "prefix_depth",
                "prefix",
                "bucket_size",
                "tail_ratio",
                "suffix_uniqueness",
                "mismatch_rate",
                "prefix_risk",
            ],
        ),
        "",
        "## Output Files",
        "",
        "- `summary.json`",
        "- `group_metrics.csv`",
        "- `item_damage_scores.csv`",
        "- `prefix_risk_scores.csv`",
        "- `report.md`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _markdown_table(rows: list[dict[str, Any]], preferred_fields: list[str]) -> str:
    if not rows:
        return "_No rows._"
    fields = [field for field in preferred_fields if any(field in row for row in rows)]
    header = "| " + " | ".join(fields) + " |"
    separator = "| " + " | ".join("---" for _ in fields) + " |"
    body = []
    for row in rows:
        values = [_format_markdown_value(row.get(field, "")) for field in fields]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body])


def _pretty_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    table = PrettyTable()
    table.field_names = fields
    for row in rows:
        table.add_row([_format_terminal_value(row.get(field, "")) for field in fields])
    return table.get_string()


def _format_terminal_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _format_markdown_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")
