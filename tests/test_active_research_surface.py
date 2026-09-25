from pathlib import Path

ROOT = Path(__file__).parents[1]

RETIRED_MODULE_PREFIXES = (
    "src.recommendation.brir",
    "src.recommendation.tiger_catalog_grounded",
    "src.recommendation.tiger_item_resolution",
    "src.recommendation.tiger_training_probe",
    "src.recommendation.liger.dynamic_mixture",
    "src.recommendation.liger.gate_training",
    "src.recommendation.liger.learned_reranker",
    "src.recommendation.liger.paired_rerank",
    "src.recommendation.liger.preference_dispersion",
    "src.data.components.liger_gate",
    "src.data.components.liger_learned",
    "src.data.components.item_resolution",
)

RETIRED_LAUNCHERS = (
    "brir",
    "tiger_catalog_grounded",
    "tiger_item_resolution",
    "tiger_training_probe",
    "liger_gate",
    "liger_learned",
    "liger_candidate_union",
    "liger_source_protection",
    "sid_partition_intervention",
    "sid_partition_probe",
)


def _active_text_files():
    for root, suffixes in (
        (ROOT / "src", {".py"}),
        (ROOT / "configs", {".yaml"}),
        (ROOT / "scripts", {".sh"}),
    ):
        yield from (path for path in root.rglob("*") if path.is_file() and path.suffix in suffixes)
    yield from ROOT.glob("*.sh")


def test_recommendation_packages_only_expose_current_methods():
    packages = {
        path.parent.name
        for path in (ROOT / "src" / "recommendation").glob("*/__init__.py")
    }

    assert packages == {"liger", "tiger"}


def test_active_surface_has_no_retired_imports_or_hydra_targets():
    violations = []
    for path in _active_text_files():
        text = path.read_text(encoding="utf-8")
        for prefix in RETIRED_MODULE_PREFIXES:
            if prefix in text:
                violations.append(f"{path.relative_to(ROOT)}: {prefix}")

    assert violations == []


def test_retired_launchers_and_configs_are_absent():
    active_paths = [
        path.relative_to(ROOT).as_posix().lower()
        for root in (ROOT / "configs", ROOT / "scripts")
        for path in root.rglob("*")
        if path.is_file()
    ]
    active_paths.extend(path.name.lower() for path in ROOT.glob("*.sh"))

    violations = [path for path in active_paths if any(name in path for name in RETIRED_LAUNCHERS)]
    assert violations == []
