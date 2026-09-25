from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_FILE = PROJECT_ROOT / "mutagen.yml"
SYNC_SCRIPT = PROJECT_ROOT / "scripts" / "mutagen_sync.ps1"
GITIGNORE = PROJECT_ROOT / ".gitignore"
AGENT_GUIDE = PROJECT_ROOT / "AGENTS.md"
REMOTE_ROOT = "node1:/data3/weizhenyu/projects/GRID"


def _sync_config() -> dict:
    return yaml.safe_load(PROJECT_FILE.read_text(encoding="utf-8"))["sync"]


def test_directory_sessions_are_narrow_one_way_replicas() -> None:
    sync = _sync_config()

    assert sync["defaults"]["mode"] == "one-way-replica"
    assert sync["defaults"]["ignore"]["vcs"] is True
    assert {
        name: (definition["alpha"], definition["beta"])
        for name, definition in sync.items()
        if name in {"grid-src", "grid-configs", "grid-scripts"}
    } == {
        "grid-src": ("./src", f"{REMOTE_ROOT}/src"),
        "grid-configs": ("./configs", f"{REMOTE_ROOT}/configs"),
        "grid-scripts": ("./scripts", f"{REMOTE_ROOT}/scripts"),
    }


def test_root_session_is_an_exact_allowlist() -> None:
    root_session = _sync_config()["grid-root-code"]

    assert root_session["alpha"] == "."
    assert root_session["beta"] == REMOTE_ROOT
    assert root_session["ignore"]["paths"] == [
        "/*",
        "!/*.sh",
        "!/pyproject.toml",
        "!/uv.lock",
    ]


def test_heavy_assets_are_outside_directory_endpoints_and_root_allowlist() -> None:
    sync = _sync_config()
    directory_alphas = {sync[name]["alpha"] for name in ("grid-src", "grid-configs", "grid-scripts")}
    root_allowlist = {pattern.removeprefix("!/") for pattern in sync["grid-root-code"]["ignore"]["paths"][1:]}

    assert directory_alphas == {"./src", "./configs", "./scripts"}
    assert sync["defaults"]["ignore"] == {"vcs": True}
    assert root_allowlist == {"*.sh", "pyproject.toml", "uv.lock"}
    assert {
        ".git",
        "data",
        "pretrained_models",
        ".venv",
        "venv",
        "logs",
        "wandb",
        "tmp",
        "tests",
        "openspec",
    }.isdisjoint(root_allowlist)


def test_management_script_preserves_the_start_gate_and_command_contract() -> None:
    script = SYNC_SCRIPT.read_text(encoding="utf-8")

    assert '"--paused"' in script
    assert '"--no-global-configuration"' in script
    assert '[ValidateSet("start", "status", "flush", "pause", "resume", "monitor", "stop")]' in script
    assert '"grid-src", "grid-configs", "grid-scripts", "grid-root-code"' in script
    assert script.index('"project", "resume"') < script.index('"project", "flush"', script.index('"resume" {'))


def test_project_lock_is_ignored_by_git() -> None:
    ignored_paths = set(GITIGNORE.read_text(encoding="utf-8").splitlines())

    assert "/mutagen.yml.lock" in ignored_paths


def test_agent_guide_records_the_sync_safety_contract() -> None:
    guide = AGENT_GUIDE.read_text(encoding="utf-8")

    assert "## 代码同步规范" in guide
    assert "本地 Git 工作区是代码的唯一可信源" in guide
    assert "远端是运行环境和重资产的可信源" in guide
    assert "one-way-replica" in guide
    assert "四个 session 均为 `Watching for changes` 且无 conflict" in guide
