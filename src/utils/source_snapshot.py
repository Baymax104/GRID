"""保存运行端实际源码，并校验由本地同步入口生成的来源记录。"""

import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ORIGIN_FILE = Path("src/.source_origin.json")
SOURCE_SUFFIXES = {"src": {".py"}, "configs": {".yaml", ".yml"}}


def _source_files(root: Path) -> list[Path]:
    files = []
    for folder, suffixes in SOURCE_SUFFIXES.items():
        base = root / folder
        if base.is_symlink():
            raise ValueError(f"Source directory must not be a symlink: {base}")
        for directory, dirs, names in os.walk(base, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__" and not d.startswith("."))
            for name in sorted(names):
                path = Path(directory) / name
                if path.suffix in suffixes:
                    if path.is_symlink():
                        raise ValueError(f"Source file must not be a symlink: {path}")
                    files.append(path)
    for path in root.iterdir():
        if path.suffix in {".sh", ".ps1"} or path.name in {"pyproject.toml", "uv.lock"}:
            if path.is_symlink():
                raise ValueError(f"Source file must not be a symlink: {path}")
            if path.is_file():
                files.append(path)
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def _entry(root: Path, path: Path, content: bytes) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _source_sha256(entries: list[dict]) -> str:
    payload = "".join(f"{e['path']}\0{e['sha256']}\n" for e in entries)
    return hashlib.sha256(payload.encode()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_source_origin(root: str | Path) -> Path:
    """仅由可信本地同步入口调用；运行端生成快照时不调用 Git。"""
    root = Path(root).resolve()
    entries = [_entry(root, p, p.read_bytes()) for p in _source_files(root)]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    path = root / ORIGIN_FILE
    _write_json(
        path,
        {
            "schema_version": 1,
            "captured_at_utc": datetime.now(UTC).isoformat(),
            "source_sha256": _source_sha256(entries),
            "git_commit": commit,
            "git_dirty": bool(status.strip()),
            "git_authority": "local-workspace",
        },
    )
    return path


def _origin(root: Path, source_sha256: str) -> dict:
    path = root / ORIGIN_FILE
    if not path.is_file() or path.is_symlink():
        return {"status": "unavailable"}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") != 1 or value.get("git_authority") != "local-workspace":
            return {"status": "invalid"}
        if value.get("source_sha256") != source_sha256:
            return {"status": "mismatch", "recorded_source_sha256": value.get("source_sha256")}
        return {**value, "status": "verified"}
    except (ValueError, AttributeError):
        return {"status": "invalid"}


def _runtime() -> dict:
    packages = {}
    for name in ("torch", "lightning", "wandb", "hydra-core", "omegaconf", "numpy", "tfrecord"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    # 不读取环境变量、认证信息或命令行参数。
    import torch

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda_build": torch.version.cuda,
    }


def create_source_snapshot(root: str | Path, output_dir: str | Path) -> dict:
    """清单 hash 和 tar 使用同一批字节；仅扫描显式源码白名单。"""
    root = Path(root).resolve()
    destination = Path(output_dir) / "metadata" / "source_snapshot"
    destination.mkdir(parents=True, exist_ok=False)
    files = _source_files(root)
    if not files:
        raise ValueError(f"No source files found in {root}")
    entries = []
    archive = destination / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in files:
            content = path.read_bytes()
            entry = _entry(root, path, content)
            entries.append(entry)
            info = tarfile.TarInfo(entry["path"])
            info.size = len(content)
            info.mode = 0o755 if path.suffix == ".sh" else 0o644
            tar.addfile(info, io.BytesIO(content))
    source_sha256 = _source_sha256(entries)
    manifest = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "source_sha256": source_sha256,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "origin": _origin(root, source_sha256),
        "files": entries,
    }
    _write_json(destination / "manifest.json", manifest)
    _write_json(destination / "runtime.json", _runtime())
    return {
        "directory": str(destination.resolve()),
        "source_sha256": source_sha256,
        "manifest_sha256": hashlib.sha256((destination / "manifest.json").read_bytes()).hexdigest(),
        "file_count": len(entries),
        "origin": manifest["origin"],
    }


def prepare_source_snapshot(cfg) -> dict | None:
    """在装配前执行；兼容 torchrun 尚未初始化进程组的阶段。"""
    options = cfg.get("source_snapshot", {})
    if cfg.get("dry_run", False) or not options.get("enabled", False):
        return None
    from lightning.pytorch.utilities.rank_zero import rank_zero_only

    from src.utils.distributed import get_distributed_rank

    # Lightning 的 subprocess DDP 在进程组初始化前可能只有 LOCAL_RANK。
    # 与 logger 共用 rank 判断，避免子进程重复创建和发布源码归档。
    if rank_zero_only.rank != 0 or int(os.environ.get("RANK", "0")) != 0 or get_distributed_rank() != 0:
        return None
    return create_source_snapshot(cfg.paths.work_dir, cfg.paths.output_dir)
