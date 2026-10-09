"""独立配置验证工具，不依赖已退役的开发版本测试。"""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import src.utils.hydra_resolvers  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


def compose_argv(argv):
    marker = argv.index("src.main")
    overrides = [arg for arg in argv[marker + 1 :] if arg != "--dry-run"]
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="main", overrides=overrides)
    cfg.paths.output_dir = "logs/test"
    cfg.paths.work_dir = "."
    cfg.paths.profile_dir = "logs/test/profile"
    OmegaConf.resolve(cfg)
    return cfg
