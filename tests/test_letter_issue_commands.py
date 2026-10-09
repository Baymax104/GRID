"""交付的九槽位命令必须与真实根脚本参数一致；stub禁止启动训练。"""

import re
import subprocess
from pathlib import Path

from bash_fixture import bash as bash_fixture

bash = bash_fixture
ROOT = Path(__file__).resolve().parents[1]


def test_all_nine_issue_command_blocks(bash):
    text = (ROOT / "docs/letter-issue-commands-20261003.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```bash\n(.*?)\n```", text, re.S)
    assert len(blocks) == 63
    prelude = r"""uv() { printf '%s\n' "$@"; }; export -f uv
export LETTER_CF_BEST_CKPT='cf best_step=004200.ckpt'
export LETTER_CF_EMBEDDING='cf with spaces.pt'
export LETTER_CF_SOURCE='protocol=32; seed=42; checkpoint=cf best_step=004200.ckpt'
export LETTER_TOKENIZER_BEST_CKPT='tokenizer best_step=004200.ckpt'
export LETTER_SID='sid with spaces.pt'
export LETTER_BEST_CKPT='recommendation best_step=004200.ckpt'
"""
    for index, block in enumerate(blocks):
        result = subprocess.run([bash, "-n"], input=block, text=True, capture_output=True, cwd=ROOT)
        assert result.returncode == 0, result.stderr
        result = subprocess.run([bash], input=prelude + block + "\n", text=True, capture_output=True, cwd=ROOT)
        assert result.returncode == 0, (index, result.stderr)
        args = result.stdout.splitlines()
        issue = 60 + index // 7
        assert any(f"issue=BMX-{issue};" in arg for arg in args)
        assert "-m" in args and "src.main" in args
        assert any(arg.startswith("logger.wandb.group=paper_main_letter_") for arg in args)
        assert args[-1] == "input_dim=1024" if index % 7 in (2, 3) else True
        if index % 7 in (4, 5):
            assert "torchrun" in args and "--nproc_per_node=2" in args
            assert f"--master_port={29560 + index // 7}" in args
        else:
            assert "python" in args
        assert ("dry_run=true" in args) == (index % 7 == 6)
