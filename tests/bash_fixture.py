from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def bash():
    candidates = (
        Path(r"D:\development\Git\bin\bash.exe"),
        Path(r"C:\Program Files\Git\bin\bash.exe"),
    )
    executable = next((path for path in candidates if path.is_file()), None)
    if executable is None:
        pytest.skip("Git Bash is required for launcher tests.")
    return str(executable)
