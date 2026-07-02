import sys


def rewrite_dry_run_flag(argv: list[str] | None = None) -> list[str]:
    """Rewrite ``--dry-run`` into a Hydra-compatible override.

    Hydra does not accept arbitrary unknown flags, so we strip the user-facing
    flag and append ``++dry_run=true`` for the composed config.
    """

    args = list(sys.argv if argv is None else argv)
    if "--dry-run" not in args:
        return args

    rewritten_args = [arg for arg in args if arg != "--dry-run"]
    rewritten_args.append("++dry_run=true")
    return rewritten_args
