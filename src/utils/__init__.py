from importlib import import_module


_EXPORTS = {
    "RankedLogger": ("src.utils.pylogger", "RankedLogger"),
    "enforce_tags": ("src.utils.rich_utils", "enforce_tags"),
    "extras": ("src.utils.utils", "extras"),
    "finalize_loggers": ("src.utils.logging_utils", "finalize_loggers"),
    "instantiate_callbacks": ("src.utils.instantiators", "instantiate_callbacks"),
    "instantiate_loggers": ("src.utils.instantiators", "instantiate_loggers"),
    "log_hyperparameters": ("src.utils.logging_utils", "log_hyperparameters"),
    "print_config_tree": ("src.utils.rich_utils", "print_config_tree"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)

__all__ = [
    "RankedLogger",
    "enforce_tags",
    "extras",
    "finalize_loggers",
    "instantiate_callbacks",
    "instantiate_loggers",
    "log_hyperparameters",
    "print_config_tree",
]
