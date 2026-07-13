from datetime import datetime

import pytz
from omegaconf import OmegaConf

"""
Hydra allows for custom resolvers, which are functions that can be used to resolve values in the config.
For example, one can manipulate strings or apply simple python functions to the config values.
"""


def now_of_timezone(pattern: str, timezone: str = "Asia/Shanghai") -> str:
    tz = pytz.timezone(timezone)
    return datetime.now(tz).strftime(pattern)


# resolvers need to be registered to be accessible during config composition.
# The resolver name is the function name without the type annotations.
OmegaConf.register_new_resolver("now_tz", now_of_timezone)
