from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from . import rlogger
from .rlogger import *  # noqa: F403

try:
    __version__ = _version("rlog")
except PackageNotFoundError:  # running from a source tree, not installed
    __version__ = "0.0.0.dev0"

__all__ = [*rlogger.__all__, "__version__"]
