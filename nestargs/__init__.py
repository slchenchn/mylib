from .decorators import ignores, option  # noqa: F401
from .parser import *  # noqa: F401

try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version

__version__ = version("nestargs")
