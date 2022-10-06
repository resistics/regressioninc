"""
A package for doing regression in the complex domain
"""
from importlib.metadata import version, PackageNotFoundError

__name__ = "resistics"
try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"


from resistics.letsgo import new, load  # noqa: F401
