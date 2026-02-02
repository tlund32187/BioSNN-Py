"""Public façade (stable API surface).

Only re-export supported, semver-stable symbols from here.
"""

from biosnn.api.version import __version__

__all__ = ["__version__"]
