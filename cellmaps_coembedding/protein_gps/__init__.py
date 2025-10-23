"""
Deprecated ProteinGPS module.

.. deprecated:: 1.5.0
   Use :mod:`cellmaps_coembedding.proteinprojector` instead. This module
   re-exports all functionality from ProteinProjector for backward
   compatibility.
"""

import warnings

warnings.warn(
    'cellmaps_coembedding.protein_gps is deprecated; use '
    'cellmaps_coembedding.proteinprojector instead.',
    DeprecationWarning,
    stacklevel=2
)

from cellmaps_coembedding.proteinprojector import *  # noqa: F401,F403
from cellmaps_coembedding.proteinprojector import __all__ as _projector_all

__all__ = _projector_all
