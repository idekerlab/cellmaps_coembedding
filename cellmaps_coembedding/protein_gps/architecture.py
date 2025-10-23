"""
Deprecated ProteinGPS architecture module.

.. deprecated:: 1.5.0
   Use :mod:`cellmaps_coembedding.proteinprojector.architecture` instead.
"""

import warnings

warnings.warn(
    'cellmaps_coembedding.protein_gps.architecture is deprecated; use '
    'cellmaps_coembedding.proteinprojector.architecture instead.',
    DeprecationWarning,
    stacklevel=2
)

from cellmaps_coembedding.proteinprojector.architecture import *  # noqa: F401,F403

