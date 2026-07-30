"""Shared library for the NOAA phenology validation project.

used by both the data prep stage (``prep_pipeline/api``) and the screening stage (``pipeline``).

  - ``PhenoloDates``   : CCRmax phenophase date detection (SOS/MOS/DOS/EOS)
  - ``phenocam_api``   : PhenoCam metadata + 3 day summary fetch/parse helpers
  - ``DynamicTimeWrap``: NDVI vs GCC agreement scoring (gap + DTW -> divergence)
  - ``plotting``       : matplotlib time series figures with phase markers
"""

from .PhenoloDates import compute_phases
from .DynamicTimeWrap import site_agreement_score

__all__ = ["compute_phases", "site_agreement_score"]
