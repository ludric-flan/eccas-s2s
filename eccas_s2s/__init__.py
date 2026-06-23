"""
eccas-s2s — Seasonal and sub-seasonal climate forecasting tool for the ECCAS region.

Developed by CAPC, the Regional Climate Centre (RCC) for the ECCAS/CEEAC region.

This is an independent tool. It draws methodological inspiration from open seasonal
forecasting work (notably WASS2S for West Africa) but shares no source code with it:
every processing step is re-implemented from the underlying published method.

Two product lines are built on one shared data layer:
  1. Forecast products   — GCM / multi-model ensemble -> calibrated tercile and
                           exceedance forecasts, with skill.
  2. Driver monitoring   — observed / reanalysis state of the climate system
                           (SST / SLP / OLR / precip anomalies, Hovmoller, ENSO / IOD).

Package layout (some subpackages are placeholders for the staged roadmap):
  config            global configuration for a forecast cycle
  core              data I/O, geographic masking, decumulation, period products
  products          high-level product assembly
  io                multi-source acquisition (C3S, NMME, ERA5, CHIRPS, TAMSAT, gauges)
  monitoring        seasonal / sub-seasonal driver monitoring
  obs               blended observed climatology and station merging
  predictors        SST indices, EOF predictors
  models            statistical / ML forecast models (shared ForecastModel contract)
  calibration       bias correction and data transforms
  validate          cross-validation and verification scores
  viz               CEEAC maps and figures
  pipeline          end-to-end orchestration
"""

__version__ = "0.1.0"
