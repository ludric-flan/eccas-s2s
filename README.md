# eccas-s2s

**Seasonal and sub-seasonal climate forecasting tool for the ECCAS region.**

Developed by **CAPC**, the Regional Climate Centre (RCC) for the ECCAS / CEEAC region.
The tool produces objective, reproducible seasonal forecasts (seasonal first,
sub-seasonal in scope) over Central Africa, together with monitoring of the
seasonal/sub-seasonal climate drivers.

> **Independence & licensing.** eccas-s2s is an independent codebase. It draws
> *methodological* inspiration from open seasonal-forecasting work (notably
> [WASS2S](https://github.com/hmandela/WASS2S) for West Africa) but shares **no source
> code** with it: each processing step is re-implemented from the underlying published
> method. Licensed under **Apache-2.0** (see [LICENSE](LICENSE)).

## Two product lines, one data layer

1. **Forecast products** — GCM / multi-model ensemble → calibrated tercile and
   exceedance forecasts, with verification skill.
2. **Driver monitoring** — observed / reanalysis state of the climate system
   (SST / SLP / OLR / precipitation anomalies, Hovmöller diagrams, ENSO / IOD indices).

Data sources: **C3S** (CDS), **NMME** (IRI Data Library), **ERA5 / ERA5-Land**,
**CHIRPS**, **TAMSAT**, and national rain-gauge networks — all normalized to a canonical
`xarray` layout `(T, Y, X[, number])`.

## Status

**Phase 0 — foundation.** The existing CEEAC C3S pipeline (GCM download, decade / month /
season / custom accumulation, ensemble-mean anomalies, tercile and exceedance
probabilities, multi-model combination, CEEAC maps, verification) has been packaged here.
These products already run; later phases extend them.

## Package layout

```
eccas_s2s/
  config.py         configuration for a forecast cycle
  core/             data I/O, masking, decumulation, period products
  products/         high-level product assembly                 [roadmap]
  io/               multi-source acquisition                    [Phase 1]
  monitoring/       seasonal / sub-seasonal driver monitoring   [Phase 2]
  obs/              blended observed climatology & merging       [Phase 3]
  predictors/       SST indices, EOF predictors                  [Phase 5]
  models/           forecast models (shared ForecastModel API)   [Phase 5/6]
  calibration/      bias correction & transforms                 [Phase 8]
  validate/         cross-validation & verification scores
  viz/              CEEAC maps & figures
  pipeline.py       end-to-end orchestration
scripts/            runnable entry points (run_forecast, run_verification, ...)
tests/  notebooks/  docs/
```

## Roadmap

| Phase | Deliverable |
|------|-------------|
| 0 | Foundation: package, env, tests (current C3S products) |
| 1 | `io/` layer: C3S + **NMME** + ERA5 + CHIRPS + TAMSAT + gauges |
| 2 | Driver monitoring line |
| 3 | Observed climatology & station merging |
| 4 | Cross-validation + verification (RPSS/BSS/ROC + **GROC/CRPS**) |
| 5 | Predictors + statistical calibration (**CCA / PCR**) |
| 6 | Multi-model ensemble weighting |
| 7 | Predictand expansion (temperature, agro indices, bimodal-year aware) |
| 8 | Bias correction / daily downscaling |
| 9 | Sub-seasonal extension |
| 10 | Dashboard & dissemination (viewer, CPT / outlook outputs) |

ML models and SST-analog forecasting run as a continuous research track behind the
`ForecastModel` contract.

## Installation (development)

```bash
conda env create -f environment.yml
conda activate eccas-s2s
pip install -e .
```

## Quick check

```bash
python -c "import eccas_s2s; print(eccas_s2s.__version__)"
pytest
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
