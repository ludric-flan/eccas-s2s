"""
Calibration and statistical downscaling (workflow E5 + E7, phase P3).

The chain follows the logic of the CAPC-AC method note: **each indicator is a
different operational question, so it gets its own method**. A rainfall total is
corrected by a bias correction or a quantile mapping; an anomaly by a correction
plus a standardisation; the tercile categories by a categorical or logistic
calibration; the probabilities by a probabilistic calibration (logistic, EMOS);
an event by its threshold plus a probabilistic calibration.

All of them end in the same object — a **predictive distribution on the
observation grid** (:mod:`eccas_s2s.calibrate.base`) — so that the products of
phase P7 read one interface, whatever method produced the forecast.

Every method is fitted **leave-one-year-out** (decision D10) and scored with
exactly the metrics and diagrams of phase P2, so that "calibrated" can be
compared with "raw" and with climatology on the same footing. A method is only
recommended where it beats both.
"""
from eccas_s2s.calibrate.base import (CATEGORIES, Calibrator, EnsembleDistribution,
                                      NormalDistribution, PredictiveDistribution, RawForecast)
from eccas_s2s.calibrate.bias import BiasCorrection, bias_calibrators
from eccas_s2s.calibrate.qmap import QuantileMapping, qmap_calibrators

__all__ = ["CATEGORIES", "BiasCorrection", "Calibrator", "EnsembleDistribution", "RawForecast",
           "NormalDistribution", "PredictiveDistribution", "QuantileMapping",
           "bias_calibrators", "qmap_calibrators"]
