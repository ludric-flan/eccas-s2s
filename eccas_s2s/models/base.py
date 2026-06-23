"""
The shared forecast-model contract.

Every statistical / ML / ensemble forecast model in eccas-s2s implements this same
three-method interface, so that models are interchangeable inside the cross-validator,
the multi-model ensemble and the operational pipeline:

    model.compute_model(X, y)   ->  cross-validated deterministic hindcast
    model.compute_prob(...)     ->  tercile probabilities (PB / PN / PA)
    model.forecast(X_new)       ->  operational forecast for the target period

This is an eccas-s2s design decision (not copied code): formalising the contract is
what turns a collection of scripts into a tool. Concrete models (CCA, PCR, MME, ...)
will subclass ForecastModel in their own modules during later phases.

Data convention: xarray objects with canonical dims ``T`` (time), ``Y``, ``X`` and,
for ensembles, ``number`` (member).
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class ForecastModel(ABC):
    """Abstract base class defining the eccas-s2s forecast-model contract."""

    #: tercile-probability estimation strategy: "bestfit" (parametric) or "nonparam"
    prob_method: str = "nonparam"

    @abstractmethod
    def compute_model(self, X, y, *args, **kwargs):
        """Fit and produce a cross-validated deterministic hindcast of ``y`` from ``X``."""
        raise NotImplementedError

    @abstractmethod
    def compute_prob(self, *args, **kwargs):
        """Convert the (hind)cast into tercile probabilities PB / PN / PA."""
        raise NotImplementedError

    @abstractmethod
    def forecast(self, X_new, *args, **kwargs):
        """Produce the operational forecast for the target period from new predictors."""
        raise NotImplementedError
