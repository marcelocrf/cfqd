"""Common interface implemented by every method (QD or baseline)."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..model import BudgetExceeded, CountingModel
from ..schema import DatasetSpec


@dataclass
class CFResult:
    method: str
    cfs: np.ndarray                 # (k, d): returned counterfactuals (valid or not)
    n_queries: int
    seconds: float
    found_at: list[int] = field(default_factory=list)  # n_queries at which each cf appeared
    extra: dict = field(default_factory=dict)          # e.g. the QD archive


class CounterfactualMethod(ABC):
    name: str = "base"

    def explain(
        self,
        x: np.ndarray,
        model: CountingModel,
        spec: DatasetSpec,
        desired: int,
        budget: int,
        seed: int = 0,
    ) -> CFResult:
        """Called by the batch script. Do not override: override _search instead."""
        model.reset(budget)
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        try:
            res = self._search(x, model, spec, desired, rng)
        except BudgetExceeded:
            res = self._on_budget_exceeded()
        res.n_queries = model.n_queries
        res.seconds = time.perf_counter() - t0
        return res

    @abstractmethod
    def _search(self, x, model, spec, desired, rng) -> CFResult: ...

    def _on_budget_exceeded(self) -> CFResult:
        """Anytime methods (QD) return what they have found so far; override this."""
        return CFResult(self.name, np.empty((0, 0)), 0, 0.0)
