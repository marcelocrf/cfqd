"""Minimal MAP-Elites (T8), built on the shared archive and operators.

Our own implementation, to keep the same interface that NS, NSLC and surprise
(T10) will use. Switching the loop to pyribs later is optional: pyribs would
need a custom emitter for mixed-type data with constraints.
"""
from __future__ import annotations

import numpy as np

from ..archive import GridArchive
from ..descriptor import Descriptor
from ..operators import crossover, mutate, random_neighbor
from .base import CFResult, CounterfactualMethod


class MapElites(CounterfactualMethod):
    name = "map_elites"

    def __init__(self, batch_size: int = 50, n_init: int = 200, p_cross: float = 0.3,
                 effort_edges: list[float] | None = None):
        self.batch_size = batch_size
        self.n_init = n_init
        self.p_cross = p_cross
        self.effort_edges = effort_edges

    def _search(self, x, model, spec, desired, rng) -> CFResult:
        self._archive = GridArchive(Descriptor(spec, self.effort_edges), x)
        self._spec = spec

        # Initial population: neighbors of the instance with few changes.
        pop = np.array([random_neighbor(x, spec, rng, n_changes=rng.integers(1, 4))
                        for _ in range(self.n_init)])
        self._insert_valid(pop, model, desired)

        while True:  # stops when the budget runs out (BudgetExceeded)
            elites = self._archive.elites()
            if len(elites) == 0:  # nothing valid yet: explore around x
                batch = [random_neighbor(x, spec, rng, n_changes=rng.integers(1, spec.d))
                         for _ in range(self.batch_size)]
            else:
                batch = []
                for _ in range(self.batch_size):
                    a = elites[rng.integers(len(elites))]
                    if rng.random() < self.p_cross and len(elites) > 1:
                        b = elites[rng.integers(len(elites))]
                        a = crossover(a, b, x, spec, rng)
                    batch.append(mutate(a, x, spec, rng))
            self._insert_valid(np.array(batch), model, desired)

    def _insert_valid(self, batch, model, desired):
        pred = model.predict(batch)  # counts queries; may raise BudgetExceeded
        for c in batch[pred == desired]:
            self._archive.add(c, found_at=model.n_queries)

    def _on_budget_exceeded(self) -> CFResult:
        cells = self._archive.cells.values()
        return CFResult(
            method=self.name,
            cfs=self._archive.elites(),
            n_queries=0,
            seconds=0.0,
            found_at=[e.found_at for e in cells],
            extra={"archive": self._archive},
        )
