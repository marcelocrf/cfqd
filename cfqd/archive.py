"""Grid archive of elites (strategy × effort).

Used by MAP-Elites and also to PROJECT the counterfactuals of the baselines
(DiCE, Wachter, FACE, MOC) onto the same grid, so that coverage and QD-score
are computed for every method.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .descriptor import Descriptor
from .metrics import fitness


@dataclass
class Elite:
    x: np.ndarray
    fitness: float
    found_at: int  # n_queries when it was found (budget curves)


class GridArchive:
    def __init__(self, descriptor: Descriptor, x_orig: np.ndarray):
        self.desc = descriptor
        self.x_orig = x_orig
        self.cells: dict[tuple, Elite] = {}

    def __len__(self) -> int:
        return len(self.cells)

    def add(self, x_cf: np.ndarray, found_at: int = 0) -> bool:
        """Insert a VALID counterfactual. Return True if it filled or improved its cell."""
        strat, eff = self.desc.cell(self.x_orig, x_cf)
        if all(s == 0 for s in strat):
            return False
        key = (strat, eff)
        fit = fitness(self.x_orig, x_cf, self.desc.spec)
        cur = self.cells.get(key)
        if cur is None or fit > cur.fitness:
            self.cells[key] = Elite(x_cf.copy(), fit, found_at)
            return True
        return False

    def add_many(self, cfs: np.ndarray, found_at: int = 0) -> int:
        return sum(self.add(c, found_at) for c in np.atleast_2d(cfs))

    # ------------------------------------------------------------ QD metrics
    def coverage(self) -> float:
        """Fraction of FEASIBLE cells that are filled."""
        return len(self.cells) / self.desc.n_feasible_cells

    def qd_score(self) -> float:
        return float(sum(e.fitness for e in self.cells.values()))

    def n_strategies(self) -> int:
        return len({k[0] for k in self.cells})

    def elites(self) -> np.ndarray:
        return np.array([e.x for e in self.cells.values()])

    def coverage_curve(self, budgets: list[int]) -> list[float]:
        """Coverage as a function of budget, based on found_at."""
        n = self.desc.n_feasible_cells
        found = np.array([e.found_at for e in self.cells.values()])
        return [float(np.sum(found <= b)) / n for b in budgets]
