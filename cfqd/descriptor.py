"""Behavior descriptor (plan, section 5.3).

Strategy axis: for each actionable group, one of three states
    0 = no change, 1 = increased, 2 = decreased  -> 3^g combinations.
Effort axis: normalized magnitude of the change (Guidotti distance),
discretized into bins.

A change only in a NOMINAL feature (no direction) is recorded as "increased" (1).
OPEN DECISION: keep it this way, add a 4th "changed" state, or handle it separately.
"""
from __future__ import annotations

import itertools

import numpy as np

from .metrics import guidotti_distance
from .schema import Constraint, DatasetSpec

NO_CHANGE, UP, DOWN = 0, 1, 2


class Descriptor:
    def __init__(self, spec: DatasetSpec, effort_edges: list[float] | None = None):
        self.spec = spec
        self.groups = spec.groups
        self.g = len(self.groups)
        # Effort bins: tune during calibration (risk: "uninformative descriptors").
        self.effort_edges = np.array(effort_edges or [0.1, 0.25, 0.5, 1.0])
        self.n_effort = len(self.effort_edges) + 1
        self._feasible = self._feasible_states()

    # ---------------------------------------------------------------- states
    def strategy(self, x_orig: np.ndarray, x_cf: np.ndarray) -> tuple[int, ...]:
        states = []
        for gname in self.groups:
            idx = self.spec.group_indices(gname)
            net, changed = 0.0, False
            for j in idx:
                f = self.spec.features[j]
                delta = x_cf[j] - x_orig[j]
                if abs(delta) < 1e-9:
                    continue
                changed = True
                if f.ftype.value != "nominal":
                    net += delta / ((f.high - f.low) or 1.0)
            if not changed:
                states.append(NO_CHANGE)
            elif net < 0:
                states.append(DOWN)
            else:
                states.append(UP)
        return tuple(states)

    def effort_bin(self, x_orig: np.ndarray, x_cf: np.ndarray) -> int:
        e = guidotti_distance(x_orig, x_cf, self.spec)
        return int(np.searchsorted(self.effort_edges, e, side="right"))

    def cell(self, x_orig: np.ndarray, x_cf: np.ndarray) -> tuple[tuple[int, ...], int]:
        return self.strategy(x_orig, x_cf), self.effort_bin(x_orig, x_cf)

    def vector(self, x_orig: np.ndarray, x_cf: np.ndarray) -> np.ndarray:
        """Continuous version for NS/surprise: one-hot state per group + effort."""
        s = np.array(self.strategy(x_orig, x_cf), dtype=float)
        onehot = np.zeros((self.g, 3))
        onehot[np.arange(self.g), s.astype(int)] = 1.0
        return np.concatenate([onehot.ravel(), [guidotti_distance(x_orig, x_cf, self.spec)]])

    # -------------------------------------------------------- feasible cells
    def _feasible_states(self) -> list[set[int]]:
        """Reachable states per group, given the constraints.

        Group with no mutable feature: NO_CHANGE only.
        Group with any mutable feature: NO_CHANGE and UP (a nominal change counts as UP).
        DOWN is added only if the group has a FREE numeric or ordinal feature.
        """
        out = []
        for gname in self.groups:
            feats = [self.spec.features[j] for j in self.spec.group_indices(gname)]
            mutable = [f for f in feats if f.mutable]
            states = {NO_CHANGE}
            if mutable:
                states.add(UP)
                can_down = any(
                    f.constraint == Constraint.FREE and f.ftype.value != "nominal" for f in mutable
                )
                if can_down:
                    states.add(DOWN)
            out.append(states)
        return out

    def feasible_strategies(self) -> list[tuple[int, ...]]:
        """Feasible combinations, excluding 'no change' in every group."""
        combos = itertools.product(*[sorted(s) for s in self._feasible])
        return [c for c in combos if any(v != NO_CHANGE for v in c)]

    @property
    def n_feasible_cells(self) -> int:
        """Coverage denominator: feasible strategies × effort bins."""
        return len(self.feasible_strategies()) * self.n_effort

    @property
    def n_total_cells(self) -> int:
        return (3**self.g - 1) * self.n_effort
