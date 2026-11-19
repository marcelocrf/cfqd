import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from cfqd import BudgetExceeded, Constraint, CountingModel, DatasetSpec, Descriptor, Feature, FeatureType
from cfqd.metrics import constraint_violations, guidotti_distance
from cfqd.operators import mutate


@pytest.fixture
def spec():
    X = np.array([[0, 0, 0, 20], [10, 4, 3, 60]], dtype=float)
    return DatasetSpec(
        name="t", target="y", groups=["a", "b"],
        features=[
            Feature("num", FeatureType.NUMERIC, Constraint.FREE, "a"),
            Feature("ord", FeatureType.ORDINAL, Constraint.INCREASE_ONLY, "b"),
            Feature("nom", FeatureType.NOMINAL, Constraint.FREE, "a"),
            Feature("age", FeatureType.NUMERIC, Constraint.IMMUTABLE),
        ],
    ).fit_ranges(X)


def test_mutation_respects_constraints(spec):
    rng = np.random.default_rng(0)
    x = np.array([5.0, 2.0, 1.0, 30.0])
    for _ in range(500):
        y = mutate(x, x, spec, rng, p_mut=1.0, sigma=1.0)
        assert y[3] == x[3]                      # immutable
        assert y[1] >= x[1]                      # increase only
        assert constraint_violations(x, y, spec) == 0
        assert 0 <= y[0] <= 10 and y[2] in (0, 1, 2, 3)


def test_feasible_cells(spec):
    d = Descriptor(spec, effort_edges=[0.5])
    # group a: no change/up/down (3); group b: increase only (2) -> 3*2 - 1 = 5
    assert len(d.feasible_strategies()) == 5
    assert d.n_feasible_cells == 5 * 2
    assert d.n_total_cells == 8 * 2


def test_distance_to_itself_is_zero(spec):
    x = np.array([5.0, 2.0, 1.0, 30.0])
    assert guidotti_distance(x, x, spec) == 0.0


def test_budget():
    class Dummy:
        def predict_proba(self, X):
            return np.tile([0.5, 0.5], (len(X), 1))
    m = CountingModel(Dummy(), budget=10)
    m.predict(np.zeros((8, 2)))
    with pytest.raises(BudgetExceeded):
        m.predict(np.zeros((3, 2)))
    assert m.n_queries == 8
