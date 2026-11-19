"""End-to-end demo on synthetic data (nothing to download).

Shows the flow the batch script (T13) will follow:
spec -> counting model -> method -> archive/grid -> metrics -> CSV.

Run from the repository root:  python examples/demo_synthetic.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from cfqd import Constraint, CountingModel, DatasetSpec, Feature, FeatureType, GridArchive, Descriptor
from cfqd.methods import MapElites
from cfqd.metrics import constraint_violations, l0, mean_pairwise_distance, redundancy, validity
from cfqd.results import append_csv, to_frame

rng = np.random.default_rng(0)
n = 2000
# 6 mixed-type features, 3 actionable groups
X = np.column_stack([
    rng.normal(50, 15, n),          # income (numeric, free)
    rng.normal(20, 10, n),          # debt (numeric, free)
    rng.integers(0, 5, n),          # education (ordinal, increase only)
    rng.integers(0, 4, n),          # housing type (nominal, free)
    rng.integers(18, 70, n),        # age (immutable)
    rng.integers(0, 3, n),          # past delinquencies (irreversible)
]).astype(float)
y = ((X[:, 0] - X[:, 1] + 5 * X[:, 2] - 8 * X[:, 5] + rng.normal(0, 8, n)) > 35).astype(int)

spec = DatasetSpec(
    name="synthetic", target="approved", groups=["income", "debt", "education"],
    features=[
        Feature("income", FeatureType.NUMERIC, Constraint.FREE, "income"),
        Feature("debt", FeatureType.NUMERIC, Constraint.FREE, "debt"),
        Feature("education", FeatureType.ORDINAL, Constraint.INCREASE_ONLY, "education"),
        Feature("housing", FeatureType.NOMINAL, Constraint.FREE, "income"),
        Feature("age", FeatureType.NUMERIC, Constraint.IMMUTABLE),
        Feature("delinquencies", FeatureType.ORDINAL, Constraint.IRREVERSIBLE),
    ],
).fit_ranges(X)

model = CountingModel(RandomForestClassifier(n_estimators=100, random_state=0).fit(X, y))

desc = Descriptor(spec)
print(f"Total cells: {desc.n_total_cells} | feasible: {desc.n_feasible_cells}")

# instance with an unfavorable decision
idx = int(np.flatnonzero(model.predict_uncounted(X) == 0)[0])
x = X[idx]

for budget in (1000, 5000):
    res = MapElites().explain(x, model, spec, desired=1, budget=budget, seed=0)
    arch = res.extra["archive"]
    cfs = res.cfs
    print(f"\nBudget {budget}: {len(cfs)} elites, {res.n_queries} queries, {res.seconds:.2f}s")
    print(f"  validity={validity(model, cfs, 1):.2f}  coverage={arch.coverage():.2f}  "
          f"QD-score={arch.qd_score():.2f}  strategies={arch.n_strategies()}")
    print(f"  mean L0={np.mean([l0(x, c) for c in cfs]):.2f}  "
          f"violations={sum(constraint_violations(x, c, spec) for c in cfs)}  "
          f"mean redundancy={np.mean([redundancy(model, x, c, 1) for c in cfs]):.2f}  "
          f"diversity={mean_pairwise_distance(cfs, spec):.3f}")
    append_csv(to_frame(res, spec, dataset="synthetic", model="rf", seed=0,
                        instance_id=idx, budget=budget), "results/demo.csv")

# Projecting a baseline onto the same grid: here a dummy "baseline" = any 5 elites.
baseline = GridArchive(desc, x)
baseline.add_many(cfs[:5])
print(f"\nBaseline projected onto the grid: coverage={baseline.coverage():.2f}")
print("Results written to results/demo.csv")
