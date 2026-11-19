# CFQD

**A benchmark of Quality-Diversity algorithms for counterfactual explanations.**

Course project for INFO7070 (Explainable Machine Learning, PPGInf/UFPR, Prof. Anderson Ara).
Working title: *CFQD: Benchmarking Quality-Diversity Algorithms for Counterfactual Explanations*
(formerly *Beyond Optimization: Benchmarking Quality-Diversity Algorithms for Counterfactual Explanations*).

## The idea

Most counterfactual methods treat generation as **optimization**: they return one solution,
or a small set, and tend to converge to variations of the same change strategy.
CFQD evaluates **Quality-Diversity (QD)** algorithms, which search for a repertoire of solutions
that are both good and different from each other. All methods are compared under a common protocol.

The strategy map is the common yardstick: a **strategy × effort** grid.

- **Strategy:** for each group of actionable features, whether the group increased, decreased or did not change (3^g combinations).
- **Effort:** the normalized magnitude of the change, in bins.

The counterfactuals of **every** method, QD or baseline, are projected onto this grid to measure
coverage and QD-score. MAP-Elites is the only method that uses the grid during the search.

### Research questions

1. **QD vs. optimization:** do QD algorithms find more distinct strategies than DiCE, Wachter, FACE and MOC, without losing validity, proximity and plausibility?
2. **QD vs. QD:** among MAP-Elites, NS, NSLC and surprise, which works best for counterfactuals, and does it depend on the dataset and the model?

## Methods

| Family | Method | Idea | Models |
| --- | --- | --- | --- |
| QD | MAP-Elites | keeps the best solution in each grid cell | LR, RF, NN |
| QD | Novelty Search (NS) | rewards novel behavior, ignoring quality | LR, RF, NN |
| QD | NSLC | novelty plus local competition on quality | LR, RF, NN |
| QD | Surprise / Novelty + Surprise | rewards deviation from predicted behavior | LR, RF, NN |
| Baseline | Wachter (2017) | smallest change via gradient; returns 1 counterfactual | LR, NN (no gradient for RF) |
| Baseline | DiCE (2020) | set with a diversity term (`genetic` mode for all models) | LR, RF, NN |
| Baseline | FACE (2020) | paths through dense regions of the data | LR, RF, NN (after adaptation) |
| Baseline | MOC (2020) | NSGA-II with 4 objectives; returns the Pareto front | LR, RF, NN |

LR = logistic regression, RF = random forest, NN = shallow neural network.

## Datasets

| Dataset | Source | Role |
| --- | --- | --- |
| South German Credit | UCI 522 (not the classic Statlog one) | main; validates the MOC port |
| Adult | UCI 2 (no `fnlwgt`; `education-num` only) | main |
| HELOC | OpenML **45554** (rows with only −9 dropped; −9/−8/−7 as missing) | main |
| Give Me Some Credit | Kaggle | optional |
| COMPAS | ProPublica | appendix |

## Layout

✅ = done · ➕ = to do

```text
cfqd/
├── configs/
│   ├── german_credit.yaml       ✅ constraints and groups (example, to be checked)
│   ├── adult.yaml               ➕
│   └── heloc.yaml               ➕
├── cfqd/
│   ├── schema.py                ✅ features, types, constraints, groups, ranges and MAD
│   ├── datasets.py              ➕ loaders for German, Adult and HELOC
│   ├── models.py                ➕ training of LR, RF and NN (70/30)
│   ├── model.py                 ✅ CountingModel: query counter and budget
│   ├── operators.py             ✅ constraint-aware mutation and crossover (shared with MOC)
│   ├── descriptor.py            ✅ strategy × effort and feasible cells
│   ├── archive.py               ✅ grid, coverage, QD-score, baseline projection
│   ├── metrics.py               ✅ distance, L0, validity, redundancy, violations, diversity
│   │                            ➕ plausibility (nearest neighbor, LOF), yNN
│   ├── results.py               ✅ CSV with one row per counterfactual
│   └── methods/
│       ├── base.py              ✅ common interface
│       ├── map_elites.py        ✅ MAP-Elites (minimal version)
│       ├── novelty.py           ➕ NS and NSLC
│       ├── surprise.py          ➕ surprise and novelty + surprise
│       ├── moc.py               ➕ port with pymoo (moc_original and moc_shared)
│       ├── dice.py              ➕ dice-ml adapter
│       ├── wachter.py           ➕ direct implementation
│       └── face.py              ➕ extracted from CARLA and adapted to predict_proba
├── scripts/
│   ├── run_batch.py             ➕ datasets × models × methods × seeds × budgets
│   └── validate_moc.R           ➕ original MOC (counterfactuals package) to validate the port
├── notebooks/
│   └── analysis.ipynb           ➕ Friedman, Nemenyi and map figures
├── examples/
│   ├── demo_synthetic.py        ✅ full pipeline on made-up data
│   └── simulation.py            ➕ simulation study generator (known routes)
└── tests/
    ├── test_basic.py            ✅ constraints, feasible cells and budget
    └── test_metrics.py          ➕ one test per metric
```

## Division of work

| Who | Files | Plan tasks |
| --- | --- | --- |
| **Mary** | `operators.py`, `descriptor.py`, `archive.py`, `methods/map_elites.py`, `novelty.py`, `surprise.py` | T4, T8, T10 |
| **Marcelo** | `configs/*.yaml`, `datasets.py`, `models.py`, `methods/dice.py`, `wachter.py`, `face.py`, `scripts/run_batch.py` (with Mary) | T3, T5, T6, T9, T11, T13 |
| **Vitor** | `metrics.py`, `tests/test_metrics.py`, `scripts/validate_moc.R` | T1, T7, T12 |
| **Israel** | `notebooks/analysis.ipynb` (or Vitor), literature review, writing, references | T2, T14, T18 |
| **TBD** | `methods/moc.py`, `examples/simulation.py` | T11, simulation study |

Order that unblocks the rest: `datasets.py` and `models.py` → `dice.py` and `wachter.py` → `moc.py` → `novelty.py` and `surprise.py` → `run_batch.py`.

## How to run

```bash
pip install -r requirements.txt
python -m pytest -q tests
python examples/demo_synthetic.py
```

## Plugging in a new method

```python
from cfqd.methods.base import CFResult, CounterfactualMethod

class Dice(CounterfactualMethod):
    name = "dice_genetic"

    def _search(self, x, model, spec, desired, rng):
        # ALWAYS query the model through model.predict / model.predict_proba,
        # never through the original model: that is what counts the queries.
        cfs = ...  # array (k, d), in the same encoding as spec
        return CFResult(self.name, cfs, n_queries=0, seconds=0.0)
```

`explain()` resets the counter, applies the budget, measures time and fills in `n_queries` and `seconds`.
External libraries that call the model on their own (such as DiCE) must receive the
`CountingModel` as their model, or a wrapper that forwards calls to it.

Projecting any method onto the grid:

```python
arch = GridArchive(Descriptor(spec), x)
arch.add_many(res.cfs[valid_mask])
arch.coverage(), arch.qd_score(), arch.n_strategies()
```

## Decisions made

- **Language:** Python. R is used only to validate the MOC port.
- **MOC:** Python port using pymoo's NSGA-II, in two versions: `moc_original` (the R operators, for validation) and `moc_shared` (the CFQD operators, for the main comparison).
- **CARLA:** not a dependency; the FACE code is extracted and adapted.
- **DiCE:** `genetic` mode for all three models.
- **Wachter:** does not run on RF (no gradient). Statistics: a global Friedman test without Wachter, plus a separate analysis with LR and NN only.
- **Budget:** curves across budgets instead of a single budget. Baselines may use the extra budget.
- **Validity is a constraint:** only valid counterfactuals enter the grid. **Fitness:** `1 / (1 + Guidotti distance)`.
- **Coverage:** computed over **feasible** cells, excluding cells made impossible by constraints.
- **Metrics:** names and formulas from Guidotti (2024) and CARLA, plus redundancy.
- **Constraints:** immutable, increase-only, irreversible and excluded-from-mutation are distinct categories.

## Open issues

- [ ] Fitness: include plausibility in elite quality? (plan, 5.3)
- [ ] Nominal features in the descriptor: today a change only in a nominal feature counts as "increased". Keep it, add a "changed" state, or handle it separately?
- [ ] Calibrate the effort bins on one dataset before the full run.
- [ ] Check `german_credit.yaml` against the T6 constraint table.
- [ ] HELOC: confirm the OpenML 45554 version and decide how to impute missing values.
- [ ] Simulation study: small version by Dec 3; full design in the paper extension.
