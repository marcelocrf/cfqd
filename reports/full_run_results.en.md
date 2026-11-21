# CFQD — Full run: results and concerns

*Branch `marcelo-21nov` · 2026-11-21 · Portuguese version: [full_run_results.pt-BR.md](full_run_results.pt-BR.md)*

All numbers come from `results/full/summary/*.csv` (700 configurations, 70,000 instance × budget rows)
and `results/full/manifest.jsonl`, plus the outputs saved in notebooks 03 and 05.
Unless a section says otherwise, numbers are at the **largest budget (10,000 queries)**. Each
(dataset, model, method, seed) is first averaged over its 20 instances, then averaged across seeds, models and datasets.

## Summary

1. **MAP-Elites has the highest coverage in all 150 (dataset, model, seed) blocks.** It fills 41% of the feasible
   strategy × effort cells on average. The best baseline fills 6%. The gap is 11× to 30× on German Credit, Adult,
   HELOC and GMC, and only 2.2× on COMPAS, whose grid has 10 cells.
2. **No method violated a constraint** (0 violations in every configuration). MAP-Elites finds a valid counterfactual
   for 97.7% of the instances. Wachter (LR and NN only) finds one for 98.5%. FACE is the weakest at 85.5%.
3. **MAP-Elites's closest counterfactual is usually closer than the baselines' closest.** It beats DiCE-genetic,
   Wachter and FACE on 99% of the paired instances and DiCE-random on 84%.
4. **These wins are not yet a fair answer to RQ1.** Coverage is capped by how many counterfactuals a method returns
   (1 for Wachter and FACE, at most 10 for DiCE, 85 on average for MAP-Elites). MAP-Elites is also the only method
   that sees the grid and the only one whose fitness is the evaluation distance. It also uses the whole budget while
   the baselines stop after a median of 1 to 2,031 queries. Per returned counterfactual, DiCE is about as diverse
   in strategies as MAP-Elites, or more (concern C1).
5. **The run mixes code versions.** 438 of the 700 configurations ran on source code older than the committed code,
   including every Adult, German Credit and HELOC configuration (concern C4).
6. **RQ2 (QD vs. QD) cannot be addressed yet.** NS, NSLC, surprise and MOC are not implemented. Plausibility is not
   measured either, so the "without losing plausibility" part of RQ1 is still open.

---

## 1. What was run

| | |
| --- | --- |
| Datasets | German Credit, Adult, HELOC, Give Me Some Credit (GMC), COMPAS (Guidotti actionability). Strict COMPAS is skipped: no actionable feature. |
| Models | LR, RF, NN (PyTorch), one model per seed |
| Methods | MAP-Elites (the only QD method so far), DiCE-genetic, DiCE-random, Wachter (LR and NN only), FACE-kNN |
| Seeds | 10 (0–9). Each seed controls the split, the training and the search. |
| Instances | 20 per configuration, drawn from test rows **predicted unfavorable by that model**. All methods share the same 20 instances within a (dataset, model, seed), so method comparisons are paired. |
| Budgets | 500, 1,000, 2,000, 5,000, 10,000 model queries. Each budget is an independent run with the same search seed. |
| Configurations | 5 datasets × 3 models × 5 methods × 10 seeds − 50 (Wachter on RF) = **700** |
| Compute | 36.5 CPU-hours recorded (675 timed configurations), finished between 2026-11-20 17:49 and 2026-11-21 00:40 UTC |

### Size of the strategy × effort grid

Effort bins use the edges `[0.1, 0.25, 0.5, 1.0]`, which gives 5 bins. Coverage is measured over **feasible** cells.

| Dataset | Features | Groups (g) | Feasible strategies | Feasible cells |
| --- | ---: | ---: | ---: | ---: |
| German Credit | 20 | 4 | 80 | 400 |
| Adult | 12 | 4 | 35 | 175 (of 400) |
| HELOC | 26 | 4 | 80 | 400 |
| GMC | 10 | 3 | 26 | 130 |
| COMPAS (Guidotti) | 9 | 1 | 2 | 10 |

### Models explained (test accuracy, seed 0, from `reports/accuracies.csv`)

| Dataset | LR | RF | NN | Majority class | Balanced accuracy (LR / RF / NN) |
| --- | ---: | ---: | ---: | ---: | --- |
| German Credit | 0.733 | 0.717 | 0.713 | 0.700 | 0.68 / 0.63 / 0.60 |
| Adult | 0.849 | 0.865 | 0.857 | 0.752 | 0.77 / 0.78 / 0.78 |
| HELOC | 0.717 | 0.724 | 0.722 | 0.520 | 0.72 / 0.72 / 0.72 |
| GMC | 0.935 | 0.935 | 0.934 | 0.931 | 0.57 / 0.58 / 0.58 |
| COMPAS (Guidotti) | 0.672 | 0.675 | 0.673 | 0.545 | 0.66 / 0.67 / 0.66 |

---

## 2. Results

### 2.1 Overview at budget 10,000

All datasets and models. Wachter covers LR and NN only.

| Method | Success | Coverage | Cells filled | Strategies | CFs returned | Median closest distance | Median L0 | Revertible changes¹ | Median queries used | Seconds / instance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **MAP-Elites** | 0.977 | **0.409** | **85.5** | **27.1** | 85.5 | **0.106** | 3.5 | 64% | 10,000 | 5.2 |
| DiCE-genetic | 0.942 | 0.056 | 4.3 | 3.5 | 8.9 | 0.410 | 4.4 | 68% | ~830 | 10.4 |
| DiCE-random | 0.943 | 0.060 | 5.1 | 3.9 | 9.1 | 0.262 | **1.7** | **33%** | 2,031 | 0.6 |
| Wachter | **0.985** | 0.022 | 1.0 | 1.0 | 1.0 | 0.535 | 5.0 | 53% | 9 | 0.6 |
| FACE | 0.855 | 0.021 | 0.9 | 0.9 | 0.9 | 0.667 | 5.0 | 69% | 1 | 0.1 |

¹ CARLA redundancy divided by L0: the share of changed features that could be reverted without losing the desired class.

**Comment.** MAP-Elites dominates every diversity metric (coverage, cells, strategies) by an order of magnitude.
Its closest counterfactual is also the closest of all methods. Validity is not traded away: success is on par with
the baselines, and every returned counterfactual respects the constraints. The price is cost: MAP-Elites always
spends the full budget. DiCE-random, nominally the "floor", is the sparsest method (1.7 changed features) and has the
fewest unnecessary changes. Its coverage and proximity also beat DiCE-genetic (see C9).

### 2.2 Coverage per dataset

Mean over models and seeds, at budget 10,000.

| Dataset | MAP-Elites | DiCE-genetic | DiCE-random | Wachter | FACE | MAP-Elites ÷ best baseline | Maximum coverage with 10 CFs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| German Credit | **0.340** | 0.019 | 0.018 | 0.002 | 0.002 | 18× | 0.025 |
| Adult | **0.371** | 0.019 | 0.035 | 0.006 | 0.005 | 11× | 0.057 |
| HELOC | **0.392** | 0.011 | 0.013 | 0.002 | 0.002 | 30× | 0.025 |
| GMC | **0.505** | 0.035 | 0.039 | 0.008 | 0.007 | 13× | 0.077 |
| COMPAS | **0.438** | 0.199 | 0.194 | 0.094 | 0.090 | 2.2× | 1.000 |

**Comment.** The last column is the coverage a method returning 10 counterfactuals could reach at most. On German
Credit and HELOC, DiCE could never exceed 2.5% coverage even if each of its 10 counterfactuals landed in a different
cell. The size of the coverage gap is therefore mostly set by set size, not by search quality (C1). COMPAS shows the
opposite problem: with 1 group, the grid has only 2 strategies, so it cannot express strategy diversity (C11).
Variability across seeds is small: the standard deviation of MAP-Elites coverage within a (dataset, model) is 0.02–0.11.

### 2.3 Budget curves

Mean over all datasets, models and seeds.

| Budget | 500 | 1,000 | 2,000 | 5,000 | 10,000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MAP-Elites: coverage | 0.240 | 0.292 | 0.335 | 0.381 | 0.409 |
| MAP-Elites: cells filled | 39.4 | 52.6 | 64.2 | 77.1 | 85.5 |
| MAP-Elites: success | 0.955 | 0.967 | 0.973 | 0.975 | 0.977 |
| DiCE-genetic: coverage | 0.057 | 0.056 | 0.056 | 0.056 | 0.056 |
| DiCE-random: coverage | **0.000** | **0.000** | **0.000** | 0.050 | 0.060 |
| Wachter: coverage | 0.022 | 0.022 | 0.022 | 0.022 | 0.022 |
| FACE: coverage | 0.021 | 0.021 | 0.021 | 0.021 | 0.021 |

**Comment.**
- **MAP-Elites has not saturated at 10,000 queries.** Going from 5,000 to 10,000 queries still adds 8 cells on
  average. On German Credit, coverage rises from 0.12 to 0.34 across the curve.
- **The baselines are flat.** DiCE-genetic, Wachter and FACE converge with very few queries and ignore the extra budget.
  This confirms the observation in notebook 04 that motivated the curve design.
- **DiCE-random returns nothing at budgets ≤ 2,000.** Its first call costs about 2,031 queries and it has no anytime
  fallback. This comes from our adapter, not from the method (C9).

### 2.4 Validity, success and constraints

Share of instances with at least one valid counterfactual, at budget 10,000.

| Method | Adult | COMPAS | German Credit | GMC | HELOC |
| --- | ---: | ---: | ---: | ---: | ---: |
| MAP-Elites | 1.000 | 0.947 | 1.000 | 0.957 | 0.982 |
| DiCE-genetic | 1.000 | 0.922 | 1.000 | 0.878 | 0.912 |
| DiCE-random | 1.000 | 0.857 | 1.000 | 0.955 | 0.902 |
| Wachter (LR, NN) | 1.000 | 0.940 | 1.000 | 0.985 | 1.000 |
| FACE | 0.845 | 0.900 | 0.882 | 0.873 | 0.777 |

**Comment.**
- **Constraints hold everywhere:** 0 violations in 70,000 rows, for every method, including immutable,
  increase-only and sentinel rules. This is a strong result for the shared repair operators and the DiCE, Wachter
  and FACE adapters.
- **Wachter has the highest overall success** (0.985, LR and NN only) and is the best on GMC and HELOC.
  MAP-Elites misses some instances on GMC, COMPAS and HELOC even with 10,000 queries, but it is the best on COMPAS.
- **FACE is the weakest.** It returns real data points, and projecting them onto the instance's frozen values
  (irreversible counters, age) often loses the desired class. Notebook 03 treats this as a property of FACE under
  actionability constraints, not as a bug.
- **DiCE-genetic is weak on GMC** (0.83 with LR, 0.81 with NN), where the CARLA check also found low numbers (§2.8).

### 2.5 Proximity and sparsity

For each instance where both methods found a valid counterfactual, we compared their closest counterfactuals
(Guidotti distance) at budget 10,000.

| MAP-Elites vs. | Paired instances | MAP-Elites closer | Median ratio (MAP-Elites ÷ baseline) |
| --- | ---: | ---: | ---: |
| DiCE-genetic | 2,826 | 99% | 0.29 |
| DiCE-random | 2,819 | 84% | 0.55 |
| Wachter | 1,950 | 99% | 0.24 |
| FACE | 2,565 | 99% | 0.17 |

Median closest distance by dataset:

| Method | Adult | COMPAS | German Credit | GMC | HELOC |
| --- | ---: | ---: | ---: | ---: | ---: |
| MAP-Elites | **0.079** | **0.468** | **0.050** | **0.173** | **0.197** |
| DiCE-genetic | 0.423 | 0.667 | 0.239 | 0.462 | 0.536 |
| DiCE-random | 0.248 | 0.889 | **0.050** | 0.467 | 0.264 |
| Wachter | 0.389 | 1.198 | 0.260 | **53.7** | 0.423 |
| FACE | 0.536 | 1.222 | 0.475 | 0.680 | 0.795 |

**Comment.**
- The QD method finds the closest counterfactual even though it is not built to minimize distance. Two things
  qualify this. MAP-Elites's fitness *is* the evaluation distance, `1 / (1 + Guidotti distance)` (C2). It also spends
  10,000 queries where DiCE-genetic stops at about 830 (C3).
- **Wachter on GMC (median 53.7, mean 152 with LR)** is a scale problem, not a search failure. GMC has heavy-tailed
  features with small MADs. A small step in Wachter's min-max space becomes huge in MAD units (C7). Means that
  include GMC are dominated by these values, which is why this report uses medians.
- **No method is sparse enough.** About two thirds of the features changed by MAP-Elites, DiCE-genetic and FACE could
  be reverted without losing the desired class (C12).

### 2.6 Statistical tests

Friedman tests with one block per (dataset, model), averaged over seeds, and Nemenyi critical difference (CD) at α = 0.05.
Lower rank is better.

**Without Wachter: 15 blocks, 4 methods, CD = 1.21**

| Metric | MAP-Elites | DiCE-random | DiCE-genetic | FACE | Friedman |
| --- | ---: | ---: | ---: | ---: | --- |
| Coverage | **1.00** | 2.27 | 2.73 | 4.00 | χ² = 41.5, p = 5e-9 |
| Strategies | **1.00** | 2.23 | 2.77 | 4.00 | χ² = 42.1, p = 4e-9 |
| Success | **1.70** | 2.43 | 2.20 | 3.67 | χ² = 23.8, p = 3e-5 |
| Median closest distance | **1.10** | 2.23 | 2.67 | 4.00 | χ² = 39.6, p = 1e-8 |

**LR and NN only, with Wachter: 10 blocks, 5 methods, CD = 1.93**

| Metric | MAP-Elites | DiCE-random | DiCE-genetic | Wachter | FACE | Friedman |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Coverage | **1.00** | 2.10 | 2.90 | 4.00 | 5.00 | χ² = 39.3, p = 6e-8 |
| Success | 2.30 | 2.75 | 3.05 | **2.10** | 4.80 | χ² = 25.4, p = 4e-5 |
| Median closest distance | **1.10** | 2.30 | 3.00 | 4.00 | 4.60 | χ² = 30.9, p = 3e-6 |

**Comment.**
- **Coverage, 15 blocks:** MAP-Elites differs significantly from every baseline, though only barely from DiCE-random
  (rank difference 1.27 vs. CD 1.21).
- **Success:** MAP-Elites is not significantly different from either DiCE variant or from Wachter.
- **Closest distance:** MAP-Elites vs. DiCE-random falls under the CD (1.13 < 1.21).
- **With Wachter (10 blocks):** Nemenyi does **not** separate MAP-Elites from either DiCE variant on coverage, even
  though MAP-Elites wins in 10 of 10 blocks (two-sided sign test p = 0.002). With this few blocks Nemenyi is very
  conservative (C13).
- Treating seeds as blocks (150 blocks) gives p ≈ 1e-89. That is pseudo-replication and should not be reported.

### 2.7 Simulation study (quick version, notebook 05)

One seed and 5 instances per model. These results are indicative only, and they are not versioned in `results/`.

| Method | Route recall, aligned descriptor (500 / 2,000 / 10,000) | Route recall, shuffled descriptor (at 10,000) |
| --- | --- | ---: |
| MAP-Elites | 0.78 / 0.72 / 0.73 | 0.70 |
| DiCE-genetic | 0.37 / 0.37 / 0.37 | 0.37 |
| FACE | 0.23 / 0.23 / 0.23 | 0.23 |
| Wachter | 0.05 / 0.05 / 0.05 | 0.05 |

**Comment.**
- **MAP-Elites finds about twice as many routes as DiCE.** Its recall barely changes when the descriptor groups are
  shuffled (−0.03), so its advantage does not come from a grid aligned with the routes. This answers Mary's concern
  about a "rigged" simulation, at least in the quick run.
- **MAP-Elites still misses routes** in 56 of its 90 instance × budget runs.
- **Recall can drop as the budget grows** (0.78 → 0.73). A closer counterfactual on another route can replace an
  elite, so the archive's fitness rule can delete route diversity.
- **The LR result is limited by the model:** LR cannot represent the OR between routes (MAP-Elites recall 0.55 with
  LR vs. 0.90 with NN, at budgets 500 and 2,000).

### 2.8 Check against CARLA (notebook 03)

10 instances, seed 0, budget 10,000. These are the cases that differ from the reference used in `build_report_data.py`
by 0.2 or more.

| Method | Dataset | Model | Ours | Reference |
| --- | --- | --- | ---: | ---: |
| DiCE-genetic | GMC | NN | 0.5 | 1.00 |
| FACE | GMC | NN | 0.4 | 1.00 |
| FACE | Adult | LR, NN | 0.8 | 1.00 |
| Wachter | GMC | NN | 0.8 | 1.00 |
| Wachter | Adult | NN | 1.0 | 0.50 |
| Wachter | COMPAS | NN | 1.0 | 0.66 |

Four more cases sit at 0.9 vs. 1.00, i.e. one failed instance out of 10: DiCE-genetic and FACE on COMPAS + LR and on
GMC + LR. Everything else matches. Wachter beats CARLA on Adult and COMPAS (NN). This was already flagged in the plan:
CARLA ignores constraints, uses its own models and a 0.5-minute timeout. GMC + NN is the weak spot for DiCE, FACE
and Wachter alike.

---

## 3. Where the research questions stand

- **RQ1 (QD vs. optimization):** the evidence points to "yes" for MAP-Elites against DiCE, Wachter and FACE. MAP-Elites
  finds far more distinct strategies with the same validity and no constraint violations, and its closest
  counterfactual is closer. Three things are missing before this can be claimed: a set-size-matched comparison (C1),
  plausibility (C5) and MOC (C6). MOC returns a Pareto front, so it is the baseline most likely to compete on coverage.
- **RQ2 (QD vs. QD):** not addressable until NS, NSLC and surprise are implemented (C6).

---

## 4. Concerns

Ordered by how much they threaten the main claim. Each concern lists the evidence and a proposed action.

### High: they affect the RQ1 conclusion

**C1. The coverage comparison is structurally biased toward MAP-Elites.**
- *Evidence:* coverage counts filled cells, so it is capped by the number of counterfactuals returned. Wachter and FACE
  return 1, DiCE at most 10, and MAP-Elites 85 on average (up to 358). On German Credit and HELOC, DiCE's ceiling is
  2.5% (§2.2). Per returned counterfactual, DiCE is about as diverse in strategies, or more: German 0.67 strategies
  per CF for DiCE-genetic vs. 0.38 for MAP-Elites; HELOC 0.44 vs. 0.30; GMC 0.41 vs. 0.30; Adult 0.28 vs. 0.25.
  MAP-Elites is also the only method that sees the grid during the search.
- *Action:* add a **matched-k** comparison, e.g. MAP-Elites restricted to k = 10 elites (best fitness, or a
  max-coverage subset) vs. DiCE with 10. Run DiCE with `total_cfs` equal to the size of the MAP-Elites archive.
  Report strategies per counterfactual next to coverage. The per-counterfactual rows needed for this are in
  `results/full/runs/` (group Drive), not in git.

**C2. MAP-Elites optimizes the evaluation metric.**
- *Evidence:* the elite fitness is `1 / (1 + Guidotti distance)`, the same distance used to evaluate proximity.
  Wachter minimizes L1 in min-max space and DiCE minimizes its own proximity + diversity loss.
- *Action:* also report proximity in a neutral metric (e.g. L1 in min-max space, or each method's own distance), and
  state this asymmetry in the paper.

**C3. "Same budget" does not mean the same search effort.**
- *Evidence:* median queries actually used at budget 10,000 are 10,000 for MAP-Elites, about 830 for DiCE-genetic,
  2,031 for DiCE-random, 9 for Wachter and 1 for FACE. On top of that, `train_predictions: free` gives DiCE (KD-tree)
  and FACE (graph) uncounted predictions on the whole training set (31k rows on Adult, 84k on GMC).
- *Action:* decide the open `train_predictions` question as a group. Report queries used next to every curve.
  Consider letting DiCE-genetic spend the budget (more generations or restarts) so the curves compare methods that
  all try.

**C4. The results mix code versions.**
- *Evidence:* the manifest has 7 different source hashes. Only 262 configurations ran on the committed code
  (`src:2144fd74c7`, almost all of COMPAS and GMC). The other 438 ran on older sources: all of Adult, German Credit
  and HELOC, plus 18 COMPAS/GMC configurations. The old sources were never committed, so we cannot show what changed.
  In addition, 25 manifest entries were rebuilt by `repair_manifest.py` (no timing, `reconstructed: true`).
- *Action:* re-run the 438 configurations on the committed code. That is about 23 CPU-hours plus the 25 untimed
  configurations, roughly 4 h on 6 cores. A cheaper first step is to re-run a sample and diff the summaries. If they
  match, document it and keep the results.

**C5. Plausibility is not measured.**
- *Evidence:* LOF, nearest-neighbor distance and yNN are not implemented (`metrics.py`), but RQ1 asks for "without
  losing plausibility". MAP-Elites builds counterfactuals by mutation. It also produces fractional values for integer
  features: notebook 03 shows `duration: 48 → 34.05` months, and Wachter gives `48 → 38.29`.
- *Action:* implement the plausibility metrics and the `integer` flag on `Feature` (round in `clip`) **before** the
  final run.

**C6. Half of the planned methods are missing.**
- *Evidence:* NS, NSLC, surprise and MOC are not implemented, and the MOC owner is still open. Without them RQ2 cannot
  be studied, and RQ1 lacks its strongest baseline.
- *Action:* assign MOC. Implement NS, NSLC and surprise through `register()`. The batch runner already supports them.

### Medium: they affect interpretation and presentation

**C7. GMC distances blow up.**
- *Evidence:* GMC's heavy-tailed features (utilization, debt ratio, income) have tiny MADs. Wachter's median closest
  distance on GMC is 53.7 (mean 152 with LR), and DiCE-random's mean pairwise diversity is 273–302. Every mean that
  includes GMC is dominated by these values.
- *Action:* report medians, or log-transform or winsorize the heavy-tailed GMC features in `configs/gmc.yaml`.

**C8. Some of the explained models are weak.**
- *Evidence:* the GMC models barely beat the majority class (0.935 vs. 0.931, balanced accuracy 0.57–0.58) and predict
  only about 1.8% of the test set as unfavorable. The German Credit models sit at 0.71–0.73 vs. a 0.70 majority, and
  the NN has balanced accuracy 0.60 (Guidotti reports 0.81). Counterfactuals for near-majority classifiers say more
  about odd decision boundaries than about the methods.
- *Action:* tune with class weights or a validation split (never the test set). At minimum, report balanced accuracy
  in the paper.

**C9. The DiCE-random results are partly an adapter artifact, and it outperforms DiCE-genetic.**
- *Evidence:* DiCE-random has success 0 at budgets ≤ 2,000 (§2.3). DiCE-random beats DiCE-genetic on average
  coverage rank (2.27 vs. 2.73), median closest distance (0.26 vs. 0.41), sparsity (L0 1.7 vs. 4.4) and redundancy. DiCE-genetic, the main baseline,
  looks under-configured.
- *Action:* add an anytime fallback for DiCE-random, or charge-aware sampling. Review the DiCE-genetic
  hyperparameters before reviewers do.

**C10. DiCE-genetic often returns its fallback, not its final answer.**
- *Evidence:* at budget 10,000 the anytime fallback fires for 30% of instances (16–39% by dataset). At budget 500 it
  fires for nearly all Adult and German Credit instances. Those outputs are the valid members of the last population,
  without DiCE's final post-processing.
- *Action:* report fallback rates and give the fallback cases a separate breakdown.

**C11. The grid design is still provisional.**
- *Evidence:*
  - COMPAS has 1 group, so 2 strategies and 10 cells. It cannot show strategy diversity.
  - Only 175 of Adult's 400 cells are feasible.
  - The effort bins were never calibrated before the full run (open item in the README).
  - A change to a nominal feature counts as "increased".
- *Action:* calibrate the effort bins on one dataset. Decide the nominal-feature state. Present COMPAS as an appendix
  case, as planned.

**C12. The MAP-Elites output is not yet an explanation.**
- *Evidence:* 85 counterfactuals per instance on average (up to 358), with 64% of their changes revertible. No user
  will read that.
- *Action:* add a selection or summary step (e.g. one elite per strategy, post-hoc sparsification), and consider a
  sparsity term in the fitness. This connects to the open fitness question (plan 5.3).

**C13. The statistical design needs care.**
- *Evidence:* seeds as blocks inflate p-values by 80 orders of magnitude. Correct blocks give 10–15 blocks, where
  Nemenyi cannot separate MAP-Elites from DiCE despite 10/10 wins (§2.6).
- *Action:* use (dataset, model) blocks. Add pairwise Wilcoxon or sign tests with Holm correction vs. the control
  method. Adding datasets increases power more than adding seeds.

### Low: worth noting

- **C14. The CARLA check is small.** 10 instances and one seed, and GMC + NN deviates for three methods (§2.8). Worth
  repeating with more instances before citing.
- **C15. The simulation is a quick run.** One seed and 5 instances. Recall is not monotonic in budget. The full design
  is planned for the paper extension.
- **C16. Instances depend on the model.** Each model explains the rows *it* predicts unfavorable, so comparisons across
  models are unpaired. Comparisons across methods are paired.
- **C17. Per-counterfactual rows are not in git.** `results/full/runs/` (~200 MB) lives on the Drive, so any re-analysis
  at the counterfactual level (C1, C12) needs that copy.

---

## 5. Suggested next steps

1. Re-run, or sample-check, the 438 configurations that used old code (C4).
2. Add the matched-k coverage analysis and strategies per counterfactual (C1). This needs `runs/` from the Drive.
3. Implement plausibility and the integer flag (C5), and calibrate the effort bins (C11).
4. Fix the DiCE-random fallback and review the DiCE-genetic settings (C9, C10).
5. Decide `train_predictions` and the GMC scaling as a group (C3, C7).
6. Implement MOC, NS, NSLC and surprise (C6), then run the final grid once, on one committed code version.
