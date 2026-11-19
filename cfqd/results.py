"""Common output format: one CSV row per counterfactual.

Fixed columns + one column per feature. Every method writes this format, and
the metrics module (T12) and the figures (T14) read from it.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .methods.base import CFResult
from .schema import DatasetSpec

COLUMNS = ["dataset", "model", "method", "seed", "instance_id", "budget",
           "cf_id", "found_at", "n_queries", "seconds"]


def to_frame(res: CFResult, spec: DatasetSpec, *, dataset: str, model: str,
             seed: int, instance_id: int, budget: int) -> pd.DataFrame:
    rows = []
    for k, cf in enumerate(res.cfs):
        row = dict(dataset=dataset, model=model, method=res.method, seed=seed,
                   instance_id=instance_id, budget=budget, cf_id=k,
                   found_at=res.found_at[k] if k < len(res.found_at) else None,
                   n_queries=res.n_queries, seconds=res.seconds)
        row.update(dict(zip(spec.names, cf)))
        rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS + spec.names)


def append_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
