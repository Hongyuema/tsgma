"""Reconcile two historical food tables and the archived dietary export by code."""

from pathlib import Path
from zipfile import ZipFile
import argparse
import numpy as np
import pandas as pd
from recover_outcomes import records


def run(root):
    src = root / "historical_sources"
    out = root / "outcome_audit"
    out.mkdir(exist_ok=True)
    reference = pd.read_csv(root / "results/analysis_snapshot.csv").set_index("Code")
    early = records(src / "early_food_155.xlsx").set_index("Code")
    # Leftmost code identifies the food columns in the historical wide sheet.
    old = records(src / "assembled_food_158.xlsx")
    old = old.iloc[:, :22].set_index("Code")
    shared = early.index.intersection(reference.index)
    assert early.index.is_unique and old.index.is_unique and reference.index.is_unique
    assert set(old.index) == set(reference.index)
    foods = list(reference.columns[:16])
    for frame, idx in [(old, reference.index), (early, shared)]:
        cols = foods + ["Calories"]
        np.testing.assert_allclose(
            frame.loc[idx, cols], reference.loc[idx, cols], atol=1e-9, rtol=0
        )
    with ZipFile(src / "historical_diet.zip") as z:
        diet = pd.read_csv(z.open(next(n for n in z.namelist() if n.endswith(".csv"))))
    codes = diet.loc[
        diet.Code.notna() & diet.Code.ne("OWID_WRL"), ["Code", "Entity"]
    ].drop_duplicates()
    assert codes.Code.is_unique
    ledger = codes.set_index("Code").sort_index()
    assert set(early.index.union(reference.index)) <= set(ledger.index)
    ledger["early_155"] = ledger.index.isin(early.index)
    ledger["assembled_158"] = ledger.index.isin(old.index)
    ledger["supplied_158"] = ledger.index.isin(reference.index)
    ledger["in_either_food_table"] = ledger.early_155 | ledger.supplied_158
    for short, end in [("nonhdl", 2017), ("hypertension", 2017), ("diabetes", 2014)]:
        raw = records(src / (short + "_long.xlsx"))
        years = (
            raw.loc[raw.Year.between(1992, end)]
            .dropna(subset=[raw.columns[-1]])
            .groupby("Code")
            .Year.nunique()
        )
        ledger[short + "_observed_years"] = years.reindex(ledger.index, fill_value=0)
    assert (len(ledger), len(shared), ledger.in_either_food_table.sum()) == (
        193,
        154,
        159,
    )
    assert set(early.index) - set(reference.index) == {"TWN"}
    assert set(reference.index) - set(early.index) == {"AUS", "NOR", "NZL", "USA"}
    ledger.to_csv(out / "historical_roster.csv")
    # The archived dietary release is a source comparator. Diagnose direct
    # group means without treating it as the exact origin of the supplied foods.
    milk = next(c for c in diet.columns if "00002948" in c)
    group = diet.loc[diet.Year.between(1992, 2017)].groupby("Code")[milk]
    check = reference[["Milk", "Calories"]].copy()
    check["archived_milk_years"] = group.count().reindex(check.index)
    check["archived_milk_mean"] = group.mean().reindex(check.index)
    check["milk_difference"] = check.archived_milk_mean - check.Milk
    check.to_csv(out / "historical_food_source_check.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    run(p.parse_args().root)
