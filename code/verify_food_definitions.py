"""Check historical cells and refit all definition comparisons using alternative designs."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy.stats import t, norm
from statsmodels.stats.multitest import multipletests


def check(root, sheet, dest):
    tested = pd.read_csv(dest / "red_meat_source_rows.csv").set_index(["Code", "Year"])
    cols = [str(i) for i in (2731, 2732, 2733)]
    verified = set()
    book = load_workbook(sheet, read_only=True, data_only=True)
    it = iter(book.active.values)
    header = next(it)
    indices = [
        next(
            i
            for i, h in enumerate(header)
            if "|" in h and int(h.split("|")[1]) == int(col)
        )
        for col in cols
    ]
    for row in it:
        key = (row[1], row[2])
        if key not in tested.index:
            continue
        assert key not in verified
        verified.add(key)
        items = [row[i] for i in indices if row[i] is not None]
        expected = tested.loc[key]
        assert (
            len(items) == expected.observed_items
            and abs(sum(items) - row[3]) < 1e-10
            and abs(expected.RMC - row[3]) < 1e-10
        )
    book.close()
    assert len(verified) == 4108 and (tested.observed_items == 2).sum() == 100
    roster = pd.read_csv(dest / "membership.csv")
    panel = pd.read_csv(root / "annual_results/annual_panel_1992_2017.csv").query(
        "Year<=2014"
    )
    raw = pd.read_csv(root / "annual_sources/dietary.csv")
    raw = raw[raw.Code.isin(panel.Code)]
    panel = panel.merge(
        raw[
            [
                "Year",
                "Code",
                "Wheat",
                "Rice",
                "Barley",
                "Other cereals",
                "Maize",
                "Pork",
                "Sheep and goat meat",
                "Beef and buffalo meat",
            ]
        ],
        on=["Year", "Code"],
        validate="one_to_one",
    )
    panel["four_items"] = (
        panel.Wheat + panel.Rice + panel.Barley + panel["Other cereals"]
    )
    panel["five_items"] = panel.four_items + panel.Maize
    panel["official"] = panel.Cereals
    meat = panel[["Pork", "Sheep and goat meat", "Beef and buffalo meat"]]
    panel["all_three"] = (
        meat.Pork + meat["Sheep and goat meat"] + meat["Beef and buffalo meat"]
    )
    panel["available_items"] = (
        meat.fillna(0).sum(axis=1).where(meat.notna().any(axis=1))
    )
    results = pd.read_csv(dest / "estimates.csv")
    assert len(results) == 30
    threshold = norm.ppf(0.975)
    for row in results.itertuples():
        codes = roster.loc[
            roster.definition.eq(row.definition) & roster.outcome.eq(row.outcome),
            "Code",
        ]
        sample = panel[panel.Code.isin(codes)].sort_values(["Code", "Year"])
        g = len(codes)
        assert g == row.n and len(sample) == 23 * g
        if row.model == "between":
            avg = sample.groupby("Code")[
                [row.definition, "GDP", "Total_energy", row.outcome]
            ].mean()
            design = np.column_stack(
                (np.ones(g), avg[row.definition], np.log(avg.GDP), avg.Total_energy)
            )
            y = avg[row.outcome].to_numpy()
            beta = np.linalg.lstsq(design, y, rcond=None)[0]
            s2 = np.sum((y - design @ beta) ** 2) / (g - 4)
            stat = beta[1] / np.sqrt(np.linalg.inv(design.T @ design)[1, 1] * s2)
            effect = stat / np.sqrt(stat**2 + g - 4)
            p = 2 * t.sf(abs(stat), g - 4)
            lo, hi = np.tanh(
                np.arctanh(effect) + np.array([-1, 1]) * threshold / np.sqrt(g - 5)
            )
        else:
            terms = np.column_stack(
                (
                    sample[row.definition] / 100,
                    np.log(sample.GDP),
                    sample.Total_energy / 100,
                    sample[row.outcome],
                )
            ).reshape(g, 23, 4)
            flat = (terms - terms.mean(axis=1, keepdims=True)).reshape(23 * g, 4)
            yr = np.eye(23)[:, 1:]
            design = np.column_stack(
                (flat[:, :3], np.tile(yr - yr.mean(axis=0), (g, 1)))
            )
            y = flat[:, -1]
            beta = np.linalg.lstsq(design, y, rcond=None)[0]
            infl = (y - design @ beta) * (
                design @ np.linalg.inv(design.T @ design)[:, 0]
            )
            n = len(y)
            df_model = g + design.shape[1]
            se = np.sqrt(
                np.sum(infl.reshape(g, 23).sum(axis=1) ** 2)
                * g
                / (g - 1)
                * (n - 1)
                / (n - df_model)
            )
            effect = beta[0]
            lo, hi = effect + np.array([-1, 1]) * t.ppf(0.975, g - 1) * se
            p = 2 * t.sf(abs(effect / se), g - 1)
        np.testing.assert_allclose(
            [effect, lo, hi, p],
            [row.estimate, row.ci_low, row.ci_high, row.p],
            rtol=1e-8,
            atol=2e-10,
        )
    for _, rows in results.groupby(["target", "model"]):
        np.testing.assert_allclose(
            multipletests(rows.p, method="fdr_bh")[1], rows.q, atol=2e-12
        )
    print(
        "Verified 4,108 source observations, 30 estimates, intervals, P values and four BH families."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--sheet", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    check(
        args.root,
        args.sheet or args.root / "historical_sources/diet_components.xlsx",
        args.output or args.root / "food_definitions",
    )
