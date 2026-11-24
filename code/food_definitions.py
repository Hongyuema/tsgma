"""Audit the historical food worksheet and vary definitions on frozen annual records."""

import argparse
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control
from audit_analysis import OUTCOMES, association
from reconstruct_annual import SELECTED, RED, panel_model


def audit(root, sheet, dest):
    # A repeated header inside a historical spreadsheet is not an observation.
    x = pd.read_excel(sheet)
    repeated = x.Year.eq("Year")
    assert x.loc[repeated, "Code"].eq("Code").all()
    x = x.loc[~repeated & x.Code.notna()].copy()
    x.Year = pd.to_numeric(x.Year, errors="raise").astype(int)
    with ZipFile(root / "historical_sources/historical_diet.zip") as z:
        raw = pd.read_csv(z.open(next(k for k in z.namelist() if k.endswith(".csv"))))
    raw = raw[raw.Code.notna()].copy()

    def keyed(frame):
        return (
            frame.set_index(["Code", "Year"])
            .sort_index()
            .rename(columns=lambda c: int(c.split("|")[1]) if "|" in c else c)
        )

    x = keyed(x)
    raw = keyed(raw)
    assert x.index.is_unique and x.index.equals(raw.index)
    items = raw.columns.drop("Entity")
    np.testing.assert_allclose(
        x[items].to_numpy(float), raw[items], rtol=0, atol=1e-10, equal_nan=True
    )
    verified = x[[2731, 2732, 2733, "RMC"]].astype(float).copy()
    verified["observed_items"] = verified.iloc[:, :3].count(axis=1)
    verified["available_sum"] = verified.iloc[:, :3].sum(axis=1, min_count=1)
    np.testing.assert_allclose(verified.RMC, verified.available_sum, atol=1e-10, rtol=0)
    submitted = pd.read_csv(root / "corrected_results/analysis_snapshot.csv").set_index(
        "Code"
    )
    roster = verified.index.get_level_values("Code").isin(submitted.index)
    years = verified.index.get_level_values("Year")
    study = verified.loc[roster & years.isin(range(1992, 2018))]
    assert len(study) == len(submitted) * 26
    all_means = study.groupby(level="Code").RMC.mean().to_frame("worksheet_mean")
    all_means["submitted_mean"] = submitted["Red meat"]
    all_means["absolute_difference"] = (
        all_means.worksheet_mean - all_means.submitted_mean
    ).abs()
    all_means["years_with_two_items"] = (
        study.observed_items.eq(2).groupby(level="Code").sum()
    )
    all_means.to_csv(dest / "red_meat_country_means.csv")
    study.to_csv(dest / "red_meat_source_rows.csv")
    original = (
        pd.read_csv(root / "historical_plot_audit/plot_year_values.csv")
        .query("exposure=='Red meat'")
        .dropna(subset=["plot_food"])
        .set_index(["Code", "Year"])
    )
    plot = original[["plot_food"]].join(verified.RMC, how="left", validate="one_to_one")
    assert plot.RMC.notna().all()
    plot["absolute_difference"] = (plot.plot_food - plot.RMC).abs()
    plot.to_csv(dest / "red_meat_plot_comparison.csv")
    return submitted


def run(root, sheet, dest):
    dest.mkdir(parents=True, exist_ok=True)
    submitted = audit(root, sheet, dest)
    a = root / "annual_results"
    panel = pd.read_csv(a / "annual_panel_1992_2017.csv")
    raw = pd.read_csv(
        root / "annual_sources/dietary.csv", usecols=["Code", "Year", *SELECTED, *RED]
    )
    panel = panel.merge(
        raw[raw.Code.isin(submitted.index)],
        on=["Code", "Year"],
        how="left",
        validate="one_to_one",
    )
    panel["four_items"] = panel[[k for k in SELECTED if k != "Maize"]].sum(
        axis=1, min_count=4
    )
    panel["five_items"] = panel[SELECTED].sum(axis=1, min_count=5)
    panel["official"] = panel.Cereals
    panel["all_three"] = panel[RED].sum(axis=1, min_count=3)
    panel["available_items"] = panel[RED].sum(axis=1, min_count=1)
    np.testing.assert_allclose(panel.five_items, panel.Selected_cereals, equal_nan=True)
    np.testing.assert_allclose(panel.all_three, panel["Red meat"], equal_nan=True)
    panel = panel.sort_values(["Code", "Year"]).set_index(["Code", "Year"])
    full = pd.read_csv(a / "full_1992_2017_means.csv").set_index("Code")
    window = panel[panel.index.get_level_values("Year").isin(range(1992, 2015))].copy()
    full["red_available"] = (
        panel.available_items.groupby(level="Code")
        .mean()
        .where(panel.available_items.groupby(level="Code").count().eq(26))
    )
    rows = []
    samples = []
    for target, keys in [
        ("cereal", ["four_items", "five_items", "official"]),
        ("red_meat", ["all_three", "available_items"]),
    ]:
        for food in keys:
            for y in OUTCOMES:
                required = (
                    ["Cereals", "Selected_cereals"]
                    if target == "cereal"
                    else ["Red meat" if food == "all_three" else "red_available"]
                )
                codes = full.dropna(
                    subset=[*required, "GDP", "Total_energy", y]
                ).index.intersection(submitted.dropna(subset=["GDP"]).index)
                df = window[
                    window.index.get_level_values("Code").isin(codes)
                ].reset_index()
                assert (
                    len(df) == 23 * len(codes)
                    and df[[food, "GDP", "Total_energy", y]].notna().all().all()
                )
                means = df.groupby("Code")[[food, "GDP", "Total_energy", y]].mean()
                means["logGDP"] = np.log(means.GDP)
                cross = association(means, food, y, ["logGDP", "Total_energy"])
                slope = panel_model(df, food, y, True)
                for spec, est, beta in [
                    ("between", cross, "r"),
                    ("within", slope, "beta"),
                ]:
                    rows.append(
                        dict(
                            target=target,
                            definition=food,
                            outcome=y,
                            model=spec,
                            n=len(codes),
                            years=23,
                            estimate=est[beta],
                            ci_low=est["ci_low"],
                            ci_high=est["ci_high"],
                            p=est["p"],
                        )
                    )
                samples.extend((target, food, y, c) for c in codes)
    results = pd.DataFrame(rows)
    # Three cereal definitions times three outcomes per model. Red meat uses the
    # same six-test family across both coverage rules and all three outcomes.
    results["q"] = results.groupby(["target", "model"]).p.transform(
        false_discovery_control
    )
    results.to_csv(dest / "estimates.csv", index=False)
    pd.DataFrame(samples, columns=["target", "definition", "outcome", "Code"]).to_csv(
        dest / "membership.csv", index=False
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--sheet", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    run(
        args.root,
        args.sheet or args.root / "historical_sources/diet_components.xlsx",
        args.output or args.root / "food_definitions",
    )
