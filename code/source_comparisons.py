"""Rebuild source comparisons and quantify heterogeneous schooling-year coverage."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, t
from audit_analysis import FOODS, OUTCOMES, association


def run(root):
    source = root / "inputs/supplied_annual_covariates"
    snapshot = pd.read_csv(root / "results/analysis_snapshot.csv").set_index("Code")
    corrected = pd.read_csv(root / "corrected_results/analysis_snapshot.csv").set_index(
        "Code"
    )
    out = root / "source_checks"
    out.mkdir(exist_ok=True)
    years = pd.read_csv(root / "annual_results/annual_panel_1992_2017.csv")
    for filename, original, rebuilt, end in [
        ("average-non-hdl-cholesterol-levels", "Non-HDL", "Non-HDL", 2017),
        ("diabetes-prevalence-who-gho", "Diabetes1992-2014", "Diabetes1992-2014", 2014),
        ("hypertension-adults-30-79", "Hypertension", "Hypertension", 2017),
        ("daily-per-capita-caloric-supply", "Calories", "Total_energy", 2017),
    ]:
        table = pd.DataFrame({"supplied": snapshot[original]})
        table["current"] = (
            years[years.Year.between(1992, end)].groupby("Code")[rebuilt].mean()
        )
        table["difference"] = table.current - table.supplied
        table.to_csv(out / (filename + "_comparison.csv"))
    entries = []
    for folder, variable in [
        ("median-age", "Median age"),
        ("share-population-female", "Gender ratio"),
        ("mean-years-of-schooling-long-run", "Mean years schooling"),
    ]:
        raw = pd.read_csv(source / folder / (folder + ".csv"))
        obs = (
            raw[raw.Code.isin(snapshot.index) & raw.Year.between(1992, 2017)]
            .dropna(subset=[raw.columns[3]])
            .sort_values(["Code", "Year"])
        )
        assert not obs.duplicated(["Code", "Year"]).any()
        values = obs.groupby("Code")[raw.columns[3]]
        avg = values.mean().reindex(snapshot.index)
        mean = (100 - avg) / avg if variable == "Gender ratio" else avg
        by_code = pd.DataFrame(
            {
                "variable": variable,
                "supplied": snapshot[variable],
                "recomputed": mean,
                "n_years": values.count().reindex(snapshot.index, fill_value=0),
            }
        )
        by_code["observed_years"] = (
            obs.groupby("Code")
            .Year.agg(lambda x: " ".join(map(str, x)))
            .reindex(snapshot.index, fill_value="")
        )
        np.testing.assert_allclose(
            mean, snapshot[variable], atol=1e-10, rtol=0, equal_nan=True
        )
        entries.append(by_code.reset_index())
        if variable == "Mean years schooling":
            corrected["school_2015"] = obs[obs.Year.eq(2015)].set_index("Code")[
                raw.columns[3]
            ]
    pd.concat(entries, ignore_index=True).to_csv(
        out / "annual_covariate_reconciliation.csv", index=False
    )
    assert corrected.school_2015.notna().equals(
        corrected["Mean years schooling"].notna()
    )
    matched = corrected.dropna(subset=["school_2015"])
    assert len(matched) == 126
    rows = []
    for variable in ["Mean years schooling", "school_2015"]:
        for x in FOODS:
            for y in OUTCOMES:
                row = association(matched, x, y, [variable])
                # Independent OLS t test of the food coefficient on this fixed country roster.
                design = np.column_stack(
                    (np.ones(len(matched)), matched[x], matched[variable])
                )
                outcome = matched[y].to_numpy()
                coef = np.linalg.lstsq(design, outcome, rcond=None)[0]
                df = len(matched) - 3
                se = np.sqrt(
                    np.sum((outcome - design @ coef) ** 2)
                    / df
                    * np.linalg.inv(design.T @ design)[1, 1]
                )
                stat = coef[1] / se
                np.testing.assert_allclose(
                    [stat / np.sqrt(df + stat**2), 2 * t.sf(abs(stat), df)],
                    [row["r"], row["p"]],
                    atol=1e-10,
                )
                rows.append(dict(schooling=variable, exposure=x, outcome=y, **row))
    checks = pd.DataFrame(rows)
    checks["q_BH_48"] = checks.groupby("schooling").p.transform(false_discovery_control)
    archived = (
        pd.read_csv(root / "corrected_results/all_associations.csv")
        .query("specification=='Schooling_only_126'")
        .set_index(["exposure", "outcome"])
    )
    baseline = (
        checks[checks.schooling.eq("Mean years schooling")]
        .set_index(["exposure", "outcome"])
        .reindex(archived.index)
    )
    np.testing.assert_allclose(
        baseline[["r", "p", "q_BH_48"]], archived[["r", "p", "q_BH"]], atol=1e-12
    )
    checks.to_csv(out / "schooling_year_comparison.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    run(parser.parse_args().root)
