"""Audit historic plotted values and annual coverage for the 18 named examples."""

import argparse
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pandas as pd
from recover_outcomes import records

CODES = (
    "CHN USA AUS FRA IND GBR DEU BRA CAN ZAF DZA ARG MEX SWE IDN SAU TUR EGY".split()
)
PAIRS = {
    "Non-HDL": ["Animal fats group", "Red meat", "Eggs"],
    "Diabetes": ["Poultry meat", "Cereals", "Sugar and sweeteners", "Starchy roots"],
    "Hypertension": ["Alcoholic beverages"],
}
HEALTH = {"Non-HDL": "nonhdl", "Diabetes": "diabetes", "Hypertension": "hypertension"}
FOOD_CODES = {
    "Animal fats group": [2946],
    "Red meat": [2731, 2732, 2733],
    "Eggs": [2949],
    "Poultry meat": [2734],
    "Cereals": [2511, 2807, 2514, 2513, 2520],
    "Sugar and sweeteners": [2909],
    "Starchy roots": [2907],
    "Alcoholic beverages": [2924],
}


def series_by_country(path):
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        data = pd.read_csv(csv_path, float_precision="round_trip")
        if data.duplicated(["Code", "variable", "Year"]).any():
            raise ValueError(f"Duplicate plotted observations: {csv_path}")
        return {
            (code, name): values.set_index("Year")["value"]
            .rename(None)
            .rename_axis(None)
            for (code, name), values in data.groupby(["Code", "variable"], sort=False)
        }
    found = {}
    for table in ET.parse(path).findall("Table"):
        code = "".join(table.find("Title").itertext()).strip().split("-")[0]
        ycol = table.find("YColumn")
        name = "".join(ycol.find("Title").itertext()).strip()
        x = table.findall("XColumn/Subcolumn/d")
        y = ycol.findall("Subcolumn/d")
        assert len(y) <= len(x)
        assert all(not cell.attrib for cell in y)
        years = [int(cell.text) for cell in x]
        assert len(set(years)) == len(years)
        series = pd.Series(
            {yr: float(v.text) for yr, v in zip(years, y) if v.text is not None}
        )
        if (code, name) in found:
            pd.testing.assert_series_equal(found[code, name], series)
        else:
            found[code, name] = series
    return found


def run(root):
    snapshot = pd.read_csv(root / "results/analysis_snapshot.csv").set_index("Code")
    src = root / "historical_plot_inputs"
    out = root / "historical_plot_audit"
    out.mkdir(exist_ok=True)
    with ZipFile(root / "historical_sources/historical_diet.zip") as z:
        diet = pd.read_csv(
            z.open(next(name for name in z.namelist() if name.endswith(".csv")))
        )
    diet = (
        diet[diet.Code.notna()]
        .set_index(["Code", "Year"])
        .drop(columns="Entity")
        .sort_index()
    )
    diet.columns = [int(col.split("|")[1]) for col in diet]
    assert diet.index.is_unique
    panel = pd.read_csv(root / "annual_results/annual_panel_1992_2017.csv").set_index(
        ["Code", "Year"]
    )
    counts, annual, cereals, health_matches = [], [], [], []
    for outcome, foods in PAIRS.items():
        raw = records(root / "historical_sources" / (HEALTH[outcome] + "_long.xlsx"))
        raw = raw.set_index(["Code", "Year"]).iloc[:, -1]
        all_health = raw[raw.index.get_level_values("Code").notna()].unstack("Year")
        for food in foods:
            series = series_by_country(src / (outcome + "_" + food + ".pzfx"))
            original = next(col for col in snapshot if col.lower() == food.lower())
            assert all((c, term) in series for c in CODES for term in (outcome, food))
            for c in CODES:
                x, y = series[c, food], series[c, outcome]
                values = pd.concat(
                    [x.rename("plot_food"), y.rename("plot_health")], axis=1
                ).sort_index()
                expected = raw.loc[c].reindex(values.index)
                items = diet.loc[c].reindex(values.index)
                ref = items[FOOD_CODES[food]].sum(axis=1, min_count=1)
                values["source_health"] = expected
                values["archived_diet_comparator"] = ref
                pre = values.loc[:2009].dropna(
                    subset=["plot_food", "archived_diet_comparator"]
                )
                window = x.reindex(range(1992, 2018)).dropna()
                counts.append(
                    dict(
                        Code=c,
                        exposure=food,
                        outcome=outcome,
                        food_n=len(x),
                        food_first=x.index.min(),
                        food_last=x.index.max(),
                        health_n=len(y),
                        health_first=y.index.min(),
                        health_last=y.index.max(),
                        common_n=len(x.index.intersection(y.index)),
                        food_window_n=len(window),
                        plot_food_mean=window.mean(),
                        supplied_food_mean=snapshot.loc[c, original],
                        difference=window.mean() - snapshot.loc[c, original],
                        health_no_source=int(expected.reindex(y.index).isna().sum()),
                        health_mismatches=int(
                            ((y - expected.reindex(y.index)).abs() > 1e-8).sum()
                        ),
                        pre2009_comparable=len(pre),
                        matches_through_2009=int(
                            (
                                (pre.plot_food - pre.archived_diet_comparator).abs()
                                < 1e-5
                            ).sum()
                        ),
                    )
                )
                values.index.name = "Year"
                annual.append(
                    values.reset_index().assign(Code=c, exposure=food, outcome=outcome)
                )
                if food == "Cereals":
                    comp = items[[2511, 2807, 2513, 2520]].sum(axis=1, min_count=1)
                    cereals.append(
                        pd.DataFrame(
                            {
                                "plot": x,
                                "four_item_available_sum": comp,
                                "five_item_available_sum": ref,
                                "n_four_observed": items[
                                    [2511, 2807, 2513, 2520]
                                ].count(axis=1),
                            }
                        )
                        .loc[1961:2009]
                        .rename_axis("Year")
                        .reset_index()
                        .assign(Code=c)
                    )
            if outcome == "Hypertension":
                for c in CODES:
                    yrs = list(range(1990, 2020))
                    values = series[c, outcome].reindex(yrs)
                    differences = all_health[yrs].sub(values, axis=1)
                    matched = differences.abs().le(1e-8).all(axis=1)
                    for matched_code in all_health.index[matched]:
                        health_matches.append(
                            dict(
                                plot_code=c,
                                matching_source_code=matched_code,
                                first_year=1990,
                                last_year=2019,
                                matched_years=30,
                            )
                        )
                assert {
                    r["plot_code"]: r["matching_source_code"]
                    for r in health_matches
                    if r["plot_code"] in ["SAU", "TUR"]
                } == {"SAU": "KAZ", "TUR": "RUS"}
            if food in ("Cereals", "Starchy roots"):
                pd.testing.assert_series_equal(series["IDN", food], series["IND", food])
                assert not series["IDN", outcome].equals(series["IND", outcome])
                # The independently reconstructed annual model uses distinct keyed exposure vectors.
                term = "Cereals" if food == "Cereals" else "Starchy Roots"
                assert not np.allclose(
                    panel.loc["IDN", term].loc[1992:2014],
                    panel.loc["IND", term].loc[1992:2014],
                )
    pd.DataFrame(health_matches).to_csv(
        out / "hypertension_source_matches.csv", index=False
    )
    results = pd.DataFrame(counts)
    results.to_csv(out / "named_example_coverage.csv", index=False)
    pd.concat(annual, ignore_index=True).to_csv(
        out / "plot_year_values.csv", index=False
    )
    pd.concat(cereals, ignore_index=True).to_csv(
        out / "cereal_component_comparison.csv", index=False
    )
    wide = records(src / "alcohol_annual.xlsx")
    keyed = wide[wide.iloc[:, 0].isin(snapshot.index)].set_index(wide.columns[0])
    assert keyed.index.is_unique and set(keyed.index) == set(snapshot.index)
    window = keyed.loc[snapshot.index, 1992:2017].apply(pd.to_numeric, errors="raise")
    assert window.notna().all().all() and window.shape == (158, 26)
    audit = pd.DataFrame(
        {
            "n_years": window.count(axis=1),
            "historical_mean": window.mean(axis=1),
            "submitted_mean": snapshot["Alcoholic Beverages"],
        }
    )
    audit["difference"] = audit.historical_mean - audit.submitted_mean
    np.testing.assert_allclose(
        audit.historical_mean, audit.submitted_mean, rtol=0, atol=1e-9
    )
    audit.rename_axis("Code").to_csv(out / "alcohol_country_means.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    run(parser.parse_args().root)
