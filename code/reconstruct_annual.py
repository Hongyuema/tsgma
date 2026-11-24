"""Reconstruct a frozen OWID annual sensitivity dataset and fit stated models.

python reconstruct_annual.py --sources sources --snapshot results/analysis_snapshot.csv --output annual_results
All decisions are post-review exploratory. Blank source cells remain missing.
"""

from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from audit_analysis import FOODS, OUTCOMES, association, add_fdr


DIRECT = {
    "Alcoholic Beverages": "Alcoholic beverages",
    "Animal fats group": "Animal fats",
    "Milk": "Milk",
    "Eggs": "Eggs",
    "Sugar and Sweeteners": "Sugar and sweeteners",
    "Vegetable Oils": "Vegetable oils",
    "Nuts": "Nuts",
    "Poultry meat": "Poultry meat",
    "Vegetables": "Vegetables",
    "Pulses": "Pulses",
    "Fish and seafood": "Fish and seafood",
    "Starchy Roots": "Starchy roots",
    "Fruit": "Fruit",
    "Other meat": "Other meat",
}
HEALTH = {
    "Non-HDL": "average-non-hdl-cholesterol-levels",
    "Diabetes1992-2014": "diabetes-prevalence-who-gho",
    "Hypertension": "hypertension-adults-30-79",
}
RED = ["Beef and buffalo meat", "Pork", "Sheep and goat meat"]
SELECTED = ["Wheat", "Rice", "Maize", "Barley", "Other cereals"]


def indicator(source, name, label):
    data = json.loads((source / f"{name}.data.json").read_text())
    metadata = json.loads((source / f"{name}.metadata.json").read_text())
    entities = metadata["dimensions"]["entities"]["values"]
    lookup = {item["id"]: item.get("code") for item in entities}
    frame = pd.DataFrame(
        {
            "Code": [lookup[x] for x in data["entities"]],
            "Year": data["years"],
            label: data["values"],
        }
    )
    return frame.dropna(subset=["Code"])


def country_means(panel, columns, start, end):
    frame = panel[panel.Year.between(start, end)]
    means = frame.groupby("Code")[columns].mean()
    counts = frame.groupby("Code")[columns].count()
    return means.where(counts.eq(end - start + 1)), counts


def panel_model(
    frame,
    exposure,
    outcome,
    energy,
    first_difference=False,
    country_trends=False,
    omit_difference_ending=None,
    regional_years=False,
):
    """Balanced-panel FWL, with country-cluster CR1 and full-model rank correction."""
    frame = frame.sort_values(["Code", "Year"]).copy()
    if first_difference and regional_years:
        raise ValueError("Regional year effects are implemented for the levels model.")
    assert not (first_difference and country_trends)
    assert not frame.duplicated(["Code", "Year"]).any()
    assert frame.groupby("Code").Year.nunique().eq(frame.Year.nunique()).all()
    regressors = ["x", "logGDP"] + (["energy"] if energy else [])
    frame["x"] = frame[exposure] / 100
    frame["logGDP"] = np.log(frame.GDP)
    frame["energy"] = frame.Total_energy / 100
    columns = [outcome] + regressors
    if omit_difference_ending is not None and not first_difference:
        raise ValueError("An omitted change requires a first-difference model.")
    if first_difference:
        frame[columns] = frame.groupby("Code")[columns].diff()
        frame = frame.dropna(subset=columns)
        if omit_difference_ending is not None:
            frame = frame.loc[frame.Year != omit_difference_ending]
        transformed = frame[columns] - frame.groupby("Year")[columns].transform("mean")
        fixed_rank = frame.Year.nunique()
    else:
        if regional_years:
            # Balanced panel. Country effects are nested in regions, so remove
            # country means and region-year means, adding back each region mean.
            if (
                country_trends
                or frame.region.isna().any()
                or not frame.groupby("Code").region.nunique().eq(1).all()
            ):
                raise ValueError(
                    "Regional year effects require one fixed region per country and no country trends."
                )
            transformed = (
                frame[columns]
                - frame.groupby("Code")[columns].transform("mean")
                - frame.groupby(["region", "Year"])[columns].transform("mean")
                + frame.groupby("region")[columns].transform("mean")
            )
            fixed_rank = frame.Code.nunique() + frame.region.nunique() * (
                frame.Year.nunique() - 1
            )
        else:
            transformed = (
                frame[columns]
                - frame.groupby("Code")[columns].transform("mean")
                - frame.groupby("Year")[columns].transform("mean")
                + frame[columns].mean()
            )
            fixed_rank = frame.Code.nunique() + frame.Year.nunique() - 1
        if country_trends:
            # In a balanced panel, centred time is orthogonal to country effects.
            # Year effects already absorb the shared linear trend, leaving G-1
            # additional country-specific slope directions.
            time = frame.Year - frame.Year.mean()
            products = transformed.mul(time, axis=0)
            slopes = products.groupby(frame.Code).transform("sum")
            time_squares = time.pow(2).groupby(frame.Code).transform("sum")
            transformed = transformed - slopes.div(time_squares, axis=0).mul(
                time, axis=0
            )
            fixed_rank += frame.Code.nunique() - 1
    design = transformed[regressors].to_numpy()
    y = transformed[outcome].to_numpy()
    assert np.linalg.matrix_rank(design) == len(regressors)
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta
    score = pd.DataFrame(design * residual[:, None], index=frame.index)
    score["Code"] = frame.Code
    cluster_score = score.groupby("Code").sum().to_numpy()
    n = len(frame)
    groups = frame.Code.nunique()
    rank = fixed_rank + len(regressors)
    bread = np.linalg.inv(design.T @ design)
    correction = groups / (groups - 1) * (n - 1) / (n - rank)
    covariance = correction * bread @ (cluster_score.T @ cluster_score) @ bread
    se = np.sqrt(covariance[0, 0])
    critical = stats.t.ppf(0.975, groups - 1)
    p = 2 * stats.t.sf(abs(beta[0] / se), groups - 1)
    return dict(
        n_countries=groups,
        n_observations=n,
        years=frame.Year.nunique(),
        design_rank=rank,
        residual_df=n - rank,
        cluster_df=groups - 1,
        beta=beta[0],
        se=se,
        ci_low=beta[0] - critical * se,
        ci_high=beta[0] + critical * se,
        p=p,
    )


def run(args):
    source = Path(args.sources)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    snapshot = pd.read_csv(args.snapshot).set_index("Code")
    raw = pd.read_csv(source / "dietary.csv")
    raw = raw[raw.Code.isin(snapshot.index) & raw.Year.between(1992, 2017)].copy()
    assert not raw.duplicated(["Code", "Year"]).any()
    panel = raw[["Code", "Year"]].copy()
    for food, column in DIRECT.items():
        panel[food] = raw[column]
    panel["Red meat"] = raw[RED].sum(axis=1, min_count=3)
    panel["Selected_cereals"] = raw[SELECTED].sum(axis=1, min_count=5)
    for filename, label in [("cereals", "Cereals"), ("total", "Total_energy")]:
        panel = panel.merge(
            indicator(source, filename, label),
            on=["Code", "Year"],
            how="left",
            validate="one_to_one",
        )
    for outcome, filename in HEALTH.items():
        data = pd.read_csv(source / (filename + ".csv"))
        data = data[["Code", "Year", data.columns[3]]].rename(
            columns={data.columns[3]: outcome}
        )
        panel = panel.merge(
            data.dropna(subset=["Code"]),
            on=["Code", "Year"],
            how="left",
            validate="one_to_one",
        )
    data = pd.read_csv(source / "gdp.csv")[["Code", "Year", "GDP per capita"]]
    panel = panel.merge(
        data.rename(columns={"GDP per capita": "GDP"}).dropna(subset=["Code"]),
        on=["Code", "Year"],
        how="left",
        validate="one_to_one",
    )
    panel = panel.sort_values(["Code", "Year"])
    assert len(panel) == 158 * 26
    numeric_values = panel.drop(columns="Code").to_numpy(dtype=float)
    assert np.isfinite(numeric_values[~np.isnan(numeric_values)]).all()
    assert (panel.GDP.dropna() > 0).all()
    assert (panel.Total_energy.dropna() > 0).all()
    assert (panel[FOODS].ge(0) | panel[FOODS].isna()).all().all()
    panel.to_csv(output / "annual_panel_1992_2017.csv", index=False)
    chart_metadata = json.loads(
        (source / "dietary-composition-by-country.metadata.json").read_text()
    )
    source_columns = list(raw.columns[3:])
    assert len(source_columns) == len(chart_metadata["columns"])
    mapping = []
    for raw_label, (metadata_label, item) in zip(
        source_columns, chart_metadata["columns"].items()
    ):
        mapping.append(
            dict(
                csv_column=raw_label,
                source_label=metadata_label.split("||")[0].strip(),
                indicator_id=item["owidVariableId"],
                catalog_column=item["shortName"],
                catalog_path="grapher/faostat/2026-02-25/faostat_fbsc/faostat_fbsc_flat#"
                + item["shortName"],
            )
        )
    for filename, label in [("cereals", "Cereals"), ("total", "Total_energy")]:
        item = json.loads((source / (filename + ".metadata.json")).read_text())
        mapping.append(
            dict(
                csv_column=label,
                source_label=item["name"].split("||")[0].strip(),
                indicator_id=item["id"],
                catalog_column=item["shortName"],
                catalog_path=item["catalogPath"],
            )
        )
    pd.DataFrame(mapping).to_csv(output / "source_item_mapping.csv", index=False)
    columns = FOODS + OUTCOMES + ["GDP", "Total_energy", "Selected_cereals"]
    common, common_counts = country_means(panel, columns, 1992, 2014)
    full, full_counts = country_means(panel, columns, 1992, 2017)
    pre, pre_counts = country_means(panel, columns, 1992, 2009)
    full["Diabetes1992-2014"] = common["Diabetes1992-2014"]
    for label, means, counts in [
        ("common_1992_2014", common, common_counts),
        ("full_1992_2017", full, full_counts),
        ("prebreak_1992_2009", pre, pre_counts),
    ]:
        means.to_csv(output / (label + "_means.csv"))
        counts.to_csv(output / (label + "_coverage.csv"))

    results = []
    availability = []
    panel_results = []
    for outcome in OUTCOMES:
        for food in FOODS:
            required = [food, outcome, "GDP", "Total_energy"]
            eligible = full.dropna(subset=required).index.intersection(
                snapshot.dropna(subset=["GDP"]).index
            )
            for code in snapshot.index:
                missing = [c for c in required if pd.isna(full.loc[code, c])]
                source_complete = not missing
                supplied_gdp_available = pd.notna(snapshot.loc[code, "GDP"])
                if pd.isna(snapshot.loc[code, "GDP"]):
                    missing.append("supplied_GDP")
                availability.append(
                    dict(
                        exposure=food,
                        outcome=outcome,
                        Code=code,
                        source_complete=source_complete,
                        supplied_GDP_available=supplied_gdp_available,
                        eligible=code in eligible,
                        unavailable=" | ".join(missing),
                    )
                )
            original = snapshot.loc[eligible].copy()
            updated = full.loc[eligible].copy()
            matched = common.loc[eligible].copy()
            early = pre.loc[eligible].copy()
            for data in [matched, early]:
                data["logGDP"] = np.log(data.GDP)
            specifications = [
                ("supplied_same_countries", original, ["GDP"]),
                ("updated_original_windows", updated, ["GDP"]),
                ("updated_common_window", matched, ["GDP"]),
                ("updated_common_window_energy", matched, ["GDP", "Total_energy"]),
                (
                    "updated_common_window_logGDP_energy",
                    matched,
                    ["logGDP", "Total_energy"],
                ),
                ("updated_pre2010_logGDP_energy", early, ["logGDP", "Total_energy"]),
            ]
            for specification, data, controls in specifications:
                results.append(
                    dict(
                        specification=specification,
                        exposure=food,
                        outcome=outcome,
                        **association(data, food, outcome, controls),
                    )
                )
            annual = panel[panel.Code.isin(eligible) & panel.Year.between(1992, 2014)]
            assert len(annual) == 23 * len(eligible)
            assert annual[required].notna().all().all()
            for specification, energy, difference, end, trends in [
                ("TWFE_logGDP", False, False, 2014, False),
                ("TWFE_logGDP_energy", True, False, 2014, False),
                ("First_difference_logGDP_energy", True, True, 2014, False),
                ("TWFE_pre2010_logGDP_energy", True, False, 2009, False),
                ("TWFE_country_trends_logGDP_energy", True, False, 2014, True),
            ]:
                result = panel_model(
                    annual[annual.Year <= end],
                    food,
                    outcome,
                    energy,
                    difference,
                    trends,
                )
                panel_results.append(
                    dict(
                        specification=specification,
                        exposure=food,
                        outcome=outcome,
                        **result,
                    )
                )

    results = add_fdr(pd.DataFrame(results), ["specification"])
    results.to_csv(output / "reconstructed_cross_sectional.csv", index=False)
    availability = pd.DataFrame(availability)
    availability.to_csv(output / "pair_specific_eligibility.csv", index=False)
    panel_results = add_fdr(pd.DataFrame(panel_results), ["specification"])
    panel_results.to_csv(output / "annual_models.csv", index=False)
    cereal_results = []
    for outcome in OUTCOMES:
        eligible = full.dropna(
            subset=["Cereals", "Selected_cereals", outcome, "GDP", "Total_energy"]
        ).index
        eligible = eligible.intersection(snapshot.dropna(subset=["GDP"]).index)
        data = common.loc[eligible]
        for food in ["Selected_cereals", "Cereals"]:
            cereal_results.append(
                dict(
                    exposure=food,
                    outcome=outcome,
                    **association(data, food, outcome, ["GDP"]),
                )
            )
    cereals = pd.DataFrame(cereal_results)
    cereals["q_BH_6_tests"] = multipletests(cereals.p, method="fdr_bh")[1]
    cereals.to_csv(output / "cereal_definition_comparison.csv", index=False)
    # Current-source totals and selected aggregates have identical population denominators.
    audit = common[["Cereals", "Selected_cereals", "Total_energy"]].copy()
    audit["omitted_cereal_energy"] = audit.Cereals - audit.Selected_cereals
    audit["supplied_Calories"] = snapshot.Calories
    audit["supplied_Cereals"] = snapshot.Cereals
    audit.to_csv(output / "energy_and_cereal_audit.csv")
    for name, table in [("cross_sectional", results), ("panel", panel_results)]:
        counts = table.groupby("specification").agg(
            tests=("p", "size"), BH=("q_BH", lambda s: int((s < 0.05).sum()))
        )
        counts.to_csv(output / (name + "_discovery_counts.csv"))
        print(name, counts.to_string())
    manifest = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(source.iterdir())
        if p.suffix in [".csv", ".json", ".py", ".pdf"]
    }
    metadata = dict(
        download_date="2026-11-10",
        source_hashes=manifest,
        food_version="OWID FAOSTAT combined food balances 2026-02-25",
        diet_population="OWID population denominators, as described in source metadata",
        method_break="Old FBS through 2009, new FBS from 2010. No additional splice adjustment.",
        annual_missingness="Require every year in the stated window. Blank source cells remain missing. No imputation.",
        aggregation="Arithmetic means over all years. Red meat requires all three components. Selected cereals requires all five.",
        comparison_sample="Per-pair complete annual coverage through 2017, diabetes through 2014, within the supplied 143 GDP-complete units.",
        current_cereals="FAO item 2905 Cereals - Excluding Beer, OWID indicator 1200369",
        current_energy="FAO item 2901 Total, OWID indicator 1203765",
        panel="Country and year fixed effects, contemporaneous exposure, log GDP, optional total energy. Country-cluster CR1 with full design rank correction and t(G-1).",
        first_difference="Annual differences and year effects, differenced log GDP and energy. Country-cluster CR1 with t(G-1).",
        country_trends="Country and year fixed effects plus country-specific linear time trends, log GDP and total energy. Full rank 2G+T-2+3. Country-cluster CR1 with t(G-1). Uses deviations from fitted linear country trajectories.",
        multiplicity="BH and BY across 48 pairs for each specification. Cereal definition comparisons form a separate six-test family.",
        scope="Post-review sensitivity using frozen current-source releases. Historical original downloads and pre-aggregation exclusions remain unverified.",
    )
    (output / "reconstruction_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
