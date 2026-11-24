"""Reproduce and audit the supplied country-level TSGMA analysis snapshot.

Usage: python audit_analysis.py --input Foods-1992-2017.xlsx --regions worldbank_regions.json --output audit
All added specifications are post-review exploratory sensitivity analyses.
No annual diet or outcome data are reconstructed by this script.
"""

from pathlib import Path
import argparse
import hashlib
import json
import platform

import numpy as np
import pandas as pd
from scipy import stats
from patsy import dmatrix
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor


FOODS = [
    "Red meat",
    "Alcoholic Beverages",
    "Animal fats group",
    "Milk",
    "Eggs",
    "Sugar and Sweeteners",
    "Vegetable Oils",
    "Nuts",
    "Poultry meat",
    "Vegetables",
    "Pulses",
    "Cereals",
    "Fish and seafood",
    "Starchy Roots",
    "Fruit",
    "Other meat",
]
OUTCOMES = ["Non-HDL", "Diabetes1992-2014", "Hypertension"]
HEADLINES = [
    ("Animal fats group", "Non-HDL"),
    ("Red meat", "Non-HDL"),
    ("Eggs", "Non-HDL"),
    ("Poultry meat", "Diabetes1992-2014"),
    ("Cereals", "Diabetes1992-2014"),
    ("Sugar and Sweeteners", "Diabetes1992-2014"),
    ("Starchy Roots", "Diabetes1992-2014"),
    ("Alcoholic Beverages", "Hypertension"),
]


def residualize(values, controls):
    design = np.column_stack([np.ones(len(values)), controls])
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def association(frame, exposure, outcome, controls, rank=False):
    columns = [exposure, outcome] + controls
    sample = frame.dropna(subset=columns)
    values = sample[columns].to_numpy(dtype=float)
    if rank:
        values = stats.rankdata(values, axis=0)
    x = residualize(values[:, 0], values[:, 2:])
    y = residualize(values[:, 1], values[:, 2:])
    n = len(sample)
    k = np.linalg.matrix_rank(np.column_stack([np.ones(n), values[:, 2:]])) - 1
    r = float(np.corrcoef(x, y)[0, 1])
    degrees = n - k - 2
    p = float(2 * stats.t.sf(abs(r * np.sqrt(degrees / (1 - r * r))), degrees))
    margin = stats.norm.ppf(0.975) / np.sqrt(n - k - 3)
    low, high = np.tanh(np.arctanh(r) + np.array([-margin, margin]))
    return dict(
        n=n, controls_count=int(k), r=r, ci_low=float(low), ci_high=float(high), p=p
    )


def add_fdr(frame, family):
    frame = frame.copy()
    group_key = family[0] if len(family) == 1 else family
    for _, indices in frame.groupby(group_key, sort=False).groups.items():
        p = frame.loc[indices, "p"].to_numpy()
        frame.loc[indices, "q_BH"] = multipletests(p, method="fdr_bh")[1]
        frame.loc[indices, "q_BY"] = multipletests(p, method="fdr_by")[1]
        frame.loc[indices, "p_Bonferroni"] = np.minimum(p * len(p), 1)
    return frame


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.input)
    data = (
        pd.read_csv(source, float_precision="round_trip")
        if source.suffix.lower() == ".csv"
        else pd.read_excel(source)
    ).iloc[:, :27]
    assert list(data.columns[1:17]) == FOODS
    assert data.Code.notna().all() and not data.Code.duplicated().any()
    numeric = data.columns.drop("Code")
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="raise")
    values = data[numeric].to_numpy(dtype=float)
    assert np.isfinite(values[~np.isnan(values)]).all()
    assert (data.GDP.dropna() > 0).all()
    data.to_csv(output / "analysis_snapshot.csv", index=False)
    source_missing = data.isna().sum()
    data["logGDP"] = np.log(data.GDP)
    wb = json.loads(Path(args.regions).read_text())[1]
    lookup = {row["id"]: row for row in wb if row["region"]["id"] != "NA"}
    data["country_name"] = data.Code.map(lambda c: lookup.get(c, {}).get("name", c))
    data["region"] = data.Code.map(
        lambda c: lookup.get(c, {}).get("region", {}).get("value")
    )
    assert data.region.notna().all(), data.loc[data.region.isna(), "Code"].tolist()
    countries = data[["Code", "country_name", "region"]].copy()
    countries["available_GDP"] = data.GDP.notna()
    countries["available_schooling"] = data["Mean years schooling"].notna()
    countries.to_csv(output / "countries_and_missingness.csv", index=False)
    missing = pd.DataFrame(
        {
            "variable": numeric,
            "n_available": data[numeric].notna().sum().values,
            "n_missing": source_missing[numeric].values,
        }
    )
    missing.to_csv(output / "missingness_summary.csv", index=False)
    descriptions = data[numeric].describe(percentiles=[0.25, 0.5, 0.75]).T
    descriptions["zero_count"] = (data[numeric] == 0).sum()
    descriptions["skewness"] = data[numeric].skew()
    descriptions.to_csv(output / "variable_distributions.csv")

    region_columns = pd.get_dummies(
        data.region, prefix="region", drop_first=True, dtype=float
    )
    data = pd.concat([data, region_columns], axis=1)
    specs = {
        "Pearson_158": ([], False, data),
        "Pearson_GDP_complete_143": ([], False, data.dropna(subset=["GDP"])),
        "GDP_partial_143": (["GDP"], False, data),
        "Spearman_158": ([], True, data),
        "Spearman_GDP_partial_143": (["GDP"], True, data),
        "logGDP_partial_143": (["logGDP"], False, data),
        "Median_age_only_158": (["Median age"], False, data),
        "BMI_only_158": (["BMI>25"], False, data),
        "Recorded_calories_only_158": (["Calories"], False, data),
        "Sex_ratio_only_158": (["Gender ratio"], False, data),
        "Tobacco_only_158": (["Smoke"], False, data),
        "Schooling_only_126": (["Mean years schooling"], False, data),
        "Pearson_school_complete_126": (
            [],
            False,
            data.dropna(subset=["Mean years schooling"]),
        ),
        "Joint_context_143": (
            ["logGDP", "Median age", "Gender ratio", "Smoke"],
            False,
            data,
        ),
        "Joint_context_BMI_143": (
            ["logGDP", "Median age", "Gender ratio", "Smoke", "BMI>25"],
            False,
            data,
        ),
        "GDP_region_partial_143": (["GDP"] + list(region_columns), False, data),
    }
    results = []
    for spec, (controls, rank, frame) in specs.items():
        for outcome in OUTCOMES:
            for exposure in FOODS:
                row = association(frame, exposure, outcome, controls, rank)
                results.append(
                    dict(specification=spec, exposure=exposure, outcome=outcome, **row)
                )
    results = add_fdr(pd.DataFrame(results), ["specification"])
    results.to_csv(output / "all_associations.csv", index=False)
    primary = results[
        results.specification.isin(["Pearson_158", "GDP_partial_143"])
    ].copy()
    primary["q_BH_96_tests"] = multipletests(primary.p, method="fdr_bh")[1]
    primary.to_csv(output / "crude_and_GDP_96_tests.csv", index=False)

    multivariable = []
    model_summary = []
    influence = []
    vif = []
    for outcome in OUTCOMES:
        complete = data.dropna(subset=FOODS + ["GDP", outcome]).copy()
        predictors = complete[FOODS + ["GDP"]].copy()
        predictors[FOODS] /= 100
        predictors["GDP"] /= 10000
        design = np.column_stack([np.ones(len(complete)), predictors.to_numpy()])
        model = OLS(complete[outcome].to_numpy(), design).fit()
        hc3 = model.get_robustcov_results(cov_type="HC3", use_t=True)
        model_summary.append(
            dict(
                outcome=outcome,
                n=len(complete),
                predictors=17,
                residual_df=model.df_resid,
                r_squared=model.rsquared,
                adjusted_r_squared=model.rsquared_adj,
                F=model.fvalue,
                p=model.f_pvalue,
            )
        )
        loo_coefficients = []
        for omitted in range(len(complete)):
            mask = np.arange(len(complete)) != omitted
            loo_coefficients.append(
                np.linalg.lstsq(
                    design[mask], complete[outcome].to_numpy()[mask], rcond=None
                )[0]
            )
        loo_coefficients = np.asarray(loo_coefficients)
        for j, predictor in enumerate(predictors, 1):
            multivariable.append(
                dict(
                    outcome=outcome,
                    exposure=predictor,
                    n=len(complete),
                    beta=model.params[j],
                    se=model.bse[j],
                    ci_low=model.conf_int()[j, 0],
                    ci_high=model.conf_int()[j, 1],
                    p=model.pvalues[j],
                    hc3_se=hc3.bse[j],
                    hc3_ci_low=hc3.conf_int()[j, 0],
                    hc3_ci_high=hc3.conf_int()[j, 1],
                    hc3_p=hc3.pvalues[j],
                    loo_beta_min=loo_coefficients[:, j].min(),
                    loo_beta_max=loo_coefficients[:, j].max(),
                    loo_sign_reversal=bool(
                        np.any(loo_coefficients[:, j] * model.params[j] < 0)
                    ),
                )
            )
            vif.append(
                dict(
                    outcome=outcome,
                    predictor=predictor,
                    VIF=variance_inflation_factor(design, j),
                )
            )
        diag = model.get_influence()
        for j, code in enumerate(complete.Code):
            influence.append(
                dict(
                    outcome=outcome,
                    Code=code,
                    cooks_d=diag.cooks_distance[0][j],
                    leverage=diag.hat_matrix_diag[j],
                    studentized_residual=diag.resid_studentized_external[j],
                    cooks_flag=bool(diag.cooks_distance[0][j] > 4 / len(complete)),
                )
            )
    multivariable = pd.DataFrame(multivariable)
    for pcol, qcol in [
        ("p", "q_BH_48_food_tests"),
        ("hc3_p", "hc3_q_BH_48_food_tests"),
    ]:
        rows = multivariable.exposure.isin(FOODS)
        multivariable.loc[rows, qcol] = multipletests(
            multivariable.loc[rows, pcol], method="fdr_bh"
        )[1]
    multivariable.to_csv(output / "multivariable_OLS_and_HC3.csv", index=False)
    pd.DataFrame(model_summary).to_csv(output / "model_summary.csv", index=False)
    pd.DataFrame(vif).to_csv(output / "VIF.csv", index=False)
    pd.DataFrame(influence).to_csv(output / "influential_countries.csv", index=False)

    # Additional shape and scale checks across all candidate pairs, with no filtering.
    nonlinear = []
    bounded = []
    for outcome in OUTCOMES:
        for exposure in FOODS:
            complete = data.dropna(subset=[exposure, outcome, "GDP"]).copy()
            x = complete[exposure].to_numpy() / 100
            y = complete[outcome].to_numpy()
            gdp = complete.GDP.to_numpy() / 10000
            linear_design = np.column_stack([np.ones(len(complete)), gdp, x])
            linear = OLS(y, linear_design).fit()
            spline = np.asarray(
                dmatrix('cr(x, df=3, constraints="center") - 1', {"x": x})
            )
            spline_design = np.column_stack([np.ones(len(complete)), gdp, spline])
            # Project the spline basis onto intercept/linear exposure. SVD retains
            # exactly two nonlinear directions, making the HC3 joint test explicit.
            nonlinear_basis = residualize(spline, np.column_stack([gdp, x]))
            u, singular_values, _ = np.linalg.svd(nonlinear_basis, full_matrices=False)
            rank = int(np.sum(singular_values > singular_values.max() * 1e-10))
            assert rank == 2
            design = np.column_stack([linear_design, u[:, :rank]])
            fit = OLS(y, design).fit(cov_type="HC3", use_t=True)
            restriction = np.zeros((rank, design.shape[1]))
            restriction[:, -rank:] = np.eye(rank)
            test = fit.f_test(restriction)
            nonlinear.append(
                dict(
                    exposure=exposure,
                    outcome=outcome,
                    n=len(complete),
                    statistic=float(test.fvalue),
                    p=float(test.pvalue),
                    extra_df=rank,
                    linear_r_squared=linear.rsquared,
                    spline_r_squared=fit.rsquared,
                )
            )
            if outcome != "Non-HDL":
                assert ((complete[outcome] > 0) & (complete[outcome] < 100)).all()
                complete["logit_outcome"] = np.log(
                    complete[outcome] / (100 - complete[outcome])
                )
                bounded.append(
                    dict(
                        exposure=exposure,
                        outcome=outcome,
                        **association(complete, exposure, "logit_outcome", ["GDP"])
                    )
                )
    nonlinear = pd.DataFrame(nonlinear)
    nonlinear["q_BH_48_tests"] = multipletests(nonlinear.p, method="fdr_bh")[1]
    nonlinear.to_csv(output / "nonlinearity_HC3_spline_tests.csv", index=False)
    bounded = pd.DataFrame(bounded)
    bounded["q_BH_32_tests"] = multipletests(bounded.p, method="fdr_bh")[1]
    bounded.to_csv(output / "logit_prevalence_sensitivity.csv", index=False)

    leave_country = []
    leave_region = []
    for spec, controls in [
        ("GDP_partial_143", ["GDP"]),
        ("Joint_context_143", ["logGDP", "Median age", "Gender ratio", "Smoke"]),
    ]:
        for outcome in OUTCOMES:
            for exposure in FOODS:
                frame = data.dropna(subset=[exposure, outcome] + controls)
                base = association(frame, exposure, outcome, controls)
                rows = []
                for idx in frame.index:
                    result = association(
                        frame.drop(index=idx), exposure, outcome, controls
                    )
                    rows.append((frame.loc[idx, "Code"], result["r"], result["p"]))
                low = min(rows, key=lambda v: v[1])
                high = max(rows, key=lambda v: v[1])
                leave_country.append(
                    dict(
                        specification=spec,
                        exposure=exposure,
                        outcome=outcome,
                        baseline_r=base["r"],
                        r_min=low[1],
                        omit_for_min=low[0],
                        r_max=high[1],
                        omit_for_max=high[0],
                        p_max=max(v[2] for v in rows),
                        sign_reversal=any(v[1] * base["r"] < 0 for v in rows),
                    )
                )
                for region in sorted(frame.region.unique()):
                    result = association(
                        frame.loc[frame.region != region], exposure, outcome, controls
                    )
                    leave_region.append(
                        dict(
                            specification=spec,
                            omitted_region=region,
                            exposure=exposure,
                            outcome=outcome,
                            **result
                        )
                    )
    pd.DataFrame(leave_country).to_csv(
        output / "leave_one_country_out.csv", index=False
    )
    add_fdr(pd.DataFrame(leave_region), ["specification", "omitted_region"]).to_csv(
        output / "leave_one_region_out.csv", index=False
    )

    selected = results.apply(lambda r: (r.exposure, r.outcome) in HEADLINES, axis=1)
    results[selected].to_csv(output / "original_eight_headline_pairs.csv", index=False)
    counts = results.groupby("specification").agg(
        tests=("p", "size"),
        nominal=("p", lambda s: int((s < 0.05).sum())),
        BH=("q_BH", lambda s: int((s < 0.05).sum())),
        BY=("q_BY", lambda s: int((s < 0.05).sum())),
    )
    counts.to_csv(output / "discovery_counts.csv")
    metadata = dict(
        source_filename=source.name,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        n_rows=len(data),
        foods=FOODS,
        outcomes=OUTCOMES,
        GDP_missing=data.loc[data.GDP.isna(), "Code"].tolist(),
        schooling_missing=data.loc[
            data["Mean years schooling"].isna(), "Code"
        ].tolist(),
        added_analysis_status=(
            "Post-review audit of year-corrected country means"
            if source.suffix.lower() == ".csv"
            else "Post-review exploratory audit of supplied cross-sectional means"
        ),
        confidence_intervals="Approximate Fisher-z, using n-k-3 for k covariates. Rank intervals are approximations.",
        multiplicity="BH and BY separately across 48 food-outcome tests per specification. Additional pooled BH across 96 crude and GDP tests.",
        sample_rules="Complete cases per specification. Country means are supplied values, with no imputation.",
        region_source="World Bank country API snapshot downloaded 2026-11-10. Contemporary region labels.",
        unavailable="Original annual diet/outcome/GDP inputs, original 193-country exclusion frame, original 159-country roster and selection record.",
        interpretation="Country-level conditional associations. P values and intervals condition on point estimates and do not propagate source-estimate uncertainty.",
        python=platform.python_version(),
        numpy=np.__version__,
        pandas=pd.__version__,
    )
    if source.suffix.lower() == ".csv":
        metadata["sample_rules"] = (
            "Original covariates, foods and roster. Year-corrected outcome means. Complete cases by specification, no imputation."
        )
        metadata["unavailable"] = (
            "Contemporaneous reasons for each food-roster exclusion, exact original food inputs and Calories formula, and exact original download dates."
        )
    (output / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2)
    )
    print(counts.to_string())
    print("\nHeadline GDP associations")
    print(
        results[selected & (results.specification == "GDP_partial_143")][
            ["exposure", "outcome", "n", "r", "ci_low", "ci_high", "p", "q_BH"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--regions", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
