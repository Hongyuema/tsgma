"""Diagnose the reconstructed common-window estimates and annual exposure scale.

All diagnostics are post-review additions. Pair-specific country sets are read
from the existing reconstruction and remain fixed for full-sample comparisons.
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS

from audit_analysis import FOODS, OUTCOMES, association, add_fdr, residualize


def annual_residuals(frame, food, outcome, specification):
    """Residualize exposure and outcome against every other model term."""
    frame = frame.sort_values(["Code", "Year"]).copy()
    assert not frame.duplicated(["Code", "Year"]).any()
    assert frame.groupby("Code").Year.nunique().eq(23).all()
    frame["logGDP"] = np.log(frame.GDP)
    columns = [food, outcome, "logGDP", "Total_energy"]
    if specification == "First_difference_logGDP_energy":
        frame[columns] = frame.groupby("Code")[columns].diff()
        frame = frame.dropna(subset=columns)
        values = frame[columns] - frame.groupby("Year")[columns].transform("mean")
    else:
        values = (
            frame[columns]
            - frame.groupby("Code")[columns].transform("mean")
            - frame.groupby("Year")[columns].transform("mean")
            + frame[columns].mean()
        )
        if specification == "TWFE_country_trends_logGDP_energy":
            time = frame.Year - frame.Year.mean()
            denominator = time.pow(2).groupby(frame.Code).transform("sum")
            slopes = values.mul(time, axis=0).groupby(frame.Code).transform("sum")
            values = values - slopes.div(denominator, axis=0).mul(time, axis=0)
    residuals = residualize(
        values[[food, outcome]].to_numpy(),
        values[["logGDP", "Total_energy"]].to_numpy(),
    )
    return residuals


def run(args):
    annual = Path(args.annual_results)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    means = pd.read_csv(annual / "common_1992_2014_means.csv").set_index("Code")
    means["logGDP"] = np.log(means.GDP)
    countries = pd.read_csv(args.countries).set_index("Code")
    means["region"] = countries.region
    assert means.region.notna().all()
    region_columns = pd.get_dummies(
        means.region, prefix="region", drop_first=True, dtype=float
    )
    means = means.join(region_columns)
    eligibility = pd.read_csv(annual / "pair_specific_eligibility.csv")
    baseline = pd.read_csv(annual / "reconstructed_cross_sectional.csv")
    baseline = baseline[baseline.specification == "updated_common_window_logGDP_energy"]
    panel = pd.read_csv(annual / "annual_panel_1992_2017.csv")
    coefficients = pd.read_csv(annual / "annual_models.csv")
    specifications = [
        "TWFE_logGDP_energy",
        "First_difference_logGDP_energy",
        "TWFE_country_trends_logGDP_energy",
    ]
    estimates, omissions, influences, scales, annual_scales = [], [], [], [], []
    for outcome in OUTCOMES:
        for food in FOODS:
            pair = dict(exposure=food, outcome=outcome)
            codes = eligibility.loc[
                (eligibility.exposure == food)
                & (eligibility.outcome == outcome)
                & eligibility.eligible,
                "Code",
            ]
            sample = means.loc[codes].sort_index()
            assert sample[[food, outcome, "logGDP", "Total_energy"]].notna().all().all()
            controls = ["logGDP", "Total_energy"]
            base = baseline[
                (baseline.exposure == food) & (baseline.outcome == outcome)
            ].iloc[0]
            check = association(sample, food, outcome, controls)
            np.testing.assert_allclose(
                [check["r"], check["p"]], [base.r, base.p], rtol=1e-10
            )
            design = np.column_stack(
                [
                    np.ones(len(sample)),
                    sample[food] / 100,
                    sample.logGDP,
                    sample.Total_energy / 100,
                ]
            )
            fit = OLS(sample[outcome].to_numpy(), design).fit()
            robust = fit.get_robustcov_results(cov_type="HC3", use_t=True)
            estimates.append(
                dict(
                    **pair,
                    specification="HC3_logGDP_energy",
                    n=len(sample),
                    r=base.r,
                    beta=fit.params[1],
                    se=robust.bse[1],
                    ci_low=robust.conf_int()[1, 0],
                    ci_high=robust.conf_int()[1, 1],
                    p=robust.pvalues[1],
                    residual_df=fit.df_resid
                )
            )
            for specification, covariates, ranks in [
                ("Rank_logGDP_energy", controls, True),
                ("Region_logGDP_energy", controls + list(region_columns), False),
            ]:
                estimates.append(
                    dict(
                        **pair,
                        specification=specification,
                        **association(sample, food, outcome, covariates, rank=ranks)
                    )
                )
            diagnostic = fit.get_influence()
            for i, code in enumerate(sample.index):
                reduced = sample.drop(code)
                result = association(reduced, food, outcome, controls)
                omissions.append(
                    dict(
                        **pair,
                        omission_type="country",
                        omitted=code,
                        n=result["n"],
                        r=result["r"],
                        baseline_r=base.r
                    )
                )
                influences.append(
                    dict(
                        **pair,
                        Code=code,
                        n=len(sample),
                        cooks_d=diagnostic.cooks_distance[0][i],
                        leverage=diagnostic.hat_matrix_diag[i],
                        studentized_residual=diagnostic.resid_studentized_external[i]
                    )
                )
            for region in sorted(sample.region.unique()):
                result = association(
                    sample[sample.region != region], food, outcome, controls
                )
                omissions.append(
                    dict(
                        **pair,
                        omission_type="region",
                        omitted=region,
                        n=result["n"],
                        r=result["r"],
                        baseline_r=base.r
                    )
                )
            q25, q50, q75 = sample[food].quantile([0.25, 0.5, 0.75])
            iqr = q75 - q25
            scales.append(
                dict(
                    **pair,
                    n=len(sample),
                    minimum=sample[food].min(),
                    q25=q25,
                    median=q50,
                    q75=q75,
                    maximum=sample[food].max(),
                    iqr=iqr,
                    zero_count=int(sample[food].eq(0).sum()),
                    hundred_kcal_in_iqr=100 / iqr,
                    beta_per_iqr=fit.params[1] * iqr / 100,
                    hc3_ci_low_per_iqr=robust.conf_int()[1, 0] * iqr / 100,
                    hc3_ci_high_per_iqr=robust.conf_int()[1, 1] * iqr / 100
                )
            )
            annual_sample = panel[
                panel.Code.isin(codes) & panel.Year.between(1992, 2014)
            ]
            residual_sums = {}
            for specification in specifications:
                residuals = annual_residuals(
                    annual_sample, food, outcome, specification
                )
                exposure_ss, outcome_ss = np.sum(residuals**2, axis=0)
                residual_sums[specification] = exposure_ss
                exposure_rms = np.sqrt(exposure_ss / len(residuals))
                coefficient = coefficients[
                    (coefficients.exposure == food)
                    & (coefficients.outcome == outcome)
                    & (coefficients.specification == specification)
                ].iloc[0]
                fwl_beta = 100 * np.dot(residuals[:, 0], residuals[:, 1]) / exposure_ss
                np.testing.assert_allclose(
                    fwl_beta, coefficient.beta, rtol=1e-8, atol=1e-10
                )
                annual_scales.append(
                    dict(
                        **pair,
                        specification=specification,
                        n_countries=len(codes),
                        n_observations=len(residuals),
                        residual_exposure_ss=exposure_ss,
                        residual_exposure_rms_kcal=exposure_rms,
                        residual_outcome_rms=np.sqrt(outcome_ss / len(residuals)),
                        beta_per_exposure_rms=coefficient.beta * exposure_rms / 100,
                        ci_low_per_exposure_rms=coefficient.ci_low * exposure_rms / 100,
                        ci_high_per_exposure_rms=coefficient.ci_high
                        * exposure_rms
                        / 100
                    )
                )
            retained = (
                residual_sums[specifications[2]] / residual_sums[specifications[0]]
            )
            assert 0 <= retained <= 1 + 1e-10
            scales[-1]["trend_retained_exposure_variance_fraction"] = retained

    estimates = add_fdr(pd.DataFrame(estimates), ["specification"])
    estimates.to_csv(output / "common_window_diagnostics.csv", index=False)
    omissions = pd.DataFrame(omissions)
    omissions["absolute_change"] = abs(omissions.r - omissions.baseline_r)
    omissions["sign_reversal"] = omissions.r * omissions.baseline_r < 0
    omissions.to_csv(output / "common_window_omissions.csv", index=False)
    summary = []
    for (food, outcome, kind), frame in omissions.groupby(
        ["exposure", "outcome", "omission_type"], sort=False
    ):
        largest = frame.loc[frame.absolute_change.idxmax()]
        summary.append(
            dict(
                exposure=food,
                outcome=outcome,
                omission_type=kind,
                n_omissions=len(frame),
                r_min=frame.r.min(),
                r_max=frame.r.max(),
                sign_reversals=int(frame.sign_reversal.sum()),
                largest_change=largest.absolute_change,
                largest_change_omitted=largest.omitted,
                largest_change_r=largest.r,
                largest_change_n=largest.n,
            )
        )
    pd.DataFrame(summary).to_csv(
        output / "common_window_omission_summary.csv", index=False
    )
    pd.DataFrame(influences).to_csv(output / "common_window_influence.csv", index=False)
    scales = pd.DataFrame(scales)
    scales.to_csv(output / "exposure_scales.csv", index=False)
    pd.DataFrame(annual_scales).to_csv(
        output / "annual_exposure_scales.csv", index=False
    )
    counts = estimates.groupby("specification").agg(
        tests=("p", "size"), BH=("q_BH", lambda x: int((x < 0.05).sum()))
    )
    counts.to_csv(output / "diagnostic_discovery_counts.csv")
    metadata = dict(
        sample="Existing pair-specific 140–143 countries with complete annual coverage, fixed across full-sample diagnostic specifications.",
        HC3="One food per model, scaled per 100 kcal/person/day, log GDP and total energy. t inference uses n-4 residual degrees of freedom.",
        rank="Average ranks for ties in exposure, outcome, log GDP and total energy. Approximate partial-correlation t and Fisher-z inference.",
        regions="Fixed effects for the existing frozen World Bank regions. Conventional independence-based partial-correlation inference.",
        omissions="Re-estimate partial correlation after each single country or single region omission. Report coefficient changes descriptively. No exclusion is applied to the reported full-sample fit.",
        multiplicity="Three separate 48-test BH and BY families for HC3, rank and region specifications. This is post-review sensitivity analysis.",
        annual_scale="Residualize each exposure and outcome on all other terms of each existing energy-adjusted annual model. Exposure scale is sqrt(sum(residual_exposure**2)/n), in kcal/person/day.",
        retained_variance="Exposure residual sum of squares in the country-trend model divided by that in the two-way fixed-effects model on the identical sample.",
        rescaled_effect="Multiply the existing coefficient and its two interval endpoints by the positive observed exposure scale divided by 100. Conditional on this empirical scale. The original p and q values are unchanged.",
        limitation="HC3 allows heteroskedasticity. Region indicators absorb mean regional differences. Neither method estimates a general cross-country covariance structure. Exposure residual variation is a descriptive design diagnostic and does not identify measurement reliability or statistical power.",
    )
    (output / "diagnostics_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2)
    )
    print(counts.to_string())
    print("Omission fits", len(omissions), "annual scale rows", len(annual_scales))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annual-results", required=True)
    parser.add_argument("--countries", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
