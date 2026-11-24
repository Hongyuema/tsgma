"""Independently validate common-window diagnostics and empirical exposure scales."""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from audit_analysis import FOODS, OUTCOMES


def regression_check(design, outcome):
    inverse = np.linalg.inv(design.T @ design)
    beta = inverse @ design.T @ outcome
    residuals = outcome - design @ beta
    rank = np.linalg.matrix_rank(design)
    df = len(outcome) - rank
    se = np.sqrt(np.sum(residuals**2) / df * inverse[1, 1])
    t_value = beta[1] / se
    partial_r = t_value / np.sqrt(df + t_value**2)
    leverage = np.sum((design @ inverse) * design, axis=1)
    scores = design * (residuals / (1 - leverage))[:, None]
    hc3 = inverse @ scores.T @ scores @ inverse
    hc3_se = np.sqrt(hc3[1, 1])
    p_value = 2 * stats.t.sf(abs(beta[1] / hc3_se), df)
    margin = stats.t.ppf(0.975, df) * hc3_se
    return partial_r, [beta[1], hc3_se, beta[1] - margin, beta[1] + margin, p_value]


def verify(args):
    annual = Path(args.annual_results)
    diagnostic = Path(args.diagnostics)
    means = pd.read_csv(annual / "common_1992_2014_means.csv").set_index("Code")
    countries = pd.read_csv(args.countries).set_index("Code")
    means["region"] = countries.region
    eligibility = pd.read_csv(annual / "pair_specific_eligibility.csv")
    results = pd.read_csv(diagnostic / "common_window_diagnostics.csv")
    omissions = pd.read_csv(diagnostic / "common_window_omissions.csv")
    annual_results = pd.read_csv(diagnostic / "annual_exposure_scales.csv")
    scales = pd.read_csv(diagnostic / "exposure_scales.csv")
    panel = pd.read_csv(annual / "annual_panel_1992_2017.csv")
    fits_checked = 0
    for outcome in OUTCOMES:
        for exposure in FOODS:
            codes = eligibility.loc[
                (eligibility.exposure == exposure)
                & (eligibility.outcome == outcome)
                & eligibility.eligible,
                "Code",
            ]
            frame = means.loc[codes].sort_index()
            predictors = np.column_stack(
                [frame[exposure] / 100, np.log(frame.GDP), frame.Total_energy / 100]
            )
            x = np.column_stack([np.ones(len(frame)), predictors])
            y = frame[outcome].to_numpy()
            r, verified = regression_check(x, y)
            row = results[
                (results.exposure == exposure)
                & (results.outcome == outcome)
                & (results.specification == "HC3_logGDP_energy")
            ].iloc[0]
            np.testing.assert_allclose(
                verified,
                row[["beta", "se", "ci_low", "ci_high", "p"]].to_numpy(float),
                rtol=1e-8,
                atol=1e-10,
            )
            np.testing.assert_allclose(r, row.r, rtol=1e-8, atol=1e-10)
            for specification in ["Rank_logGDP_energy", "Region_logGDP_energy"]:
                if specification.startswith("Rank"):
                    other_x = np.column_stack(
                        [np.ones(len(frame)), stats.rankdata(predictors, axis=0)]
                    )
                    other_y = stats.rankdata(y)
                else:
                    other_x = np.column_stack(
                        [x, pd.get_dummies(frame.region, drop_first=True, dtype=float)]
                    )
                    other_y = y
                partial_r, _ = regression_check(other_x, other_y)
                observed = results[
                    (results.exposure == exposure)
                    & (results.outcome == outcome)
                    & (results.specification == specification)
                ].iloc[0]
                np.testing.assert_allclose(partial_r, observed.r, rtol=1e-8, atol=1e-10)
                df = len(frame) - np.linalg.matrix_rank(other_x)
                p_value = 2 * stats.t.sf(
                    abs(partial_r * np.sqrt(df / (1 - partial_r**2))), df
                )
                np.testing.assert_allclose(p_value, observed.p, rtol=1e-8, atol=1e-10)
            omitted_results = omissions[
                (omissions.exposure == exposure) & (omissions.outcome == outcome)
            ]
            for omitted in omitted_results.itertuples():
                keep = frame.index != omitted.omitted
                if omitted.omission_type == "region":
                    keep = frame.region != omitted.omitted
                partial_r, _ = regression_check(x[keep], y[keep])
                np.testing.assert_allclose(partial_r, omitted.r, rtol=1e-8, atol=1e-10)
                assert sum(keep) == omitted.n
                fits_checked += 1
            sample = panel[
                panel.Code.isin(codes) & panel.Year.between(1992, 2014)
            ].copy()
            sample = sample.sort_values(["Code", "Year"])
            sample["logGDP"] = np.log(sample.GDP)
            sums = {}
            for specification in [
                "TWFE_logGDP_energy",
                "First_difference_logGDP_energy",
                "TWFE_country_trends_logGDP_energy",
            ]:
                current = sample.copy()
                columns = [exposure, outcome, "logGDP", "Total_energy"]
                difference = specification.startswith("First_difference")
                if difference:
                    current[columns] = current.groupby("Code")[columns].diff()
                    current = current.dropna(subset=columns)
                nuisance = pd.get_dummies(current.Year, drop_first=False, dtype=float)
                if not difference:
                    country_dummies = pd.get_dummies(
                        current.Code, drop_first=True, dtype=float
                    )
                    nuisance = pd.concat([nuisance, country_dummies], axis=1)
                if "country_trends" in specification:
                    trend_terms = country_dummies.mul(
                        current.Year - current.Year.mean(), axis=0
                    )
                    nuisance = pd.concat([nuisance, trend_terms], axis=1)
                design = np.column_stack(
                    [nuisance, current.logGDP, current.Total_energy / 100]
                )
                values = current[[exposure, outcome]].to_numpy()
                residuals = (
                    values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
                )
                residual_ss = np.sum(residuals**2, axis=0)
                rms = np.sqrt(residual_ss / len(current))
                observed = annual_results[
                    (annual_results.exposure == exposure)
                    & (annual_results.outcome == outcome)
                    & (annual_results.specification == specification)
                ].iloc[0]
                np.testing.assert_allclose(
                    rms,
                    [
                        observed.residual_exposure_rms_kcal,
                        observed.residual_outcome_rms,
                    ],
                    rtol=1e-8,
                    atol=1e-10,
                )
                sums[specification] = residual_ss[0]
            observed_scale = scales[
                (scales.exposure == exposure) & (scales.outcome == outcome)
            ].iloc[0]
            retained = (
                sums["TWFE_country_trends_logGDP_energy"] / sums["TWFE_logGDP_energy"]
            )
            np.testing.assert_allclose(
                retained,
                observed_scale.trend_retained_exposure_variance_fraction,
                rtol=1e-8,
                atol=1e-10,
            )
    for _, frame in results.groupby("specification"):
        assert len(frame) == 48
        np.testing.assert_allclose(
            frame.q_BH, multipletests(frame.p, method="fdr_bh")[1], atol=1e-12
        )
        np.testing.assert_allclose(
            frame.q_BY, multipletests(frame.p, method="fdr_by")[1], atol=1e-12
        )
    for _, frame in scales.groupby("exposure"):
        assert frame.n.nunique() == 1
        assert frame.iqr.nunique() == 1
        np.testing.assert_allclose(
            frame.trend_retained_exposure_variance_fraction,
            frame.trend_retained_exposure_variance_fraction.iloc[0],
            atol=1e-12,
        )
    print(
        "Verified 48 HC3 covariance matrices, 96 rank/regional estimates,",
        fits_checked,
        "omission fits,",
    )
    print(
        "144 annual residual scales against explicit dummy-variable projections, and every BH/BY family."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annual-results", required=True)
    parser.add_argument("--countries", required=True)
    parser.add_argument("--diagnostics", required=True)
    verify(parser.parse_args())
