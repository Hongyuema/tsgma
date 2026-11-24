"""Independently check annual model coefficients and cluster inference.

python verify_annual.py --results annual_results
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests

from audit_analysis import FOODS, OUTCOMES, HEADLINES


def record_dependence(frame, model, identifiers, country_rows, summaries):
    # Align only consecutive years within a country; each correlation uses that country's own means.
    residuals = frame[["Code", "Year"]].copy()
    residuals["residual"] = model.resid
    wide = residuals.pivot(index="Code", columns="Year", values="residual")
    if wide.isna().any().any() or not (np.diff(wide.columns) == 1).all():
        raise ValueError("Residual diagnostics require complete, contiguous years.")
    matrix = wide.to_numpy()
    before, after = matrix[:, :-1], matrix[:, 1:]
    left = before - before.mean(axis=1, keepdims=True)
    right = after - after.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    correlations = np.divide(
        (left * right).sum(axis=1),
        denominator,
        out=np.full(len(wide), np.nan),
        where=denominator > 0,
    )
    for code, value in zip(wide.index, correlations, strict=True):
        country_rows.append(
            {
                **identifiers,
                "Code": code,
                "adjacent_pairs": before.shape[1],
                "residual_r1": value,
            }
        )
    valid = correlations[np.isfinite(correlations)]
    if not len(valid):
        raise ValueError("No estimable adjacent-year residual correlations.")
    q25, median, q75 = np.quantile(valid, [0.25, 0.5, 0.75])
    iid_se = np.sqrt(model.scale * model.normalized_cov_params[0, 0])
    summaries.append(
        {
            **identifiers,
            "n_countries": len(wide),
            "n_estimable": len(valid),
            "within_country_q25": q25,
            "within_country_median": median,
            "within_country_q75": q75,
            "iid_se": iid_se,
            "country_cluster_se": model.bse[0],
            "cluster_to_iid_se": model.bse[0] / iid_se,
        }
    )


def verify(results, dependence_dir=None):
    country_rows, summaries = [], []
    annual = pd.read_csv(results / "annual_panel_1992_2017.csv")
    eligibility = pd.read_csv(results / "pair_specific_eligibility.csv")
    estimates = pd.read_csv(results / "annual_models.csv")
    checked = 0
    for outcome in OUTCOMES:
        for exposure in FOODS:
            codes = eligibility.loc[
                (eligibility.exposure == exposure)
                & (eligibility.outcome == outcome)
                & eligibility.eligible,
                "Code",
            ]
            for specification, end, difference, energy, trends in [
                ("TWFE_logGDP", 2014, False, False, False),
                ("TWFE_logGDP_energy", 2014, False, True, False),
                ("First_difference_logGDP_energy", 2014, True, True, False),
                ("TWFE_pre2010_logGDP_energy", 2009, False, True, False),
                ("TWFE_country_trends_logGDP_energy", 2014, False, True, True),
            ]:
                frame = (
                    annual[annual.Code.isin(codes) & annual.Year.between(1992, end)]
                    .sort_values(["Code", "Year"])
                    .copy()
                )
                frame["x"] = frame[exposure] / 100
                frame["logGDP"] = np.log(frame.GDP)
                frame["energy"] = frame.Total_energy / 100
                predictors = ["x", "logGDP"] + (["energy"] if energy else [])
                columns = [outcome] + predictors
                if difference:
                    frame[columns] = frame.groupby("Code")[columns].diff()
                    frame = frame.dropna(subset=columns)
                dummies = pd.get_dummies(
                    frame.Year, prefix="year", drop_first=False, dtype=float
                )
                if not difference:
                    dummies = pd.concat(
                        [
                            dummies,
                            pd.get_dummies(frame.Code, drop_first=True, dtype=float),
                        ],
                        axis=1,
                    )
                if trends:
                    trend_design = pd.get_dummies(
                        frame.Code, drop_first=True, dtype=float
                    )
                    trend_design = trend_design.mul(
                        frame.Year - frame.Year.mean(), axis=0
                    )
                    dummies = pd.concat([dummies, trend_design], axis=1)
                design = np.column_stack([frame[predictors], dummies])
                model = OLS(frame[outcome].to_numpy(), design).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": frame.Code, "use_correction": True},
                    use_t=True,
                )
                row = estimates[
                    (estimates.exposure == exposure)
                    & (estimates.outcome == outcome)
                    & (estimates.specification == specification)
                ].iloc[0]
                np.testing.assert_allclose(
                    [
                        model.params[0],
                        model.bse[0],
                        model.pvalues[0],
                        *model.conf_int()[0],
                    ],
                    [row.beta, row.se, row.p, row.ci_low, row.ci_high],
                    rtol=1e-8,
                    atol=1e-9,
                )
                assert model.df_resid == row.residual_df
                assert model.nobs == row.n_observations
                assert np.linalg.matrix_rank(design) == row.design_rank
                assert frame.Code.nunique() - 1 == row.cluster_df
                if dependence_dir is not None and energy and end == 2014:
                    identifiers = {
                        "specification": specification,
                        "outcome": outcome,
                        "exposure": exposure,
                    }
                    record_dependence(
                        frame, model, identifiers, country_rows, summaries
                    )
                checked += 1
    if dependence_dir is not None:
        dependence_dir.mkdir(exist_ok=True, parents=True)
        if len(summaries) != 144:
            raise ValueError(
                "Expected 48 pairs in each of three annual specifications."
            )
        pd.DataFrame(summaries).to_csv(
            dependence_dir / "annual_residual_dependence.csv", index=False
        )
        pd.DataFrame(country_rows).to_csv(
            dependence_dir / "annual_residual_by_country.csv", index=False
        )
    for _, frame in estimates.groupby("specification"):
        np.testing.assert_allclose(
            frame.q_BH,
            multipletests(frame.p, method="fdr_bh")[1],
            rtol=1e-9,
            atol=1e-12,
        )
    cross = pd.read_csv(results / "reconstructed_cross_sectional.csv")
    assert cross.groupby(["exposure", "outcome"]).n.nunique().eq(1).all()
    assert len(annual) == 158 * 26 and not annual.duplicated(["Code", "Year"]).any()
    selected = estimates.apply(lambda r: (r.exposure, r.outcome) in HEADLINES, axis=1)
    assert (
        (estimates.specification == "First_difference_logGDP_energy")
        & selected
        & (estimates.q_BH < 0.05)
    ).sum() == 0
    print(
        f"All {checked} annual fits match explicit dummy-variable OLS and statsmodels country-cluster inference."
    )
    print(
        "Country-year uniqueness, fixed comparison samples and BH calculations verified."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--dependence-dir",
        type=Path,
        help="Optional output for descriptive residual diagnostics.",
    )
    args = parser.parse_args()
    verify(args.results, args.dependence_dir)
