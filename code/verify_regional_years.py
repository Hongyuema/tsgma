"""Verify the regional-year model against OLS with explicit country and region-year indicators."""

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.api import OLS
from statsmodels.stats.multitest import multipletests


def verify(root):
    annual = root / "annual_results"
    data = pd.read_csv(annual / "annual_panel_1992_2017.csv").query(
        "1992 <= Year <= 2014"
    )
    data = data.merge(
        pd.read_csv(root / "corrected_results/countries_and_missingness.csv")[
            ["Code", "region"]
        ],
        on="Code",
        validate="many_to_one",
    )
    eligible = pd.read_csv(annual / "pair_specific_eligibility.csv").query("eligible")
    results = pd.read_csv(root / "diagnostics/region_year_effects.csv")
    for row in results.itertuples():
        units = eligible[
            (eligible.exposure == row.exposure) & (eligible.outcome == row.outcome)
        ].Code
        sample = data[data.Code.isin(units)].sort_values(["Code", "Year"])
        assert (
            sample.groupby("Code").Year.nunique().eq(23).all()
            and sample.Code.nunique() == row.n_countries
        )
        country = pd.get_dummies(sample.Code, dtype=float)
        ry = pd.get_dummies(
            sample.region.str.strip() + "_" + sample.Year.astype(str), dtype=float
        )
        ry = ry.loc[:, ~ry.columns.str.endswith("_1992")]
        design = np.column_stack(
            (
                sample[row.exposure] / 100,
                np.log(sample.GDP),
                sample.Total_energy / 100,
                country,
                ry,
            )
        )
        model = OLS(sample[row.outcome].to_numpy(), design).fit(
            cov_type="cluster",
            cov_kwds={"groups": sample.Code.to_numpy(), "use_correction": True},
            use_t=True,
        )
        np.testing.assert_allclose(
            [model.params[0], model.bse[0], *model.conf_int()[0], model.pvalues[0]],
            [row.beta, row.se, row.ci_low, row.ci_high, row.p],
            rtol=1e-8,
            atol=1e-9,
        )
        assert (
            model.nobs == row.n_observations
            and model.df_resid == row.residual_df
            and design.shape[1] == row.design_rank
        )
    for name, method in [("q_BH", "fdr_bh"), ("q_BY", "fdr_by")]:
        np.testing.assert_allclose(
            results[name], multipletests(results.p, method=method)[1], atol=1e-12
        )
    print(
        "Verified all 48 regional-year fits, coefficient intervals, design ranks and both FDR adjustments."
    )


if __name__ == "__main__":
    verify(Path("."))
