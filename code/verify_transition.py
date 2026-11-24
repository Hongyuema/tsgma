"""Verify transition-year sensitivity with explicit year indicators and cluster covariance."""

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.api import OLS
from statsmodels.stats.multitest import multipletests


def verify(root):
    folder = root / "annual_results"
    annual = pd.read_csv(folder / "annual_panel_1992_2017.csv").query(
        "1992 <= Year <= 2014"
    )
    roster = pd.read_csv(folder / "pair_specific_eligibility.csv").query("eligible")
    table = pd.read_csv(root / "comparison/first_difference_year_omissions.csv")
    baseline = pd.read_csv(folder / "annual_models.csv").query(
        "specification == 'First_difference_logGDP_energy'"
    )
    for (food, outcome), units in roster.groupby(["exposure", "outcome"]):
        frame = annual[annual.Code.isin(units.Code)].sort_values(["Code", "Year"])
        n = len(units)
        values = (
            frame[[outcome, food, "GDP", "Total_energy"]].to_numpy().reshape(n, 23, 4)
        )
        values[:, :, 2] = np.log(values[:, :, 2])
        values[:, :, [1, 3]] /= 100
        changes = np.diff(values, axis=1)
        rows = table[(table.exposure == food) & (table.outcome == outcome)].set_index(
            "omitted_end_year"
        )
        if sorted(rows.index) != list(range(1993, 2015)):
            raise ValueError("Missing or repeated comparison year.")
        full = baseline[
            (baseline.exposure == food) & (baseline.outcome == outcome)
        ].iloc[0]
        for year, entry in rows.iterrows():
            assert year - 1 == entry.omitted_start_year
            subset = changes[:, np.arange(1993, 2015) != year, :].reshape(-1, 4)
            # Retained differences are created before masking one interval; they never bridge missing years.
            design = np.column_stack([subset[:, 1:], np.tile(np.eye(21), (n, 1))])
            model = OLS(subset[:, 0], design).fit(
                cov_type="cluster",
                cov_kwds={
                    "groups": np.repeat(np.arange(n), 21),
                    "use_correction": True,
                },
                use_t=True,
            )
            np.testing.assert_allclose(
                [
                    model.params[0],
                    model.bse[0],
                    *model.conf_int()[0],
                    model.pvalues[0],
                    full.beta,
                    full.q_BH,
                ],
                [
                    entry.beta,
                    entry.se,
                    entry.ci_low,
                    entry.ci_high,
                    entry.p,
                    entry.full_beta,
                    entry.full_q_BH,
                ],
                atol=1e-10,
                rtol=1e-8,
            )
            assert (
                entry.n_observations == model.nobs == 21 * n
                and entry.n_countries == n
                and entry.residual_df == model.df_resid
            )
    for _, rows in table.groupby("omitted_end_year"):
        assert len(rows) == 48
        np.testing.assert_allclose(
            multipletests(rows.p, method="fdr_bh")[1], rows.q_BH, atol=1e-12
        )
    print(
        "Verified all 1,056 interval-omission fits, 22 BH families and full-window reference coefficients."
    )


if __name__ == "__main__":
    verify(Path("."))
