"""Check matched year links, all fixed-lag slopes and cluster intervals independently."""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from statsmodels.stats.multitest import multipletests


def check(root):
    fit = pd.read_csv(root / "fixed_lags/estimates.csv")
    source = pd.read_csv(root / "annual_results/annual_panel_1992_2017.csv").set_index(
        ["Code", "Year"]
    )
    roster = pd.read_csv(root / "annual_results/pair_specific_eligibility.csv")
    # Explicit year indicators, country indicators and optional country trends.
    # QR projection is cached for pairs with the same balanced design.
    cache = {}
    for row in fit.itertuples(index=False):
        entry = roster[
            (roster.exposure == row.exposure)
            & (roster.outcome == row.outcome)
            & roster.eligible
        ]
        codes = sorted(entry.Code)
        g = len(codes)
        years = np.arange(1997, 2015)
        n = 18 * g
        key = (g, row.country_trends)
        if key not in cache:
            country = np.repeat(np.eye(g), 18, axis=0)[:, 1:]
            yrs = np.tile(np.eye(18), (g, 1))
            terms = [yrs, country]
            if row.country_trends:
                terms.append(country * np.tile(years - years.mean(), g)[:, None])
            z = np.column_stack(terms)
            cache[key] = np.linalg.qr(z, mode="reduced")[0]
        q = cache[key]

        def index(time):
            return pd.MultiIndex.from_product([codes, time], names=["Code", "Year"])

        y = source.loc[index(years), row.outcome].to_numpy()
        x = (
            source.loc[
                index(years - row.lag_years), [row.exposure, "GDP", "Total_energy"]
            ]
            .to_numpy()
            .copy()
        )
        x[:, 0] /= 100
        x[:, 1] = np.log(x[:, 1])
        x[:, 2] /= 100
        y = y - q @ (q.T @ y)
        x = x - q @ (q.T @ x)
        b = np.linalg.lstsq(x, y, rcond=None)[0]
        res = y - x @ b
        # Influence for the exposure coefficient computed separately for each country.
        influence = (x @ np.linalg.inv(x.T @ x)[:, 0]) * res
        rank = q.shape[1] + 3
        se = np.sqrt(
            sum(influence.reshape(g, 18).sum(1) ** 2)
            * g
            / (g - 1)
            * (n - 1)
            / (n - rank)
        )
        lim = t.ppf(0.975, g - 1) * se
        np.testing.assert_allclose(
            [b[0], se, b[0] - lim, b[0] + lim, 2 * t.sf(abs(b[0] / se), g - 1)],
            [row.beta, row.se, row.ci_low, row.ci_high, row.p],
            rtol=1e-8,
            atol=1e-10,
        )
        assert (row.n_countries, row.n_observations, row.design_rank) == (g, n, rank)
    for group, col in [
        (["country_trends", "lag_years"], "q_BH_48"),
        (["country_trends"], "q_BH_144"),
    ]:
        for _, rows in fit.groupby(group):
            np.testing.assert_allclose(
                rows[col], multipletests(rows.p, method="fdr_bh")[1], atol=1e-12
            )
    print("Verified 288 matched-year fits and both sets of BH values.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    check(ap.parse_args().root)
