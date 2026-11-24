"""Rebuild all grading inputs from year-keyed records and explicit indicator designs."""

from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, t
from statsmodels.stats.multitest import multipletests


def verify(root, extended=False):
    annual = root / ("grading_1990_2014" if extended else "annual_results")
    source = pd.read_csv(
        annual / ("panel_1985_2014.csv" if extended else "annual_panel_1992_2017.csv")
    ).set_index(["Code", "Year"])
    roster = pd.read_csv(annual / "pair_specific_eligibility.csv").query("eligible")
    files = annual if extended else root / "fixed_lags"
    results = pd.read_csv(files / "third_star_grades.csv")
    profiles = (
        pd.read_csv(files / "unrestricted_lag_profile.csv")
        .set_index(["exposure", "outcome"])
        .sort_index()
    )
    years = np.arange(1990 if extended else 1997, 2015)
    npcheck = lambda x, y: np.testing.assert_allclose(x, y, rtol=2e-8, atol=1e-10)

    @lru_cache(None)
    def nuisance(g, trends):
        country = np.repeat(np.eye(g)[:, 1:], len(years), axis=0)
        yr = np.tile(np.eye(len(years)), (g, 1))
        cols = [yr, country]
        if trends:
            cols.append(country * np.tile(years - years.mean(), g)[:, None])
        return np.linalg.qr(np.column_stack(cols), mode="reduced")[0]

    for a in results.itertuples(index=False):
        codes = sorted(
            roster.loc[
                (roster.exposure == a.exposure) & (roster.outcome == a.outcome), "Code"
            ]
        )
        g = len(codes)
        assert g == a.n and g * len(years) == a.country_years

        def frame(ys):
            return source.loc[pd.MultiIndex.from_product([codes, ys])]

        means = (
            frame(np.arange(1990 if extended else 1992, 2015))
            .groupby(level=0)
            .mean(numeric_only=True)
        )
        x = means[a.exposure].to_numpy()
        y = means[a.outcome].to_numpy()
        crude = pearsonr(x, y)
        cov = np.column_stack([np.ones(g), np.log(means.GDP), means.Total_energy])
        resid = lambda v: v - cov @ np.linalg.lstsq(cov, v, rcond=None)[0]
        adj = pearsonr(resid(x), resid(y)).statistic
        p = 2 * t.sf(abs(adj) * np.sqrt((g - 4) / (1 - adj**2)), g - 4)
        npcheck(
            [crude.statistic, crude.pvalue, adj, p],
            [a.crude_r, a.crude_p, a.adjusted_r, a.adjusted_p],
        )
        series = [frame(years - k) for k in range(6)]
        x, gdp, energy = [
            np.column_stack([v[col] for v in series])
            for col in (a.exposure, "GDP", "Total_energy")
        ]
        x = x / 100
        gdp = np.log(gdp)
        energy = energy / 100
        controls = [
            x[:, 0],
            gdp[:, 0],
            energy[:, 0],
            gdp[:, 1:].mean(1),
            energy[:, 1:].mean(1),
        ]
        y = series[0][a.outcome].to_numpy()
        for trends, free in [(True, False), (False, False), (True, True)]:
            mat = np.column_stack([x[:, 1:] if free else x[:, 1:].mean(1), *controls])
            q = nuisance(g, trends)
            z = mat - q @ (q.T @ mat)
            v = y - q @ (q.T @ y)
            inv = np.linalg.pinv(z)
            b = inv @ v
            err = v - z @ b
            contributions = (inv * err).reshape(len(b), g, len(years)).sum(axis=2)
            n = len(v)
            rank = q.shape[1] + len(b)
            covariance = (
                contributions @ contributions.T * (g / (g - 1)) * ((n - 1) / (n - rank))
            )
            contrast = np.r_[np.ones(5 if free else 1), np.zeros(len(controls))]
            slope = contrast @ b
            se = np.sqrt(contrast @ covariance @ contrast)
            lim = se * t.ppf(0.975, g - 1)
            pvalue = 2 * t.sf(abs(slope / se), g - 1)
            if free:
                npcheck(
                    [slope, slope - lim, slope + lim, pvalue],
                    [
                        a.unrestricted_sum,
                        a.unrestricted_lo,
                        a.unrestricted_hi,
                        a.unrestricted_p,
                    ],
                )
                individual = profiles.loc[(a.exposure, a.outcome)].sort_values("lag")
                npcheck(individual.lag, np.arange(1, 6))
                ci = np.sqrt(np.diag(covariance)[:5]) * t.ppf(0.975, g - 1)
                npcheck(
                    individual[["beta", "lo", "hi"]],
                    np.column_stack([b[:5], b[:5] - ci, b[:5] + ci]),
                )
            elif trends:
                npcheck(
                    [slope, se, slope - lim, slope + lim, pvalue],
                    [a.beta, a.se, a.lo, a.hi, a.p],
                )
            else:
                npcheck([slope, pvalue], [a.without_trends, a.without_trends_p])
    for name in ["crude", "adjusted", "lag"]:
        prob = results["p" if name == "lag" else name + "_p"]
        npcheck(multipletests(prob, method="fdr_bh")[1], results[name + "_q"])
    pj = np.where(
        results.adjusted_r * results.beta > 0,
        np.maximum(results.adjusted_p, results.p),
        1,
    )
    order = np.argsort(pj)
    adjusted = np.empty(len(pj))
    adjusted[order] = np.minimum(
        1, np.maximum.accumulate(pj[order] * np.arange(len(pj), 0, -1))
    )
    npcheck(pj, results.directional_p)
    npcheck(adjusted, results.joint_Holm)
    levels = np.where(
        results.adjusted_q < 0.05,
        2 + (adjusted < 0.05).astype(int),
        (results.crude_q < 0.05).astype(int),
    )
    np.testing.assert_array_equal(levels, results.stars)
    print(
        "Verified all 48 grades, 144 lag fits, 240 unrestricted lag coefficients and their intervals."
    )


if __name__ == "__main__":
    verify(Path.cwd())
    verify(Path.cwd(), extended=True)
