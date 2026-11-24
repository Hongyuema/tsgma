"""Within-country counterpart to the Table S19 poultry-for-red-meat contrast."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from statsmodels.regression.linear_model import OLS

from audit_analysis import FOODS, OUTCOMES, add_fdr


def run(root):
    results = root / "annual_results"
    eligibility = pd.read_csv(results / "pair_specific_eligibility.csv")
    flags = eligibility.pivot(
        index="Code", columns=["exposure", "outcome"], values="eligible"
    )
    assert flags.shape[1] == 48 and flags.notna().all().all()
    roster = flags.index[flags.all(axis=1)]
    annual = pd.read_csv(results / "annual_panel_1992_2017.csv")
    frame = annual.loc[
        annual.Code.isin(roster) & annual.Year.between(1992, 2014)
    ].copy()
    frame = frame.sort_values(["Code", "Year"]).reset_index(drop=True)
    countries, years = frame.Code.nunique(), frame.Year.nunique()
    assert (countries, years) == (138, 23) and len(frame) == countries * years
    assert not frame.duplicated(["Code", "Year"]).any()
    design = (
        frame[FOODS]
        .div(10)
        .assign(energy=frame.Total_energy / 10, logGDP=np.log(frame.GDP))
    )
    combined = pd.concat([design, frame[OUTCOMES]], axis=1)
    assert np.isfinite(combined.to_numpy()).all()
    demeaned = (
        combined
        - combined.groupby(frame.Code).transform("mean")
        - combined.groupby(frame.Year).transform("mean")
        + combined.mean()
    )
    rows = []
    time = frame.Year - frame.Year.mean()
    for trends, family in [
        (False, "within_country_year"),
        (True, "within_country_year_and_country_trends"),
    ]:
        transformed = demeaned
        if trends:
            # In a balanced panel, the country-specific slopes of the two-way
            # demeaned columns sum to zero, preserving the year constraints.
            trend_slopes = demeaned.mul(time, axis=0).groupby(frame.Code).transform(
                "sum"
            ) / np.sum(np.unique(time) ** 2)
            transformed = demeaned - trend_slopes.mul(time, axis=0)
        x = transformed[design.columns].to_numpy()
        obs, slopes = x.shape
        rank = countries + years - 1 + slopes + (countries - 1 if trends else 0)
        assert np.linalg.matrix_rank(x) == slopes
        poultry, red = "Poultry meat", "Red meat"
        contrast = np.zeros(slopes)
        contrast[design.columns.get_loc(poultry)] = 1
        contrast[design.columns.get_loc(red)] = -1
        bread = np.linalg.inv(x.T @ x)
        correction = countries / (countries - 1) * (obs - 1) / (obs - rank)

        # Independent parameterization and full indicator fit checks both the
        # variance of the difference and the absorbed-rank CR1 adjustment.
        fixed_sum = design.copy()
        fixed_sum[red] = design[poultry] + design[red]
        indicators = pd.get_dummies(
            frame[["Code", "Year"]].astype(str), drop_first=True, dtype=float
        )
        full = pd.concat([fixed_sum, indicators], axis=1).assign(intercept=1.0)
        if trends:
            country_indicators = pd.get_dummies(
                frame.Code, drop_first=True, dtype=float
            )
            full = pd.concat(
                [full, country_indicators.mul(time, axis=0).add_prefix("trend_")],
                axis=1,
            )
        assert np.linalg.matrix_rank(full) == rank
        residual_poultry = OLS(full[poultry], full.drop(columns=poultry)).fit().resid
        rms = 10 * np.sqrt(np.mean(residual_poultry**2))
        for outcome in OUTCOMES:
            y = transformed[outcome].to_numpy()
            coef = np.linalg.lstsq(x, y, rcond=None)[0]
            scores = (
                pd.DataFrame(x * (y - x @ coef)[:, None])
                .groupby(frame.Code)
                .sum()
                .to_numpy()
            )
            beta = contrast @ coef
            se = np.sqrt(correction * np.sum((scores @ bread @ contrast) ** 2))
            halfwidth = t.ppf(0.975, countries - 1) * se
            p = 2 * t.sf(abs(beta / se), countries - 1)
            check = OLS(frame[outcome], full).fit(
                cov_type="cluster", cov_kwds={"groups": frame.Code}, use_t=True
            )
            np.testing.assert_allclose(
                [beta, se, p, beta - halfwidth, beta + halfwidth],
                [
                    check.params[poultry],
                    check.bse[poultry],
                    check.pvalues[poultry],
                    *check.conf_int().loc[poultry],
                ],
                rtol=1e-8,
                atol=1e-10,
            )
            rows.append(
                dict(
                    outcome=outcome,
                    family=family,
                    n_countries=countries,
                    n_country_years=obs,
                    design_rank=rank,
                    df=countries - 1,
                    increment_kcal=10,
                    beta=beta,
                    se_CR1=se,
                    ci_low=beta - halfwidth,
                    ci_high=beta + halfwidth,
                    p=p,
                    conditional_poultry_rms_kcal=rms,
                )
            )
    add_fdr(pd.DataFrame(rows), ["family"]).to_csv(
        root / "contrasts/within_poultry_for_red.csv", index=False
    )


if __name__ == "__main__":
    run(Path.cwd())
