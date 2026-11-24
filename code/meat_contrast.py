"""Post-review equal-energy poultry versus red-meat contrast, 1992–2014."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from audit_analysis import FOODS, OUTCOMES, add_fdr


def run(annual_results, output):
    means = pd.read_csv(annual_results / "common_1992_2014_means.csv").set_index("Code")
    roster = pd.read_csv(annual_results / "pair_specific_eligibility.csv")
    flags = roster.pivot(
        index="Code", columns=["exposure", "outcome"], values="eligible"
    )
    assert flags.shape[1] == 48 and flags.notna().all().all()
    common = means.loc[flags.index[flags.all(axis=1)]].sort_index()
    assert len(common) == 138
    # Units of 10 kcal/person/day. All food and energy columns use the same scale.
    design = pd.DataFrame(
        {
            "intercept": 1.0,
            "logGDP": np.log(common.GDP),
            "energy": common.Total_energy / 10,
        },
        index=common.index,
    ).join(common[FOODS] / 10)
    poultry, red = "Poultry meat", "Red meat"
    contrast = pd.Series(0.0, index=design.columns)
    contrast.loc[[poultry, red]] = [1.0, -1.0]
    # In this equivalent basis a one-unit increase in poultry at fixed combined meat
    # requires a one-unit decrease in red meat. The poultry slope equals the contrast.
    fixed_sum = design.copy()
    fixed_sum[red] = design[poultry] + design[red]
    residual = OLS(common[poultry], fixed_sum.drop(columns=poultry)).fit().resid
    residual_rms = float(np.sqrt(np.mean(residual**2)))
    assert np.linalg.matrix_rank(design) == design.shape[1]
    rows = []
    for outcome in OUTCOMES:
        fitted = OLS(common[outcome], design).fit(cov_type="HC3", use_t=True)
        test = fitted.t_test(contrast.to_numpy())
        low, high = test.conf_int()[0]
        beta, se, p = (
            float(test.effect.item()),
            float(test.sd.item()),
            float(test.pvalue.item()),
        )
        equivalent = OLS(common[outcome], fixed_sum).fit(cov_type="HC3", use_t=True)
        np.testing.assert_allclose(
            [beta, se, p, low, high],
            [
                equivalent.params[poultry],
                equivalent.bse[poultry],
                equivalent.pvalues[poultry],
                *equivalent.conf_int().loc[poultry],
            ],
            atol=1e-10,
            rtol=1e-9,
        )
        rows.append(
            dict(
                outcome=outcome,
                family="poultry_for_red_three_outcomes",
                n=len(common),
                df=int(fitted.df_resid),
                increment_kcal=10,
                beta=beta,
                se_HC3=se,
                ci_low=low,
                ci_high=high,
                p=p,
                conditional_poultry_rms_kcal=residual_rms,
            )
        )
    output.mkdir(exist_ok=True, parents=True)
    add_fdr(pd.DataFrame(rows), ["family"]).to_csv(
        output / "poultry_for_red.csv", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annual-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.annual_results, args.output)
