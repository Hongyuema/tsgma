"""Common-window dietary composition, shape and outcome-scale sensitivities."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from patsy import dmatrix
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from audit_analysis import FOODS, OUTCOMES, association, add_fdr


def run(annual, output):
    output.mkdir(parents=True, exist_ok=True)
    means = pd.read_csv(annual / "common_1992_2014_means.csv").set_index("Code")
    eligibility = pd.read_csv(annual / "pair_specific_eligibility.csv")
    assert not eligibility.duplicated(["Code", "exposure", "outcome"]).any()
    all_ok = eligibility.groupby("Code").eligible.agg(["all", "size"])
    assert all_ok["size"].eq(48).all()
    common = means.loc[all_ok.index[all_ok["all"]]].sort_index().copy()
    assert len(common) == 138
    common["remainder"] = common.Total_energy - common[FOODS].sum(axis=1)
    assert common.remainder.min() > 0
    common[["Total_energy", "remainder"]].to_csv(output / "common_sample.csv")
    means["logGDP"] = np.log(means.GDP)
    design = pd.DataFrame(
        {
            "intercept": 1.0,
            "logGDP": np.log(common.GDP),
            "energy": common.Total_energy / 100,
        },
        index=common.index,
    ).join(common[FOODS] / 100)
    assert np.linalg.matrix_rank(design) == design.shape[1]
    vifs = [
        variance_inflation_factor(design.values, i)
        for i in range(1, len(design.columns))
    ]
    pd.DataFrame({"term": design.columns[1:], "VIF": vifs}).to_csv(
        output / "collinearity.csv", index=False
    )
    slopes, shape, scales, stability = [], [], [], []
    for outcome in OUTCOMES:
        joint = OLS(common[outcome], design).fit(cov_type="HC3", use_t=True)
        for code in common.index:
            keep = common.index != code
            omitted = np.linalg.lstsq(
                design.values[keep], common[outcome].values[keep], rcond=None
            )[0]
            stability.extend(
                dict(exposure=f, outcome=outcome, omitted=code, beta=b)
                for f, b in zip(FOODS, omitted[3:])
            )
        for food in FOODS:
            pair = dict(exposure=food, outcome=outcome)
            single_design = design[["intercept", "logGDP", "energy", food]]
            matched = OLS(common[outcome], single_design).fit(
                cov_type="HC3", use_t=True
            )
            for spec, fit in [("matched_one_food", matched), ("all_foods", joint)]:
                lo, hi = fit.conf_int().loc[food]
                slopes.append(
                    dict(
                        **pair,
                        family=spec,
                        n=len(common),
                        beta=fit.params[food],
                        se=fit.bse[food],
                        ci_low=lo,
                        ci_high=hi,
                        p=fit.pvalues[food],
                    )
                )
            codes = eligibility.loc[
                (eligibility.outcome == outcome)
                & (eligibility.exposure == food)
                & eligibility.eligible,
                "Code",
            ]
            s = means.loc[codes].copy()
            x = s[food].values / 100
            z = (x - x.min()) / (x.max() - x.min())
            k = np.quantile(z, [0, 1 / 3, 2 / 3, 1])
            assert len(np.unique(k)) == 4

            # Natural cubic spline with linear tails; the linear fit is nested.
            def cubic(t):
                return np.maximum(z - t, 0) ** 3

            extra = [
                cubic(t)
                - (k[-1] - t) / (k[-1] - k[-2]) * cubic(k[-2])
                + (k[-2] - t) / (k[-1] - k[-2]) * cubic(k[-1])
                for t in k[:2]
            ]
            linear = np.column_stack(
                [np.ones(len(s)), s.logGDP, s.Total_energy / 100, z]
            )
            full = np.column_stack([linear, *extra])
            fit = OLS(s[outcome].values, full).fit(cov_type="HC3", use_t=True)
            test = fit.f_test(np.eye(6)[-2:])
            # Independent spline basis, with a direct sandwich/Wald calculation.
            basis = np.asarray(
                dmatrix("cr(z, knots=internal) - 1", dict(z=z, internal=k[1:3]))
            )
            alternative = np.column_stack([linear[:, 1:3], basis])
            pinv = np.linalg.pinv(alternative)
            coef = pinv @ s[outcome].values
            pred = alternative @ coef
            np.testing.assert_allclose(pred, fit.fittedvalues, rtol=1e-8, atol=1e-8)
            lever = np.sum(alternative * pinv.T, axis=1)
            sandwich = pinv * ((s[outcome].values - pred) / (1 - lever))
            transform = np.linalg.lstsq(full, alternative, rcond=None)[0][-2:]
            b = transform @ coef
            robust = transform @ sandwich
            f_value = b @ np.linalg.solve(robust @ robust.T, b) / 2
            np.testing.assert_allclose(f_value, float(test.fvalue), rtol=1e-7)
            shape.append(
                dict(
                    **pair,
                    n=len(s),
                    F=float(test.fvalue),
                    df_num=2,
                    df_den=len(s) - 6,
                    **{
                        f"knot{i+1}_kcal": v
                        for i, v in enumerate(
                            np.quantile(x * 100, [0, 1 / 3, 2 / 3, 1])
                        )
                    },
                    p=float(test.pvalue),
                    family="nonlinear_shape",
                )
            )
            if outcome in OUTCOMES[1:]:
                y = s[outcome]
                assert y.between(0, 100, inclusive="neither").all()
                s["logit"] = np.log(y / (100 - y))
                for spec, target in [
                    ("prevalence", outcome),
                    ("logit_prevalence", "logit"),
                ]:
                    scales.append(
                        dict(
                            **pair,
                            family=spec,
                            **association(s, food, target, ["logGDP", "Total_energy"]),
                        )
                    )
    for name, rows in [
        ("dietary_slopes", slopes),
        ("spline_tests", shape),
        ("prevalence_scales", scales),
    ]:
        result = add_fdr(pd.DataFrame(rows), ["family"])
        result.to_csv(output / (name + ".csv"), index=False)
        print(result.groupby("family").q_BH.agg(["size", lambda x: sum(x < 0.05)]))
    pd.DataFrame(stability).to_csv(output / "joint_omissions.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annual-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.annual_results, args.output)
