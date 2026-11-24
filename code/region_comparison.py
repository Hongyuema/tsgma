"""Repeat the complete between- and within-country screens on matched regional exclusions."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t, false_discovery_control
from statsmodels.api import OLS
from reconstruct_annual import panel_model


def run(root, out):
    input_dir = root / "annual_results"
    panel = pd.read_csv(input_dir / "annual_panel_1992_2017.csv").query(
        "1992<=Year<=2014"
    )
    assignment = (
        pd.read_csv(root / "corrected_results/countries_and_missingness.csv")
        .set_index("Code")
        .region.str.strip()
    )
    eligibility = pd.read_csv(input_dir / "pair_specific_eligibility.csv").query(
        "eligible"
    )
    regions = sorted(assignment.loc[eligibility.Code.unique()].unique())
    assert len(regions) == 7
    means = pd.read_csv(input_dir / "common_1992_2014_means.csv").set_index("Code")
    records = []
    for (x, y), group in eligibility.groupby(["exposure", "outcome"], sort=False):
        for omitted in ["None", *regions]:
            codes = group.Code[assignment.loc[group.Code].to_numpy() != omitted]
            sample = means.loc[codes]
            design = np.column_stack(
                (
                    np.ones(len(codes)),
                    sample[x] / 100,
                    np.log(sample.GDP),
                    sample.Total_energy / 100,
                )
            )
            fit = OLS(sample[y].values, design).fit()
            r = fit.tvalues[1] / np.sqrt(fit.tvalues[1] ** 2 + fit.df_resid)
            robust = fit.get_robustcov_results(cov_type="HC3", use_t=True)
            annual = panel[panel.Code.isin(codes)]
            assert (
                len(annual) == 23 * len(codes)
                and annual[[x, y, "GDP", "Total_energy"]].notna().all().all()
            )
            temporal = panel_model(annual, x, y, energy=True)
            records.append(
                dict(
                    omitted_region=omitted,
                    exposure=x,
                    outcome=y,
                    n=len(codes),
                    between_r=r,
                    partial_p=fit.pvalues[1],
                    between=robust.params[1],
                    between_low=robust.conf_int()[1, 0],
                    between_high=robust.conf_int()[1, 1],
                    between_p=robust.pvalues[1],
                    within=temporal["beta"],
                    within_low=temporal["ci_low"],
                    within_high=temporal["ci_high"],
                    within_p=temporal["p"],
                )
            )
    results = pd.DataFrame(records)
    assert results.groupby("omitted_region").size().eq(48).all()
    for model in ["partial", "between", "within"]:
        results[model + "_q"] = results.groupby("omitted_region")[
            model + "_p"
        ].transform(false_discovery_control)
    out.mkdir(exist_ok=True, parents=True)
    results.to_csv(out / "estimates.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    run(args.root, args.output or args.root / "region_comparison")
