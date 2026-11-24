"""Pair-specific calendar coverage and GDP-selection sensitivity on the supplied roster."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from audit_analysis import FOODS, OUTCOMES, association, add_fdr


def run(annual, output):
    output.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(annual / "annual_panel_1992_2017.csv").query(
        "1992 <= Year <= 2014"
    )
    assert not panel.duplicated(["Code", "Year"]).any()
    assert panel.groupby("Code").Year.nunique().eq(23).all()
    old = (
        pd.read_csv(annual / "pair_specific_eligibility.csv")
        .set_index(["exposure", "outcome", "Code"])
        .sort_index()
    )
    benchmark = (
        pd.read_csv(annual / "reconstructed_cross_sectional.csv")
        .query('specification=="updated_common_window_logGDP_energy"')
        .set_index(["exposure", "outcome"])
    )
    estimates, roster = [], []
    for food in FOODS:
        for outcome in OUTCOMES:
            pair = (food, outcome)
            variables = [food, outcome, "Total_energy"]
            obs = panel.dropna(subset=variables)
            complete = obs.dropna(subset=["GDP"])
            a_counts = obs.groupby("Code").size()
            b_counts = complete.groupby("Code").size()
            all_means = obs.groupby("Code")[variables].mean()
            with_gdp = complete.groupby("Code")[variables + ["GDP"]].mean()
            with_gdp["logGDP"] = np.log(with_gdp.GDP)
            old_pair = old.loc[pair]
            strict = set(b_counts.index[b_counts.eq(23)])
            assert strict == set(old_pair.index[old_pair.eligible])
            expanded = set(b_counts.index[b_counts.ge(22)])
            for code in old_pair.index:
                years = complete.loc[complete.Code.eq(code), "Year"]
                roster.append(
                    dict(
                        exposure=food,
                        outcome=outcome,
                        Code=code,
                        energy_complete_years=int(a_counts.get(code, 0)),
                        jointly_complete_years=len(years),
                        legacy_eligible=bool(old_pair.loc[code, "eligible"]),
                        complete_23=code in strict,
                        complete_22plus=code in expanded,
                        missing_joint_years=" ".join(
                            str(y) for y in sorted(set(range(1992, 2015)) - set(years))
                        ),
                    )
                )
            families = [
                ("energy_all23", all_means.loc[a_counts.eq(23)], ["Total_energy"]),
                ("energy_gdp23", with_gdp.loc[sorted(strict)], ["Total_energy"]),
                (
                    "energy_logGDP23",
                    with_gdp.loc[sorted(strict)],
                    ["Total_energy", "logGDP"],
                ),
                (
                    "energy_logGDP22plus",
                    with_gdp.loc[sorted(expanded)],
                    ["Total_energy", "logGDP"],
                ),
            ]
            for family, frame, controls in families:
                result = association(frame, food, outcome, controls)
                # Independent check through the t statistic for the food slope.
                X = np.column_stack([np.ones(len(frame)), frame[controls], frame[food]])
                Q, R = np.linalg.qr(X)
                y = frame[outcome].to_numpy()
                b = np.linalg.solve(R, Q.T @ y)
                df = len(frame) - len(b)
                se = (
                    np.linalg.norm(y - X @ b)
                    / np.sqrt(df)
                    * np.linalg.norm(np.linalg.inv(R)[-1])
                )
                t = b[-1] / se
                np.testing.assert_allclose(
                    [result["r"], result["p"]],
                    [t / np.sqrt(df + t * t), 2 * stats.t.sf(abs(t), df)],
                    atol=1e-10,
                )
                estimates.append(
                    dict(exposure=food, outcome=outcome, family=family, **result)
                )
    results = add_fdr(pd.DataFrame(estimates), ["family"])
    reproduced = (
        results.query('family=="energy_logGDP23"')
        .set_index(["exposure", "outcome"])
        .sort_index()
    )
    keys = ["r", "n", "ci_low", "ci_high", "p", "q_BH", "q_BY"]
    np.testing.assert_allclose(
        reproduced[keys], benchmark.sort_index()[keys], rtol=1e-8, atol=1e-11
    )
    results.to_csv(output / "coverage_estimates.csv", index=False)
    pd.DataFrame(roster).to_csv(output / "coverage_roster.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annual-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.annual_results, args.output)
