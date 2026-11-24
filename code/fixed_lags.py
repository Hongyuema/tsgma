"""Compare fixed exposure offsets on a common outcome window. Run from the archive root."""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control
from reconstruct_annual import panel_model


def run(root):
    annual = root / "annual_results"
    panel = pd.read_csv(annual / "annual_panel_1992_2017.csv")
    roster = pd.read_csv(annual / "pair_specific_eligibility.csv")
    assert not panel.duplicated(["Code", "Year"]).any()
    records = []
    for (food, outcome), membership in roster.groupby(
        ["exposure", "outcome"], sort=False
    ):
        codes = membership.loc[membership.eligible, "Code"]
        original = panel.loc[panel.Code.isin(codes)]
        ys = original.loc[original.Year.between(1997, 2014), ["Code", "Year", outcome]]
        for lag in (0, 1, 5):
            xs = original.loc[
                original.Year.between(1997 - lag, 2014 - lag),
                ["Code", "Year", food, "GDP", "Total_energy"],
            ].copy()
            xs["Year"] += lag
            observations = ys.merge(xs, on=["Code", "Year"], validate="one_to_one")
            assert (
                len(observations) == 18 * len(codes)
                and observations.notna().all().all()
            )
            for trends in (False, True):
                estimate = panel_model(
                    observations, food, outcome, energy=True, country_trends=trends
                )
                records.append(
                    dict(
                        exposure=food,
                        outcome=outcome,
                        lag_years=lag,
                        country_trends=trends,
                        **estimate
                    )
                )
    results = pd.DataFrame(records)
    assert (
        len(results) == 288
        and results.groupby(["country_trends", "lag_years"]).size().eq(48).all()
    )
    results["q_BH_48"] = results.groupby(["country_trends", "lag_years"]).p.transform(
        false_discovery_control
    )
    results["q_BH_144"] = results.groupby("country_trends").p.transform(
        false_discovery_control
    )
    dest = root / "fixed_lags"
    dest.mkdir(exist_ok=True)
    results.to_csv(dest / "estimates.csv", index=False)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    run(ap.parse_args().root)
