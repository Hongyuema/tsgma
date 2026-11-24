"""Assess sensitivity of first-difference estimates to individual adjacent-year changes."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

from reconstruct_annual import panel_model


def run(root):
    data = root / "annual_results"
    annual = pd.read_csv(data / "annual_panel_1992_2017.csv").query(
        "1992 <= Year <= 2014"
    )
    eligible = pd.read_csv(data / "pair_specific_eligibility.csv").query("eligible")
    original = pd.read_csv(data / "annual_models.csv").query(
        "specification == 'First_difference_logGDP_energy'"
    )
    baseline = original.set_index(["exposure", "outcome"])
    records = []
    for (food, outcome), units in eligible.groupby(["exposure", "outcome"], sort=False):
        frame = annual.loc[annual.Code.isin(units.Code)].copy()
        if (
            frame.groupby("Code").Year.nunique().ne(23).any()
            or len(frame) != len(units) * 23
        ):
            raise ValueError("Expected 23 complete, consecutive years per unit.")
        ref = baseline.loc[food, outcome]
        for year in range(1993, 2015):
            stats = panel_model(
                frame,
                food,
                outcome,
                energy=True,
                first_difference=True,
                omit_difference_ending=year,
            )
            if (
                stats["n_countries"] != ref.n_countries
                or stats["n_observations"] != 21 * ref.n_countries
            ):
                raise ValueError("Unexpected sample after omission.")
            records.append(
                dict(
                    exposure=food,
                    outcome=outcome,
                    omitted_start_year=year - 1,
                    omitted_end_year=year,
                    **stats,
                    full_beta=ref.beta,
                    full_q_BH=ref.q_BH
                )
            )
    table = pd.DataFrame(records)
    if (
        len(table) != 48 * 22
        or not table.groupby("omitted_end_year").size().eq(48).all()
    ):
        raise ValueError("Incomplete year-specific testing families.")
    table["q_BH"] = table.groupby("omitted_end_year").p.transform(
        false_discovery_control
    )
    table.to_csv(root / "comparison/first_difference_year_omissions.csv", index=False)


if __name__ == "__main__":
    run(Path("."))
