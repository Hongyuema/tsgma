"""Compare common year effects with separate calendar effects for each region."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

from reconstruct_annual import panel_model


def run(root):
    annual = root / "annual_results"
    panel = pd.read_csv(annual / "annual_panel_1992_2017.csv").query(
        "1992 <= Year <= 2014"
    )
    region = (
        pd.read_csv(root / "corrected_results/countries_and_missingness.csv")
        .set_index("Code")
        .region.str.strip()
    )
    panel["region"] = panel.Code.map(region)
    eligibility = pd.read_csv(annual / "pair_specific_eligibility.csv").query(
        "eligible"
    )
    reference = (
        pd.read_csv(annual / "annual_models.csv")
        .query("specification == 'TWFE_logGDP_energy'")
        .set_index(["exposure", "outcome"])
    )
    fits = []
    for (food, outcome), eligible in eligibility.groupby(
        ["exposure", "outcome"], sort=False
    ):
        sample = panel.loc[panel.Code.isin(eligible.Code)]
        sizes = sample.groupby("region").Code.nunique()
        if sample.region.isna().any() or len(sizes) != 7 or sizes.min() < 2:
            raise ValueError("Incomplete assignment to the seven geographic regions.")
        fit = panel_model(sample, food, outcome, energy=True, regional_years=True)
        ref = reference.loc[food, outcome]
        if (
            fit["n_countries"] != ref.n_countries
            or fit["n_observations"] != ref.n_observations
        ):
            raise ValueError(
                "Regional-year fit does not reproduce the original analytical sample."
            )
        fits.append(
            dict(
                exposure=food,
                outcome=outcome,
                n_regions=len(sizes),
                min_region_size=int(sizes.min()),
                max_region_size=int(sizes.max()),
                common_year_beta=ref.beta,
                common_year_ci_low=ref.ci_low,
                common_year_ci_high=ref.ci_high,
                common_year_q=ref.q_BH,
                **fit
            )
        )
    result = pd.DataFrame(fits)
    if len(result) != 48 or result.duplicated(["exposure", "outcome"]).any():
        raise ValueError(
            "The regional-year family requires 48 distinct food–outcome pairs."
        )
    result["q_BH"] = false_discovery_control(result.p)
    result["q_BY"] = false_discovery_control(result.p, method="by")
    result.to_csv(root / "diagnostics/region_year_effects.csv", index=False)


if __name__ == "__main__":
    run(Path("."))
