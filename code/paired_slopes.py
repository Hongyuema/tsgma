"""Paired country-deletion jackknife for the within-minus-between slope difference."""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t, false_discovery_control


def centered_slopes(residualized):
    """Refit centered OLS with every possible single-country omission in a balanced panel.

    Rows within a country have already been centered over time if country effects apply.
    Centering each year over the remaining countries updates its year intercept after deletion.
    For country means the same calculation, with one row per country, updates the intercept.
    """
    countries = len(residualized)
    centered = residualized - residualized.mean(axis=0)
    products = np.einsum("gti,gtj->gij", centered, centered)
    totals = products.sum(axis=0)
    deleted = totals - products * countries / (countries - 1)
    # Fields are food, log GDP, energy and outcome, in that order.
    full = np.linalg.solve(totals[:3, :3], totals[:3, 3])[0]
    fits = np.linalg.solve(deleted[:, :3, :3], deleted[:, :3, 3, None])[:, 0, 0]
    return full, fits


def run(root):
    annual = root / "annual_results"
    panel = pd.read_csv(annual / "annual_panel_1992_2017.csv").set_index(
        ["Code", "Year"]
    )
    roster = pd.read_csv(annual / "pair_specific_eligibility.csv").query("eligible")
    regions = (
        pd.read_csv(root / "corrected_results/countries_and_missingness.csv")
        .set_index("Code")
        .region.str.strip()
    )
    records, checks = [], []
    for (food, outcome), pair in roster.groupby(["exposure", "outcome"], sort=False):
        codes = pair.Code.sort_values().to_numpy()
        index = pd.MultiIndex.from_product([codes, range(1992, 2015)])
        values = (
            panel.loc[index, [food, "GDP", "Total_energy", outcome]]
            .to_numpy()
            .reshape(len(codes), 23, 4)
        )
        if not np.isfinite(values).all():
            raise ValueError(f"Incomplete panel for {food}, {outcome}")
        values[:, :, [0, 2]] /= 100
        means = values.mean(axis=1, keepdims=True)
        means[:, :, 1] = np.log(means[:, :, 1])
        values[:, :, 1] = np.log(values[:, :, 1])
        deviations = values - values.mean(axis=1, keepdims=True)
        assignment = regions.loc[codes].to_numpy()
        for region in ["Full sample", *sorted(set(assignment))]:
            keep = assignment != region
            n = int(keep.sum())
            between, b_deleted = centered_slopes(means[keep])
            within, w_deleted = centered_slopes(deviations[keep])
            delta = within - between
            deleted = w_deleted - b_deleted
            se = np.sqrt((n - 1) / n * np.square(deleted - deleted.mean()).sum())
            margin = t.ppf(0.975, n - 1) * se
            records.append(
                dict(
                    omitted_region=region,
                    exposure=food,
                    outcome=outcome,
                    n=n,
                    between=between,
                    within=within,
                    within_minus_between=delta,
                    jackknife_se=se,
                    ci_low=delta - margin,
                    ci_high=delta + margin,
                    df=n - 1,
                    p=2 * t.sf(abs(delta / se), n - 1),
                )
            )
            if region == "Full sample":
                checks.extend(
                    dict(
                        exposure=food,
                        outcome=outcome,
                        omitted_code=code,
                        between=b,
                        within=v,
                        within_minus_between=v - b,
                    )
                    for code, b, v in zip(codes, b_deleted, w_deleted, strict=True)
                )
    table = pd.DataFrame(records)
    if len(table) != 384 or not table.groupby("omitted_region").size().eq(48).all():
        raise ValueError("Expected 48 contrasts per geographic sample.")
    table["q_BH"] = table.groupby("omitted_region").p.transform(false_discovery_control)
    prior = pd.read_csv(root / "region_comparison/estimates.csv", keep_default_na=False)
    prior.omitted_region = prior.omitted_region.replace({"None": "Full sample"})
    key = ["omitted_region", "exposure", "outcome", "n"]
    old = prior.set_index(key).sort_index()
    new = table.set_index(key).sort_index()
    pd.testing.assert_index_equal(new.index, old.index)
    np.testing.assert_allclose(
        new[["between", "within"]], old[["between", "within"]], rtol=1e-8, atol=1e-10
    )
    table.to_csv(root / "comparison/paired_slope_contrasts.csv", index=False)
    pd.DataFrame(checks).to_csv(
        root / "comparison/paired_slope_country_omissions.csv", index=False
    )


if __name__ == "__main__":
    run(Path("."))
