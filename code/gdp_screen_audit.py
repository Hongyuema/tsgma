"""Trace GDP-screen threshold crossings on identical country rosters; no new model family is selected from the data."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from statsmodels.stats.multitest import multipletests
from audit_analysis import FOODS, OUTCOMES


def run(root, dest):
    base = root / "corrected_results"
    snapshot = pd.read_csv(base / "analysis_snapshot.csv").set_index("Code")
    reference = pd.read_csv(base / "all_associations.csv").set_index(
        ["exposure", "outcome", "specification"]
    )
    sample = snapshot[["GDP", *FOODS, *OUTCOMES]].dropna()
    assert len(sample) == 143 and sample.index.is_unique
    index = pd.MultiIndex.from_product([FOODS, OUTCOMES], names=["exposure", "outcome"])
    pieces = []

    def probabilities(r, df):
        return 2 * t.sf(abs(r) * np.sqrt(df / (1 - r * r)), df)

    for omitted in [None, *sample.index]:
        dat = sample if omitted is None else sample.drop(omitted)
        cov = dat.corr().loc[[*FOODS, *OUTCOMES, "GDP"], [*FOODS, *OUTCOMES, "GDP"]]
        rx = cov.loc[FOODS, "GDP"].to_numpy()[:, None]
        ry = cov.loc[OUTCOMES, "GDP"].to_numpy()[None, :]
        crude = cov.loc[FOODS, OUTCOMES].to_numpy()
        partial = (crude - rx * ry) / np.sqrt((1 - rx**2) * (1 - ry**2))
        result = pd.DataFrame(
            {
                "matched_r": crude.ravel(),
                "food_r_GDP": np.repeat(rx.ravel(), len(OUTCOMES)),
                "outcome_r_GDP": np.tile(ry.ravel(), len(FOODS)),
                "GDP_r": partial.ravel(),
            },
            index=index,
        )
        for tag, df in [("matched", len(dat) - 2), ("GDP", len(dat) - 3)]:
            p = probabilities(result[tag + "_r"], df)
            result[tag + "_p"] = p
            result[tag + "_q"] = multipletests(p, method="fdr_bh")[1]
        result["matched_gain"] = (result.matched_q >= 0.05) & (result.GDP_q < 0.05)
        if omitted is None:
            summary = result.copy()
        else:
            result["omitted_code"] = omitted
            pieces.append(result.reset_index())
    omits = pd.concat(pieces, ignore_index=True)
    for tag, spec in [
        ("matched", "Pearson_GDP_complete_143"),
        ("GDP", "GDP_partial_143"),
    ]:
        ref = reference.xs(spec, level="specification").loc[index]
        np.testing.assert_allclose(
            summary[[tag + "_r", tag + "_p", tag + "_q"]],
            ref[["r", "p", "q_BH"]],
            atol=2e-13,
        )
    for col, spec in [
        ("full", "Pearson_158"),
        ("rank", "Spearman_GDP_partial_143"),
        ("joint", "Joint_context_143"),
    ]:
        ref = reference.xs(spec, level="specification").loc[index]
        summary[col + "_r"] = ref.r
        summary[col + "_q"] = ref.q_BH
    summary["full_to_adjusted_gain"] = (summary.full_q >= 0.05) & (summary.GDP_q < 0.05)
    region = pd.read_csv(base / "leave_one_region_out.csv").query(
        "specification=='GDP_partial_143'"
    )
    by = ["exposure", "outcome"]
    summary["region_omissions_GDP_q_lt_05"] = region.groupby(by).q_BH.apply(
        lambda v: (v < 0.05).sum()
    )
    summary["country_omissions_GDP_q_lt_05"] = omits.groupby(by).GDP_q.apply(
        lambda v: (v < 0.05).sum()
    )
    summary["country_omissions_matched_gain"] = omits.groupby(by).matched_gain.sum()
    summary["shape_q"] = (
        pd.read_csv(base / "nonlinearity_HC3_spline_tests.csv")
        .set_index(by)
        .q_BH_48_tests
    )
    summary["restriction_change_r"] = summary.matched_r - summary.full_r
    summary["adjustment_change_r"] = summary.GDP_r - summary.matched_r
    assert summary.notna().all().all() and len(omits) == 143 * 48
    dest.mkdir(parents=True, exist_ok=True)
    summary.to_csv(dest / "screen_transitions.csv", float_format="%.16g")
    omits.to_csv(
        dest / "screen_country_omissions.csv", index=False, float_format="%.16g"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--output", type=Path)
    opts = ap.parse_args()
    run(opts.root, opts.output or opts.root / "gdp_screen")
