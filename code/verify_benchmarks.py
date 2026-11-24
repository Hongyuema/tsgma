"""Validate the reported panel simulation and region-deletion comparisons independently."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from statsmodels.stats.multitest import multipletests
from reconstruct_annual import panel_model
from panel_benchmark import SEED, G, T, LINK, draw, series, fit


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=3e-8, atol=2e-10)


def check_sim(out):
    counts = pd.read_csv(out / "replicate_counts.csv")
    summary = pd.read_csv(out / "summary.csv")
    rng = np.random.default_rng(SEED)
    shared = draw(rng)
    for shape in ("linear", "linear_quadratic"):
        x, y, z, e = series(shared, shape)
        for links in [0, 3]:
            target = y + LINK * x[:, :, :3] if links else y
            for trend in [False, True]:
                beta, se, p, q = fit(x, target, z, e, trend)
                independent = []
                for j in range(16):
                    for k in range(3):
                        data = pd.DataFrame(
                            {
                                "Code": np.repeat(np.arange(G), T),
                                "Year": np.tile(np.arange(T), G),
                                "food": 100 * x[:, :, j].ravel(),
                                "y": target[:, :, k].ravel(),
                                "GDP": np.exp(z.ravel()),
                                "Total_energy": 100 * e.ravel(),
                            }
                        )
                        f = panel_model(data, "food", "y", True, country_trends=trend)
                        close(
                            [f["beta"], f["se"], f["p"]],
                            [beta[j, k], se[j, k], p[j, k]],
                        )
                        independent.append(f["p"])
                close(multipletests(independent, method="fdr_bh")[1], q.ravel())
                sub = counts[
                    (counts.trajectory == shape)
                    & (counts.direct_links == links)
                    & (counts.country_trends == trend)
                ]
                assert len(sub) == 1000 and sub.replicate.nunique() == 1000
                first = sub[sub.replicate == 1].iloc[0]
                sel = q < 0.05
                assert first.selected == sel.sum() and first.selected_direct == (
                    np.trace(sel[:3]) if links else 0
                )
                row = summary[
                    (summary.trajectory == shape)
                    & (summary.direct_links == links)
                    & (summary.country_trends == trend)
                ].iloc[0]
                metrics = {
                    "any_selected": sub.selected.gt(0),
                    "null_selection_rate": sub.selected_null / (48 - links),
                    "direct_link_fdp": sub.selected_null / sub.selected.clip(lower=1),
                }
                if links:
                    metrics["direct_link_power"] = sub.selected_direct / links
                for name, v in metrics.items():
                    close([v.mean(), v.sem()], [row[name], row["mcse_" + name]])


def check_regions(root, out):
    results = pd.read_csv(out / "estimates.csv", keep_default_na=False)
    annual = (
        pd.read_csv(root / "annual_results/annual_panel_1992_2017.csv")
        .query("Year<=2014")
        .sort_values(["Code", "Year"])
    )
    elig = pd.read_csv(root / "annual_results/pair_specific_eligibility.csv").query(
        "eligible"
    )
    region = (
        pd.read_csv(root / "corrected_results/countries_and_missingness.csv")
        .set_index("Code")
        .region.str.strip()
    )
    previous = pd.read_csv(root / "diagnostics/common_window_omissions.csv").query(
        "omission_type=='region'"
    )
    previous["omitted"] = previous.omitted.str.strip()
    comp = results[results.omitted_region != "None"].merge(
        previous,
        left_on=["exposure", "outcome", "omitted_region"],
        right_on=["exposure", "outcome", "omitted"],
        validate="one_to_one",
    )
    assert len(comp) == 336
    close(comp.between_r, comp.r)
    close(comp.n_x, comp.n_y)
    full = results[results.omitted_region == "None"].merge(
        pd.read_csv(root / "comparison/scale_comparison.csv"),
        on=["exposure", "outcome"],
        validate="one_to_one",
    )
    assert len(full) == 48
    for a, b in [
        ("between_r", "partial"),
        ("between_x", "between_y"),
        ("within_x", "within_y"),
        ("partial_q", "partial_q_BH"),
        ("between_q", "between_q_BH"),
        ("within_q", "within_q_BH"),
    ]:
        close(full[a], full[b])
    for (x, y), block in results.groupby(["exposure", "outcome"]):
        roster = elig[(elig.exposure == x) & (elig.outcome == y)].Code
        for row in block.itertuples():
            codes = roster[region.loc[roster].to_numpy() != row.omitted_region]
            frame = annual[annual.Code.isin(codes)]
            group_count = len(codes)
            nt = len(frame)
            assert group_count == row.n and nt == group_count * 23
            # Eliminate country intercepts, then fit year indicators and the three covariates together.
            terms = np.column_stack(
                (frame[x] / 100, np.log(frame.GDP), frame.Total_energy / 100, frame[y])
            ).reshape(group_count, 23, 4)
            demeaned = (terms - terms.mean(axis=1, keepdims=True)).reshape(nt, 4)
            yrs = np.eye(23)[:, 1:]
            yrs = yrs - yrs.mean(axis=0)
            design = np.column_stack((demeaned[:, :3], np.tile(yrs, (group_count, 1))))
            outcome = demeaned[:, 3]
            b = np.linalg.lstsq(design, outcome, rcond=None)[0]
            leverage = design @ np.linalg.inv(design.T @ design)[:, 0]
            influence = (
                (leverage * (outcome - design @ b)).reshape(group_count, 23).sum(1)
            )
            rank = group_count + design.shape[1]
            se = np.sqrt(
                np.sum(influence**2)
                * group_count
                / (group_count - 1)
                * (nt - 1)
                / (nt - rank)
            )
            h = t.ppf(0.975, group_count - 1) * se
            close(
                [b[0], b[0] - h, b[0] + h, 2 * t.sf(abs(b[0] / se), group_count - 1)],
                [row.within, row.within_low, row.within_high, row.within_p],
            )
    for _, block in results.groupby("omitted_region"):
        for column in ["partial", "between", "within"]:
            close(
                multipletests(block[column + "_p"], method="fdr_bh")[1],
                block[column + "_q"],
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--results-root", type=Path)
    args = ap.parse_args()
    out = args.results_root or args.root
    check_sim(out / "panel_benchmark")
    check_regions(args.root, out / "region_comparison")
    print(
        "Checked all 384 first-replicate simulation fits, all Monte Carlo summaries and all 384 geographical panel estimates."
    )
