"""Matched country and year comparison of dietary exposure releases, 1992–2014."""

from pathlib import Path
from zipfile import ZipFile
import argparse
import numpy as np
import pandas as pd
from audit_analysis import FOODS, OUTCOMES, association, add_fdr
from reconstruct_annual import DIRECT, RED, SELECTED, panel_model
from recover_outcomes import records


def run(root):
    out = root / "food_release_comparison"
    out.mkdir(exist_ok=True)
    source = root / "annual_results"
    panel = pd.read_csv(source / "annual_panel_1992_2017.csv")
    panel = (
        panel.loc[panel.Year.between(1992, 2014)]
        .set_index(["Code", "Year"])
        .sort_index()
    )
    assert panel.index.is_unique and len(panel) == 158 * 23
    lookup = (
        pd.read_csv(source / "source_item_mapping.csv")
        .set_index("csv_column")
        .source_label
    )
    items = {k: [v] for k, v in DIRECT.items()}
    items.update({"Red meat": RED, "Cereals": SELECTED})
    with ZipFile(root / "historical_sources/historical_diet.zip") as archive:
        old = pd.read_csv(
            archive.open(next(k for k in archive.namelist() if k.endswith(".csv")))
        )
    old = (
        old.loc[old.Code.notna()]
        .set_index(["Code", "Year"])
        .drop(columns="Entity")
        .reindex(panel.index)
    )
    old.columns = [int(c.split("|")[1]) for c in old]
    assert old.columns.is_unique
    wide = records(root / "historical_plot_inputs/alcohol_annual.xlsx")
    wide = wide.loc[
        wide.iloc[:, 0].isin(panel.index.get_level_values("Code"))
    ].set_index(wide.columns[0])
    project = (
        wide.loc[:, 1992:2014]
        .rename_axis(index="Code", columns="Year")
        .stack()
        .reindex(panel.index)
    )
    cross, within, rosters, discrepancy = [], [], [], []
    for food in FOODS:
        codes = [int(lookup[name].split("|")[1]) for name in items[food]]
        frame = panel[["GDP", "Total_energy", *OUTCOMES]].copy()
        frame["export_2024"] = old[codes].sum(axis=1, min_count=len(codes))
        frame["release_2026"] = panel["Selected_cereals" if food == "Cereals" else food]
        releases = ["export_2024", "release_2026"]
        if food == "Alcoholic Beverages":
            frame["original_worksheet"] = pd.to_numeric(project, errors="raise").astype(
                float
            )
            releases.append("original_worksheet")
            assert frame.original_worksheet.notna().all()
        for outcome in OUTCOMES:
            valid = (
                frame[["GDP", "Total_energy", outcome, *releases]].notna().all(axis=1)
            )
            observed = valid.groupby(level="Code").sum()
            roster = observed.index[observed.eq(23)]
            rosters.extend(
                dict(
                    exposure=food,
                    outcome=outcome,
                    Code=code,
                    years=int(n),
                    included=n == 23,
                )
                for code, n in observed.items()
            )
            sample = frame.loc[
                frame.index.get_level_values("Code").isin(roster)
            ].reset_index()
            means = sample.groupby("Code")[
                ["GDP", "Total_energy", outcome, *releases]
            ].mean()
            means["logGDP"] = np.log(means.GDP)
            for release in releases:
                pair = dict(exposure=food, outcome=outcome, release=release)
                estimate = association(
                    means, release, outcome, ["logGDP", "Total_energy"]
                )
                cross.append({**pair, **estimate})
                fit = panel_model(sample, release, outcome, energy=True)
                within.append({**pair, **fit})
                assert estimate["n"] == fit["n_countries"] == len(roster)
                assert fit["n_observations"] == 23 * len(roster)
            delta = means.release_2026 - means.export_2024
            discrepancy.append(
                dict(
                    exposure=food,
                    outcome=outcome,
                    n=len(roster),
                    item_codes=",".join(map(str, codes)),
                    median_change=delta.median(),
                    max_abs_change=delta.abs().max(),
                    largest_change_code=delta.abs().idxmax(),
                )
            )
    for rows, name in [(cross, "between_countries"), (within, "within_countries")]:
        frame = pd.DataFrame(rows)
        two = frame.release.ne("original_worksheet")
        primary = add_fdr(frame.loc[two], ["release"])
        pd.concat([primary, frame.loc[~two]], ignore_index=True).to_csv(
            out / (name + ".csv"), index=False
        )
    pd.DataFrame(rosters).to_csv(out / "eligibility.csv", index=False)
    pd.DataFrame(discrepancy).to_csv(out / "exposure_differences.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    run(parser.parse_args().root)
