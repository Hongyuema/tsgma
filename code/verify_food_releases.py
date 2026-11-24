"""Independent check of two specifications using direct OLS and explicit fixed effects."""

from pathlib import Path
from zipfile import ZipFile
import argparse
import numpy as np
import pandas as pd
from statsmodels.api import OLS
from statsmodels.stats.multitest import multipletests

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default=Path("."))
b = parser.parse_args().root
from recover_outcomes import records

p = pd.read_csv(b / "annual_results/annual_panel_1992_2017.csv").query("Year<=2014")
with ZipFile(b / "historical_sources/historical_diet.zip") as z:
    hist = pd.read_csv(z.open(next(n for n in z.namelist() if n.endswith(".csv"))))
hist = hist[hist.Code.notna()].set_index(["Code", "Year"])
hist.columns = [
    str(c.split("|")[1]).strip().lstrip("0") if "|" in c else c for c in hist
]
roster = pd.read_csv(b / "food_release_comparison/eligibility.csv")
items = (
    pd.read_csv(b / "food_release_comparison/exposure_differences.csv")
    .drop_duplicates("exposure")
    .set_index("exposure")
    .item_codes
)
cross = pd.read_csv(b / "food_release_comparison/between_countries.csv").set_index(
    ["exposure", "outcome", "release"]
)
within = pd.read_csv(b / "food_release_comparison/within_countries.csv").set_index(
    cross.index.names
)
alcohol = records(b / "historical_plot_inputs/alcohol_annual.xlsx")
alcohol = alcohol[alcohol.iloc[:, 0].isin(p.Code)].set_index(alcohol.columns[0])
for (food, outcome, release), ex in cross.iterrows():
    take = roster[
        (roster.exposure == food) & (roster.outcome == outcome) & roster.included
    ].Code
    a = p[p.Code.isin(take)].sort_values(["Code", "Year"]).copy()
    if release == "release_2026":
        a["food_value"] = a["Selected_cereals" if food == "Cereals" else food]
    elif release == "export_2024":
        vals = hist.loc[
            pd.MultiIndex.from_frame(a[["Code", "Year"]]), items[food].split(",")
        ]
        assert vals.notna().all().all()
        a["food_value"] = vals.to_numpy().sum(axis=1)
    else:
        a["food_value"] = [alcohol.loc[c, y] for c, y in zip(a.Code, a.Year)]
    v = "food_value"
    m = a.groupby("Code")[[v, outcome, "GDP", "Total_energy"]].mean()
    X = np.column_stack([np.ones(len(m)), np.log(m.GDP), m.Total_energy, m[v]])
    cf = OLS(m[outcome].to_numpy(), X).fit()
    r = cf.tvalues[-1] / np.sqrt(cf.tvalues[-1] ** 2 + cf.df_resid)
    np.testing.assert_allclose([ex.r, ex.p], [r, cf.pvalues[-1]], atol=1e-9)
    X = np.column_stack(
        [
            np.ones(len(a)),
            a[v] / 100,
            np.log(a.GDP),
            a.Total_energy / 100,
            pd.get_dummies(a.Code, drop_first=True, dtype=float),
            pd.get_dummies(a.Year, drop_first=True, dtype=float),
        ]
    )
    fit = OLS(a[outcome].to_numpy(), X).fit(
        cov_type="cluster", cov_kwds={"groups": a.Code.to_numpy()}, use_t=True
    )
    ref = within.loc[food, outcome, release]
    np.testing.assert_allclose(
        [ref.beta, ref.se, ref.p, ref.ci_low, ref.ci_high],
        [fit.params[1], fit.bse[1], fit.pvalues[1], *fit.conf_int()[1]],
        rtol=1e-7,
        atol=1e-9,
    )
for d in (cross, within):
    for rel in ("export_2024", "release_2026"):
        group = d.xs(rel, level="release")
        assert len(group) == 48
        np.testing.assert_allclose(
            group.q_BH, multipletests(group.p, method="fdr_bh")[1], atol=1e-11
        )
print(
    "Verified 99 partial correlations, 99 fixed-effects fits and four 48-test families."
)
