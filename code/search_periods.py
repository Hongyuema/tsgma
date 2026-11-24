"""Exploratory time search. Run from the analysis directory after grade_extended_period.py."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

root = Path.cwd()
sys.path.insert(0, str(root / "code"))
from grade_extended_period import panel
from third_star import transform, fit, grades
from statsmodels.stats.multitest import multipletests

destination = root / "period_search"
destination.mkdir(exist_ok=True)
primary = root / "grading_1990_2014"
members = pd.read_csv(primary / "pair_specific_eligibility.csv").query("eligible")
data = panel(root, sorted(members.Code.unique()))
periods = [(a, b) for a in range(1990, 2006) for b in range(a + 9, 2015)]
lags = {
    **{f"lag_{k}": [k] for k in (1, 3, 5, 10)},
    **{f"mean_1_{k}": list(range(1, k + 1)) for k in (3, 5, 10)},
}
assert len(periods) == 136 and len(lags) == 7
records = []


def row_cor(x, y):
    x = x - x.mean()
    y = y - y.mean(0)
    return x @ y / np.sqrt((x @ x) * (y * y).sum(0))


def correlation_p(r, df):
    return 2 * stats.t.sf(abs(r) * np.sqrt(df / (1 - r * r)), df)


def holm(p):
    order = np.argsort(p)
    ans = np.empty_like(p)
    ans[order] = np.minimum(
        1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1))
    )
    return ans


for food, block in members.groupby("exposure"):
    health = sorted(block.outcome.unique())
    lists = [sorted(block[block.outcome == h].Code) for h in health]
    assert all(codes == lists[0] for codes in lists)
    codes = lists[0]
    g = len(codes)
    d = data.loc[pd.MultiIndex.from_product([codes, range(1980, 2015)])]

    def col(k):
        return d[k].to_numpy().reshape(g, 35)

    x = col(food) / 100
    gdp = col("GDP")
    energy = col("Total_energy") / 100
    assert all(np.isfinite(v).all() for v in (x, gdp, energy)) and np.all(gdp > 0)
    lg = np.log(gdp)
    y = np.stack([col(h) for h in health], axis=-1)
    assert np.isfinite(y[:, 10:]).all()
    yn = y[:, 10:]
    previous = {}
    for label, ks in lags.items():
        hist = lambda v: np.mean([v[:, 10 - j : 35 - j] for j in ks], axis=0)
        previous[label] = np.stack(
            [hist(x), x[:, 10:], lg[:, 10:], energy[:, 10:], hist(lg), hist(energy)],
            axis=-1,
        )
    crit = stats.t.ppf(0.975, g - 1)
    for start, end in periods:
        sel = slice(start - 1990, end - 1989)
        full = slice(start - 1980, end - 1979)
        t = end - start + 1
        n = g * t
        mx = x[:, full].mean(1)
        my = yn[:, sel].mean(1)
        z = np.column_stack(
            [np.ones(g), np.log(gdp[:, full].mean(1)), energy[:, full].mean(1)]
        )
        q, _ = np.linalg.qr(z)
        crude = row_cor(mx, my)
        adj = row_cor(mx - q @ (q.T @ mx), my - q @ (q.T @ my))
        pc = correlation_p(crude, g - 2)
        pa = correlation_p(adj, g - 4)
        v = transform(yn[:, sel]).reshape(n, 3)
        for label, design in previous.items():
            a = transform(design[:, sel]).reshape(n, 6)
            q, r = np.linalg.qr(a, mode="reduced")
            assert np.linalg.matrix_rank(r) == 6
            b = np.linalg.solve(r, q.T @ v)
            err = v - a @ b
            influence = q @ np.linalg.solve(r.T, np.eye(6)[0])
            cluster = (influence[:, None] * err).reshape(g, t, 3).sum(1)
            rank = 2 * g + t - 2 + 6
            se = np.sqrt(
                (cluster * cluster).sum(0) * g / (g - 1) * (n - 1) / (n - rank)
            )
            p = 2 * stats.t.sf(abs(b[0] / se), g - 1)
            for j, h in enumerate(health):
                records.append(
                    (
                        start,
                        end,
                        label,
                        food,
                        h,
                        g,
                        crude[j],
                        pc[j],
                        adj[j],
                        pa[j],
                        b[0, j],
                        se[j],
                        b[0, j] - crit * se[j],
                        b[0, j] + crit * se[j],
                        p[j],
                    )
                )
                if adj[j] * b[0, j] > 0 and max(pa[j], p[j]) < 0.002:
                    test = fit(
                        yn[:, sel, j],
                        [design[:, sel, k] for k in range(6)],
                        np.eye(6)[0],
                    )[0]
                    np.testing.assert_allclose(
                        [b[0, j], se[j], p[j]],
                        [test["beta"], test["se"], test["p"]],
                        rtol=1e-8,
                        atol=1e-10,
                    )
df = pd.DataFrame(
    records,
    columns=[
        "start",
        "end",
        "lag",
        "exposure",
        "outcome",
        "n",
        "crude_r",
        "crude_p",
        "adjusted_r",
        "adjusted_p",
        "beta",
        "se",
        "lo",
        "hi",
        "p",
    ],
)
groups = ["start", "end", "lag"]
parts = []
for _, a in df.groupby(groups):
    a = a.copy()
    assert len(a) == 48
    q1 = stats.false_discovery_control(a.crude_p)
    q2 = stats.false_discovery_control(a.adjusted_p)
    joint = np.where(a.adjusted_r * a.beta > 0, np.maximum(a.adjusted_p, a.p), 1.0)
    adj = holm(joint)
    a["adjusted_q"] = q2
    a["conjunction_p"] = joint
    a["holm_48"] = adj
    a["stars"] = np.select(
        [(q2 < 0.05) & (adj < 0.05), q2 < 0.05, q1 < 0.05], [3, 2, 1], default=0
    )
    if (a.stars == 3).any():
        q1b, q2b, jb, hb, sb = grades(
            a.crude_p, a.adjusted_p, a.adjusted_r, a.p, a.beta
        )
        np.testing.assert_allclose(hb, adj)
        np.testing.assert_array_equal(sb, a.stars)
    parts.append(a)
df = pd.concat(parts, ignore_index=True)
assert len(df) == 45696
df["holm_search"] = holm(df.conjunction_p.to_numpy())
np.testing.assert_allclose(
    df.holm_search, multipletests(df.conjunction_p, method="holm")[1]
)
df["search_supported"] = (df.stars == 3) & (df.holm_search < 0.05)
ix = ["exposure", "outcome"]
base = df[(df.start == 1990) & (df.end == 2014) & (df.lag == "mean_1_5")].set_index(ix)
ref = pd.read_csv(primary / "third_star_grades.csv").set_index(ix).reindex(base.index)
for k in ["adjusted_r", "adjusted_q", "beta", "p", "lo", "hi", "stars"]:
    np.testing.assert_allclose(ref[k], base[k], rtol=1e-8, atol=1e-10)
hits = df[df.stars == 3].sort_values("conjunction_p")
counts = (
    df.assign(third=(df.stars == 3))
    .groupby(groups, as_index=False)
    .agg(
        three_stars=("third", "sum"),
        search_supported=("search_supported", "sum"),
        minimum_holm_48=("holm_48", "min"),
        minimum_search_holm=("holm_search", "min"),
    )
)
counts.to_csv(destination / "window_counts.csv", index=False)
df.to_csv(destination / "all_tests.csv", index=False)
hits.to_csv(destination / "three_star_hits.csv", index=False)
print(
    "Configurations:",
    len(counts),
    "nonzero:",
    (counts.three_stars > 0).sum(),
    "hits:",
    len(hits),
    "global:",
    df.search_supported.sum(),
    "min global P:",
    df.holm_search.min(),
)
print(hits.groupby(ix).size())
print(
    hits.head(10)[
        groups + ix + ["adjusted_r", "beta", "lo", "hi", "holm_48", "holm_search"]
    ].to_string(index=False)
)
