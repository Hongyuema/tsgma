"""Primary TSGMA grading on the full common outcome period, 1990–2014."""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from reconstruct_annual import DIRECT, HEALTH, RED, indicator
from audit_analysis import FOODS, OUTCOMES
from third_star import fit, grades


def panel(root, codes):
    src = root / "annual_sources"
    raw = pd.read_csv(src / "dietary.csv")
    keys = ["Code", "Year"]
    raw = raw.loc[raw.Code.isin(codes) & raw.Year.between(1980, 2014)].set_index(keys)
    frame = raw[list(DIRECT.values())].rename(columns={v: k for k, v in DIRECT.items()})
    frame["Red meat"] = raw[RED].sum(axis=1, min_count=3)
    for name, col in [("cereals", "Cereals"), ("total", "Total_energy")]:
        frame = frame.join(
            indicator(src, name, col).set_index(keys), validate="one_to_one"
        )
    for col, name in HEALTH.items():
        v = pd.read_csv(src / (name + ".csv")).dropna(subset=["Code"])
        frame = frame.join(
            v.set_index(keys)[v.columns[3]].rename(col), validate="one_to_one"
        )
    v = pd.read_csv(src / "gdp.csv").dropna(subset=["Code"]).set_index(keys)
    frame = frame.join(v["GDP per capita"].rename("GDP"), validate="one_to_one")
    return frame.reindex(
        pd.MultiIndex.from_product([codes, np.arange(1980, 2015)], names=keys)
    )


def grade(data, candidates, food, outcome, history_start, outcome_start, length, full):
    def values(col):
        return data[col].to_numpy().reshape(len(candidates), 35)

    x = values(food) / 100
    y = values(outcome)
    g = values("GDP")
    e = values("Total_energy") / 100
    inputs = [("food", x), ("GDP", g), ("energy", e), ("outcome", y)]
    missing = {
        k: np.any(
            ~np.isfinite(v[:, (1990 if k == "outcome" else history_start) - 1980 :]),
            axis=1,
        )
        for k, v in inputs
    }
    ok = ~np.logical_or.reduce(list(missing.values()))
    codes = np.asarray(candidates)[ok]
    roster = [
        dict(
            exposure=food,
            outcome=outcome,
            Code=c,
            eligible=bool(ok[i]),
            missing_series="|".join(k for k in missing if missing[k][i]),
        )
        for i, c in enumerate(candidates)
    ]
    x, y, g, e = (v[ok] for v in (x, y, g, e))
    means = [v[:, 10:].mean(1) for v in (x, y, g, e)]
    mx, my, mg, me = means
    crude = stats.pearsonr(mx, my)
    z = np.column_stack([np.ones(len(codes)), np.log(mg), me])
    remove = lambda v: v - z @ np.linalg.lstsq(z, v, rcond=None)[0]
    r = stats.pearsonr(remove(mx), remove(my)).statistic
    p = 2 * stats.t.sf(abs(r) * np.sqrt((len(codes) - 4) / (1 - r * r)), len(codes) - 4)
    t = np.arange(outcome_start - 1980, 35)

    def past(v):
        return [v[:, t - j] for j in range(1, length + 1)]

    lg = np.log(g)
    h = past(x)
    controls = [
        x[:, t],
        lg[:, t],
        e[:, t],
        np.mean(past(lg), axis=0),
        np.mean(past(e), axis=0),
    ]
    terms = [np.mean(h, axis=0), *controls]
    primary = fit(y[:, t], terms, np.eye(6)[0])[0]
    row = dict(
        exposure=food,
        outcome=outcome,
        n=len(codes),
        country_years=len(codes) * len(t),
        crude_r=crude.statistic,
        crude_p=crude.pvalue,
        adjusted_r=r,
        adjusted_p=p,
        **primary
    )
    profiles = []
    if full:
        plain = fit(y[:, t], terms, np.eye(6)[0], False)[0]
        free, b, cov = fit(y[:, t], h + controls, np.r_[np.ones(length), np.zeros(5)])
        row.update(
            without_trends=plain["beta"],
            without_trends_p=plain["p"],
            unrestricted_sum=free["beta"],
            unrestricted_lo=free["lo"],
            unrestricted_hi=free["hi"],
            unrestricted_p=free["p"],
        )
        for j in range(length):
            lim = np.sqrt(cov[j, j]) * stats.t.ppf(0.975, len(codes) - 1)
            profiles.append(
                dict(
                    exposure=food,
                    outcome=outcome,
                    lag=j + 1,
                    beta=b[j],
                    lo=b[j] - lim,
                    hi=b[j] + lim,
                )
            )
    return row, roster, profiles


def run(root):
    dest = root / "grading_1990_2014"
    dest.mkdir(exist_ok=True)
    snap = pd.read_csv(root / "results/analysis_snapshot.csv")
    candidates = sorted(snap.loc[snap.GDP.notna(), "Code"])
    assert len(candidates) == 143
    data = panel(root, candidates)
    overlap = pd.read_csv(root / "annual_results/annual_panel_1992_2017.csv").set_index(
        ["Code", "Year"]
    )
    match = data.loc[(slice(None), slice(1992, 2014)), :]
    np.testing.assert_allclose(
        match, overlap.loc[match.index, match.columns], equal_nan=True, rtol=1e-10
    )
    data.loc[(slice(None), slice(1985, 2014)), :].to_csv(dest / "panel_1985_2014.csv")
    members = []
    profiles = []
    comparisons = []
    for name, hstart, start, k in [
        ("primary", 1985, 1990, 5),
        ("shorter_observation", 1985, 1997, 5),
        ("matched_3y", 1980, 1990, 3),
        ("matched_5y", 1980, 1990, 5),
        ("matched_10y", 1980, 1990, 10),
    ]:
        rows = []
        for out in OUTCOMES:
            for food in FOODS:
                row, ro, pr = grade(
                    data, candidates, food, out, hstart, start, k, name == "primary"
                )
                rows.append(row)
                if name == "primary":
                    members.extend(ro)
                    profiles.extend(pr)
        df = pd.DataFrame(rows)
        assert len(df) == 48
        for col, v in zip(
            ["crude_q", "adjusted_q", "directional_p", "joint_Holm", "stars"],
            grades(df.crude_p, df.adjusted_p, df.adjusted_r, df.p, df.beta, df.crude_r),
        ):
            df[col] = v
        df["lag_q"] = stats.false_discovery_control(df.p)
        if name == "primary":
            df.to_csv(dest / "third_star_grades.csv", index=False)
        else:
            comparisons.append(
                df.assign(
                    comparison=name,
                    history_start=hstart,
                    first_outcome=start,
                    last_outcome=2014,
                    lag_window=k,
                )
            )
    pd.DataFrame(members).to_csv(dest / "pair_specific_eligibility.csv", index=False)
    pd.DataFrame(profiles).to_csv(dest / "unrestricted_lag_profile.csv", index=False)
    pd.concat(comparisons, ignore_index=True).to_csv(
        dest / "timing_sensitivity.csv", index=False
    )


if __name__ == "__main__":
    run(Path.cwd())
