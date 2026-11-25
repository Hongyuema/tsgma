"""Quantitative TSGMA grades on the harmonized national panel."""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def transform(a, trends=True):
    """Remove country effects, year effects and (optionally) country linear trends."""
    d = (
        a
        - a.mean(axis=1, keepdims=True)
        - a.mean(axis=0, keepdims=True)
        + a.mean(axis=(0, 1), keepdims=True)
    )
    if trends:
        time = np.arange(a.shape[1], dtype=float)
        time -= time.mean()
        slope = np.einsum("gtj,t->gj", d, time) / (time @ time)
        d = d - slope[:, None, :] * time[None, :, None]
    return d


def fit(y, xs, test, trends=True):
    g, t = y.shape
    x = transform(np.stack(xs, axis=-1), trends).reshape(g * t, -1)
    v = transform(y[:, :, None], trends).ravel()
    assert np.linalg.matrix_rank(x) == x.shape[1]
    beta = np.linalg.lstsq(x, v, rcond=None)[0]
    bread = np.linalg.inv(x.T @ x)
    e = (v - x @ beta).reshape(g, t)
    scores = np.einsum("gtj,gt->gj", x.reshape(g, t, -1), e)
    rank = g + t - 1 + (g - 1 if trends else 0) + len(xs)
    cr = g / (g - 1) * (g * t - 1) / (g * t - rank)
    cov = cr * bread @ scores.T @ scores @ bread
    b = float(test @ beta)
    se = float(np.sqrt(test @ cov @ test))
    ci = stats.t.ppf(0.975, g - 1) * se
    # Independent scalar partial-regression check when the contrast selects one term.
    if np.count_nonzero(test) == 1:
        j = int(np.argmax(test))
        z = np.delete(x, j, axis=1)
        h = x[:, j] - z @ np.linalg.lstsq(z, x[:, j], rcond=None)[0]
        independent = h @ v / (h @ h)
        seh = np.sqrt(cr * np.sum(np.sum(h.reshape(g, t) * e, axis=1) ** 2)) / (h @ h)
        np.testing.assert_allclose([b, se], [independent, seh], rtol=1e-8, atol=1e-11)
    return (
        dict(beta=b, se=se, lo=b - ci, hi=b + ci, p=2 * stats.t.sf(abs(b / se), g - 1)),
        beta,
        cov,
    )


def directional_p(sign1, p1, sign2, p2):
    """Two possible common signs; intersection-union test within each sign."""
    return np.where(np.sign(sign1) == np.sign(sign2), np.maximum(p1, p2), 1.0)


def grades(pcrude, padj, adjusted, plag, delayed, crude):
    """Cumulative grades; multiplicity families contain every eligible pair.

    Stage-specific BH screens retain all pairs. Promotion requires matching
    nonzero signs. The third-star conjunction includes all three tests, with
    Holm adjustment across the complete family before any screening.
    """
    q1 = stats.false_discovery_control(pcrude)
    q2 = stats.false_discovery_control(padj)
    same_between = np.asarray(crude) * np.asarray(adjusted) > 0
    same_all = same_between & (np.asarray(adjusted) * np.asarray(delayed) > 0)
    pjoint = np.where(same_all, np.maximum.reduce([pcrude, padj, plag]), 1.0)
    holm = multipletests(pjoint, method="holm")[1]
    first = q1 < 0.05
    second = first & (q2 < 0.05) & same_between
    stars = np.select([second & (holm < 0.05), second, first], [3, 2, 1], default=0)
    return q1, q2, pjoint, holm, stars


def run(root):
    a = root / "annual_results"
    panel = pd.read_csv(a / "annual_panel_1992_2017.csv")
    roster = pd.read_csv(a / "pair_specific_eligibility.csv")
    cross = (
        pd.read_csv(a / "reconstructed_cross_sectional.csv")
        .query("specification=='updated_common_window_logGDP_energy'")
        .set_index(["exposure", "outcome"])
    )
    rows = []
    profiles = []
    for (food, out), group in roster.groupby(["exposure", "outcome"], sort=False):
        codes = sorted(group.loc[group.eligible, "Code"])
        data = panel.loc[
            panel.Code.isin(codes) & panel.Year.between(1992, 2014)
        ].sort_values(["Code", "Year"])
        assert (
            len(data) == len(codes) * 23 and not data.duplicated(["Code", "Year"]).any()
        )

        def column(k):
            v = data[k].to_numpy().reshape(-1, 23)
            assert np.isfinite(v).all()
            return v

        def history(v):
            return [v[:, 5 - j : 23 - j] for j in range(1, 6)]

        x = column(food) / 100
        y = column(out)
        gdp = np.log(column("GDP"))
        energy = column("Total_energy") / 100
        h = history(x)
        controls = [
            x[:, 5:],
            gdp[:, 5:],
            energy[:, 5:],
            np.mean(history(gdp), axis=0),
            np.mean(history(energy), axis=0),
        ]
        vec = np.array([1.0, 0, 0, 0, 0, 0])
        primary, _, _ = fit(y[:, 5:], [np.mean(h, axis=0), *controls], vec)
        plain, _, _ = fit(y[:, 5:], [np.mean(h, axis=0), *controls], vec, False)
        free, b, cov = fit(y[:, 5:], h + controls, np.r_[np.ones(5), np.zeros(5)])
        for j in range(5):
            se = np.sqrt(cov[j, j])
            crit = stats.t.ppf(0.975, len(codes) - 1)
            profiles.append(
                dict(
                    exposure=food,
                    outcome=out,
                    lag=j + 1,
                    beta=b[j],
                    lo=b[j] - crit * se,
                    hi=b[j] + crit * se,
                )
            )
        crude = stats.pearsonr(x.mean(axis=1), y.mean(axis=1))
        ref = cross.loc[food, out]
        assert int(ref.n) == len(codes)
        rows.append(
            dict(
                exposure=food,
                outcome=out,
                n=len(codes),
                country_years=len(codes) * 18,
                crude_r=crude.statistic,
                crude_p=crude.pvalue,
                adjusted_r=ref.r,
                adjusted_p=ref.p,
                **primary,
                without_trends=plain["beta"],
                without_trends_p=plain["p"],
                unrestricted_sum=free["beta"],
                unrestricted_lo=free["lo"],
                unrestricted_hi=free["hi"],
                unrestricted_p=free["p"]
            )
        )
    frame = pd.DataFrame(rows)
    assert len(frame) == 48
    for col, vals in zip(
        ["crude_q", "adjusted_q", "directional_p", "joint_Holm", "stars"],
        grades(frame.crude_p, frame.adjusted_p, frame.adjusted_r, frame.p, frame.beta, frame.crude_r),
    ):
        frame[col] = vals
    frame["lag_q"] = stats.false_discovery_control(frame.p)
    dest = root / "fixed_lags"
    frame.to_csv(dest / "third_star_grades.csv", index=False)
    pd.DataFrame(profiles).to_csv(dest / "unrestricted_lag_profile.csv", index=False)


if __name__ == "__main__":
    run(Path.cwd())
