"""Fixed-design benchmark of the complete three-star decision in 48-pair screens."""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from third_star import transform, grades, fit


def ar(rng, shape, rho):
    v = rng.normal(size=shape)
    for j in range(1, shape[1]):
        v[:, j] = rho * v[:, j - 1] + np.sqrt(1 - rho * rho) * v[:, j]
    return v


def run(root):
    rng = np.random.default_rng(32203232)
    g, t = 90, 23
    reps = 200
    time = np.linspace(-1, 1, t)[None, :]
    t0 = t - 5
    context = rng.normal(size=(g, 1)) + 0.4 * time + ar(rng, (g, t), 0.65) * 0.3
    energy = rng.normal(size=(g, 1)) + 0.2 * time + ar(rng, (g, t), 0.55) * 0.4
    foods = [
        rng.normal(size=(g, 1)) + 0.4 * context + 0.2 * energy + ar(rng, (g, t), 0.7)
        for _ in range(16)
    ]

    def hist(x):
        return np.mean([x[:, 5 - j : t - j] for j in range(1, 6)], axis=0)

    # Use lag history before the observed window only to generate early outcomes;
    # burn-in padding has no role in fitting annual models in years 6–23.
    def lag_full(x):
        pad = np.pad(x, ((0, 0), (5, 0)), mode="edge")
        return np.mean([pad[:, 5 - j : 5 - j + t] for j in range(1, 6)], axis=0)

    between = np.column_stack(
        [np.ones(g), np.log(np.exp(context).mean(axis=1)), energy.mean(axis=1)]
    )
    designs = []
    for x in foods:
        raw = [
            hist(x),
            x[:, 5:],
            context[:, 5:],
            energy[:, 5:],
            hist(context),
            hist(energy),
        ]
        z = transform(np.stack(raw, axis=-1)).reshape(g * t0, 6)
        inv = np.linalg.inv(z.T @ z)
        pinv = inv @ z.T
        designs.append((z, pinv, pinv[0].reshape(g, t0, 1), raw))
    cr = g / (g - 1) * (g * t0 - 1) / (g * t0 - (2 * g + t0 - 2 + 6))
    noise = [
        rng.normal(size=(g, 1, reps))
        + 0.55 * context[:, :, None]
        + 0.35 * energy[:, :, None]
        + rng.normal(size=(g, 1, reps)) * time[:, :, None]
        + ar(rng, (g, t, reps), 0.7)
        for _ in range(3)
    ]
    rows = []

    def assoc(x, y, k):
        x = x - np.mean(x)
        yy = y - y.mean(axis=0)
        r = x @ yy / np.sqrt((x @ x) * np.sum(yy**2, axis=0))
        return r, 2 * stats.t.sf(abs(r) * np.sqrt((g - k - 2) / (1 - r * r)), g - k - 2)

    for scenario, now, lag in [
        ("no_food_link", 0.0, 0.0),
        ("current_only", 0.4, 0.0),
        ("current_and_delayed", 0.4, 0.4),
    ]:
        measures = []
        for j in range(3):
            x = foods[j]
            out = noise[j] + (now * x + lag * lag_full(x))[:, :, None]
            y = out.mean(axis=1)
            yc = y - between @ np.linalg.lstsq(between, y, rcond=None)[0]
            annual = transform(out[:, 5:]).reshape(g * t0, reps)
            for i, (z, pinv, h, raw) in enumerate(designs):
                xm = foods[i].mean(axis=1)
                xc = xm - between @ np.linalg.lstsq(between, xm, rcond=None)[0]
                _, p1 = assoc(xm, y, 0)
                r2, p2 = assoc(xc, yc, 2)
                beta = pinv @ annual
                residual = (annual - z @ beta).reshape(g, t0, reps)
                se = np.sqrt(cr * np.sum(np.sum(h * residual, axis=1) ** 2, axis=0))
                p3 = 2 * stats.t.sf(abs(beta[0] / se), g - 1)
                if j == 0 and i == 0:
                    reference = fit(out[:, 5:, 0], raw, np.eye(6)[0])[0]
                    np.testing.assert_allclose(
                        [beta[0, 0], se[0]], [reference["beta"], reference["se"]]
                    )
                measures.append(np.stack([p1, p2, r2, p3, beta[0]]))
        arr = np.stack(measures, axis=1)
        stars = np.array([grades(*arr[:, :, k])[-1] for k in range(reps)])
        target = np.arange(48) % 16 == np.arange(48) // 16
        for label, mask in [
            ("three_designated_pairs", target),
            ("other_45_pairs", ~target),
        ]:
            v = stars[:, mask]
            row = dict(
                scenario=scenario,
                group=label,
                replicates=reps,
                n_pairs=int(mask.sum()),
                replicates_with_any_three=int((v == 3).any(axis=1).sum()),
            )
            for grade in range(4):
                row[f"count_grade_{grade}"] = int((v == grade).sum())
            rows.append(row)
    pd.DataFrame(rows).to_csv(root / "fixed_lags/third_star_benchmark.csv", index=False)


if __name__ == "__main__":
    run(Path.cwd())
