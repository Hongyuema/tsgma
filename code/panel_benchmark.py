"""Synthetic benchmark for annual screening, paired across scenarios within replicate."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t, false_discovery_control

SEED = 20261120
G, T = 143, 23
TIME = np.linspace(-1, 1, T)
QUADRATIC = TIME**2 - np.mean(TIME**2)
QUADRATIC *= np.linalg.norm(TIME) / np.linalg.norm(QUADRATIC)
LINK = 0.10


def draw(rng):
    level, slope, curve = rng.normal(size=(3, G, 1))
    linear = level + (1 + slope) * TIME
    # AR(1) series start with their stationary N(0,1) distribution.
    errors = rng.normal(size=(G, T, 19))
    for i in range(1, T):
        errors[:, i, :] = (
            0.65 * errors[:, i - 1, :] + np.sqrt(1 - 0.65**2) * errors[:, i, :]
        )
    return (
        linear,
        0.6 * curve * QUADRATIC,
        errors,
        rng.normal(size=(G, T)),
        rng.normal(size=(G, T)),
    )


def series(draws, shape):
    linear, curve, errors, measurement_error, energy = draws
    driver = linear if shape == "linear" else linear + curve
    common = 0.7 * driver + 0.25 * energy
    return (
        common[:, :, None] + errors[:, :, :16],
        common[:, :, None] + errors[:, :, 16:],
        driver + measurement_error,
        energy,
    )


def transform(v, trends):
    z = (
        v
        - v.mean(axis=1, keepdims=True)
        - v.mean(axis=0, keepdims=True)
        + v.mean(axis=(0, 1))
    )
    if trends:
        slopes = np.einsum("gtv,t->gv", z, TIME) / (TIME @ TIME)
        z -= slopes[:, None, :] * TIME[None, :, None]
    return z


def fit(x, y, proxy, energy, trends):
    # All 48 single-exposure models have the same nuisance covariates.
    all_z = transform(np.dstack((x, y, proxy, energy)), trends).reshape(G * T, -1)
    nuisance, target = all_z[:, -2:], all_z[:, :-2]
    target = target - nuisance @ np.linalg.lstsq(nuisance, target, rcond=None)[0]
    x = target[:, :16].reshape(G, T, 16)
    y = target[:, 16:].reshape(G, T, 3)
    xx = np.einsum("gtf,gtf->gf", x, x)
    xy = x.transpose(0, 2, 1) @ y
    total = xx.sum(axis=0)[:, None]
    beta = xy.sum(axis=0) / total
    scores = xy - xx[:, :, None] * beta
    rank = G + T - 1 + 3 + (G - 1 if trends else 0)
    correction = (G / (G - 1)) * (G * T - 1) / (G * T - rank)
    se = np.sqrt(correction * np.sum(scores**2, axis=0)) / total
    p = 2 * t.sf(np.abs(beta / se), G - 1)
    q = false_discovery_control(p, axis=None).reshape(16, 3)
    return beta, se, p, q


def run(out, replicates):
    rng = np.random.default_rng(SEED)
    records = []
    for rep in range(replicates):
        draws = draw(rng)
        for shape in ["linear", "linear_quadratic"]:
            x, y, proxy, energy = series(draws, shape)
            for n_links in [0, 3]:
                target = y.copy()
                if n_links:
                    target += LINK * x[:, :, :3]
                for trends in [False, True]:
                    q = fit(x, target, proxy, energy, trends)[3]
                    selected = q < 0.05
                    n = int(selected.sum())
                    true = int(np.trace(selected[:3])) if n_links else 0
                    records.append((rep + 1, shape, n_links, trends, n, true, n - true))
    out.mkdir(parents=True, exist_ok=True)
    counts = pd.DataFrame(
        records,
        columns=[
            "replicate",
            "trajectory",
            "direct_links",
            "country_trends",
            "selected",
            "selected_direct",
            "selected_null",
        ],
    )
    counts.to_csv(out / "replicate_counts.csv", index=False)
    counts["any_selected"] = counts.selected.gt(0).astype(float)
    counts["null_selection_rate"] = counts.selected_null / (48 - counts.direct_links)
    counts["direct_link_fdp"] = counts.selected_null / counts.selected.clip(lower=1)
    counts["direct_link_power"] = counts.selected_direct / counts.direct_links.replace(
        0, np.nan
    )
    metrics = [
        "any_selected",
        "null_selection_rate",
        "direct_link_fdp",
        "direct_link_power",
    ]
    group = counts.groupby(["trajectory", "direct_links", "country_trends"])[metrics]
    report = group.mean().join(group.sem().add_prefix("mcse_"))
    report["replicates"] = replicates
    report.to_csv(out / "summary.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("panel_benchmark"))
    ap.add_argument("--replicates", type=int, default=1000)
    args = ap.parse_args()
    run(args.output, args.replicates)
