"""Seeded synthetic benchmark for the between-country screen. Run from archive root."""

import csv
from pathlib import Path
import numpy as np
from scipy.stats import t
from statsmodels.stats.multitest import multipletests


def run(out):
    rng = np.random.default_rng(20261119)
    summary = []
    for scenario in ("No direct links", "Three direct links"):
        metrics = {model: [] for model in ("Crude", "Adjusted")}
        truth = np.zeros((16, 3), bool)
        if scenario == "Three direct links":
            truth[:3] = np.eye(3, dtype=bool)
        for _ in range(2000):
            # Mean common trend over 23 years is 0.5. Each country has its own trend slope.
            d = rng.normal(size=143) + 0.5 * (1 + rng.normal(scale=0.4, size=143))
            x = 0.7 * d[:, None] + rng.normal(size=(143, 16))
            y = 0.7 * d[:, None] + rng.normal(size=(143, 3))
            if truth.any():
                y += 0.35 * x[:, :3]
            for model in metrics:
                a = x - x.mean(axis=0)
                b = y - y.mean(axis=0)
                df = len(d) - 2
                if model == "Adjusted":
                    c = d - d.mean()
                    a -= np.outer(c, c @ a) / (c @ c)
                    b -= np.outer(c, c @ b) / (c @ c)
                    df -= 1
                r = (a.T @ b) / np.outer(
                    np.linalg.norm(a, axis=0), np.linalg.norm(b, axis=0)
                )
                p = 2 * t.sf(abs(r) * np.sqrt(df / (1 - r * r)), df)
                selected = multipletests(p.ravel(), alpha=0.05, method="fdr_bh")[
                    0
                ].reshape(16, 3)
                false = (selected & ~truth).sum()
                metrics[model].append(
                    [
                        false / (~truth).sum(),
                        false / max(1, selected.sum()),
                        (selected & truth).sum() / max(1, truth.sum()),
                    ]
                )
        for model, a in metrics.items():
            v = np.array(a)
            means = v.mean(0)
            se = v.std(0, ddof=1) / np.sqrt(len(v))
            summary.append(
                [
                    scenario,
                    model,
                    2000,
                    *means[:2],
                    means[2] if truth.any() else "",
                    *se[:2],
                    se[2] if truth.any() else "",
                ]
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scenario",
                "model",
                "replicates",
                "null_pair_selection_rate",
                "mean_false_discovery_proportion",
                "power",
                "mcse_null_selection",
                "mcse_fdp",
                "mcse_power",
            ]
        )
        writer.writerows(summary)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("simulation/screen.csv"))
    run(parser.parse_args().output)
