"""Fit a delayed association in a synthetic balanced panel."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from third_star import fit


def main():
    rng = np.random.default_rng(3107)
    countries, years, lag = 40, 30, 5
    exposure = rng.normal(size=(countries, years + lag))
    current = exposure[:, lag:]
    delayed = np.mean(
        [exposure[:, lag - j : years + lag - j] for j in range(1, lag + 1)], axis=0
    )
    outcome = (
        0.6 * delayed
        + 0.2 * current
        + rng.normal(size=(countries, 1))
        + rng.normal(size=(1, years))
        + rng.normal(scale=0.03, size=(countries, 1)) * np.arange(years)
        + rng.normal(scale=0.5, size=(countries, years))
    )
    result, _, _ = fit(outcome, [delayed, current], np.array([1.0, 0.0]))
    print("Synthetic delayed coefficient (generating value: 0.600)")
    print(
        f"Estimate: {result['beta']:.3f}; 95% CI: {result['lo']:.3f} to {result['hi']:.3f}"
    )
    print(
        "This example demonstrates estimation, not study results or a 48-pair star screen."
    )


if __name__ == "__main__":
    main()
