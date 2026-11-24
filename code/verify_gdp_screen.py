"""Verify each crude and GDP-adjusted screen by residual least squares and SciPy BH."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, t
from audit_analysis import FOODS, OUTCOMES


def verify(root, results):
    snap = pd.read_csv(root / "corrected_results/analysis_snapshot.csv").set_index(
        "Code"
    )
    dat = snap[[*FOODS, *OUTCOMES, "GDP"]].dropna()
    outputs = pd.read_csv(results / "screen_country_omissions.csv")
    summary = pd.read_csv(results / "screen_transitions.csv")
    pairs = pd.MultiIndex.from_product([FOODS, OUTCOMES], names=["exposure", "outcome"])
    assert len(dat) == 143 and len(outputs) == 143 * 48 and len(summary) == 48
    for code in [None, *dat.index]:
        rows = dat if code is None else dat.drop(code)
        expected = (
            (summary if code is None else outputs[outputs.omitted_code.eq(code)])
            .set_index(["exposure", "outcome"])
            .loc[pairs]
        )
        for k, name in [(0, "matched"), (1, "GDP")]:
            design = np.ones((len(rows), k + 1))
            if k:
                design[:, 1] = rows.GDP
            v = rows[[*FOODS, *OUTCOMES]].values
            residual = v - design @ np.linalg.lstsq(design, v, rcond=None)[0]
            unit = residual / np.linalg.norm(residual, axis=0)
            r = (unit[:, : len(FOODS)].T @ unit[:, len(FOODS) :]).ravel()
            df = len(rows) - 2 - k
            p = 2 * t.sf(abs(r) * np.sqrt(df / (1 - r * r)), df)
            q = false_discovery_control(p)
            for field, value in [("r", r), ("p", p), ("q", q)]:
                np.testing.assert_allclose(
                    expected[name + "_" + field], value, rtol=1e-9, atol=2e-13
                )
        assert np.array_equal(
            expected.matched_gain,
            (expected.matched_q >= 0.05) & (expected.GDP_q < 0.05),
        )
    print(
        "Verified the complete 48-pair comparison and all 6,864 country-omission comparisons."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    verify(args.root, args.output or args.root / "gdp_screen")
