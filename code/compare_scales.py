"""Join previously fitted country-mean and panel estimates on the same units."""

from pathlib import Path
import argparse
import pandas as pd


def compare(root):
    files = [
        (
            "partial",
            "annual_results/reconstructed_cross_sectional.csv",
            "updated_common_window_logGDP_energy",
            "r",
            "n",
        ),
        (
            "between",
            "diagnostics/common_window_diagnostics.csv",
            "HC3_logGDP_energy",
            "beta",
            "n",
        ),
        (
            "within",
            "annual_results/annual_models.csv",
            "TWFE_logGDP_energy",
            "beta",
            "n_countries",
        ),
        (
            "difference",
            "annual_results/annual_models.csv",
            "First_difference_logGDP_energy",
            "beta",
            "n_countries",
        ),
        (
            "trend",
            "annual_results/annual_models.csv",
            "TWFE_country_trends_logGDP_energy",
            "beta",
            "n_countries",
        ),
    ]
    key = ["exposure", "outcome"]
    out = None
    for label, path, spec, estimate, count in files:
        data = pd.read_csv(root / path)
        sel = data.loc[
            data.specification.eq(spec),
            key + [count, estimate, "ci_low", "ci_high", "q_BH"],
        ].copy()
        assert (
            len(sel) == 48 and not sel.duplicated(key).any() and sel.notna().all().all()
        )
        sel = sel.rename(
            columns={
                count: "n",
                estimate: label,
                **{k: label + "_" + k for k in ["ci_low", "ci_high", "q_BH"]},
            }
        )
        out = (
            sel
            if out is None
            else out.merge(sel, on=key + ["n"], validate="one_to_one")
        )
        assert len(out) == 48
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.root).to_csv(args.output, index=False)
