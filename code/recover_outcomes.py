"""Compare archived long and wide annual source sheets to the exact submitted means."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from openpyxl import load_workbook


def records(path):
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = iter(wb.active.values)
        header = list(next(rows))
        return pd.DataFrame(rows, columns=header)
    finally:
        wb.close()


def run(root):
    archive = root / "historical_sources"
    out = root / "outcome_audit"
    out.mkdir(exist_ok=True)
    original = pd.read_csv(
        root / "results/analysis_snapshot.csv", float_precision="round_trip"
    ).set_index("Code")
    annual = pd.read_csv(root / "annual_results/annual_panel_1992_2017.csv")
    corrected = original.copy()
    rows, cells = [], []
    for short, outcome, end in [
        ("nonhdl", "Non-HDL", 2017),
        ("hypertension", "Hypertension", 2017),
        ("diabetes", "Diabetes1992-2014", 2014),
    ]:
        raw = records(archive / (short + "_long.xlsx"))
        wide = records(archive / (short + "_wide.xlsx"))
        codes = original.index
        years = list(range(1992, end + 1))
        long = (
            raw.loc[raw.Code.isin(codes)]
            .pivot(index="Code", columns="Year", values=raw.columns[-1])
            .loc[codes, years]
        )
        code_col = list(wide.columns).index("Code")
        processed = wide[years].set_axis(wide.iloc[:, code_col]).loc[codes]
        stored = wide.iloc[:, -1].set_axis(wide.iloc[:, -2]).loc[codes]
        frozen = annual.pivot(index="Code", columns="Year", values=outcome).loc[
            codes, years
        ]
        assert long.notna().all().all() and processed.notna().all().all()
        np.testing.assert_allclose(stored, original[outcome], rtol=0, atol=1e-10)
        np.testing.assert_allclose(processed.mean(axis=1), stored, rtol=0, atol=1e-10)
        np.testing.assert_allclose(long, frozen, rtol=0, atol=1e-10)
        for year in years:
            changed = (processed[year] - long[year]).abs() > 1e-10
            if changed.any():
                matches = [
                    y
                    for y in years
                    if np.allclose(processed[year], long[y], atol=1e-10, rtol=0)
                ]
                assert matches == [year - 1]
                for code in codes[changed]:
                    cells.append(
                        dict(
                            outcome=outcome,
                            Code=code,
                            year=year,
                            raw=long.loc[code, year],
                            processed=processed.loc[code, year],
                            matching_raw_year=year - 1,
                        )
                    )
        # Preserve the submitted bit pattern when the annual source cells agree.
        changed_mean = (long.mean(axis=1) - original[outcome]).abs() > 1e-10
        corrected.loc[changed_mean, outcome] = long.mean(axis=1).loc[changed_mean]
        for code in codes:
            rows.append(
                dict(
                    outcome=outcome,
                    Code=code,
                    first_year=1992,
                    last_year=end,
                    n_observed=len(years),
                    supplied=original.loc[code, outcome],
                    processed_mean=processed.loc[code].mean(),
                    source_mean=corrected.loc[code, outcome],
                    source_minus_supplied=corrected.loc[code, outcome]
                    - original.loc[code, outcome],
                )
            )
    corrected.to_csv(out / "corrected_snapshot.csv")
    pd.DataFrame(rows).to_csv(out / "country_mean_comparison.csv", index=False)
    pd.DataFrame(cells).to_csv(out / "changed_annual_cells.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    run(p.parse_args().root)
