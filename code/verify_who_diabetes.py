"""Match WHO's archived annual prevalence and uncertainty bounds to the analytical panel."""

import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    source = json.loads(Path("inputs/WHO_NCD_GLUC_04.json").read_text())
    if "@odata.nextLink" in source:
        raise ValueError("Incomplete WHO response: additional records are available.")
    who = pd.DataFrame(source["value"])
    scope = {
        "IndicatorCode": "NCD_GLUC_04",
        "SpatialDimType": "COUNTRY",
        "Dim1": "SEX_BTSX",
        "Dim2": "AGEGROUP_YEARS18-PLUS",
    }
    for column, expected in scope.items():
        if set(who[column]) != {expected}:
            raise ValueError(f"Unexpected WHO dimension: {column}")
    who = who.rename(
        columns={
            "SpatialDim": "Code",
            "TimeDim": "Year",
            "NumericValue": "WHO_percent",
            "Low": "WHO_lower_percent",
            "High": "WHO_upper_percent",
        }
    )
    columns = ["Code", "Year", "WHO_percent", "WHO_lower_percent", "WHO_upper_percent"]
    panel = pd.read_csv("annual_results/annual_panel_1992_2017.csv")
    roster = pd.read_csv("outcome_audit/corrected_snapshot.csv").set_index("Code")
    panel = panel.loc[
        panel.Code.isin(roster.index) & panel.Year.between(1992, 2014),
        ["Code", "Year", "Diabetes1992-2014"],
    ].rename(columns={"Diabetes1992-2014": "panel_percent"})
    out = panel.merge(
        who[columns], on=["Code", "Year"], how="left", validate="one_to_one"
    ).sort_values(["Code", "Year"])
    expected = pd.MultiIndex.from_product(
        [roster.index, range(1992, 2015)], names=["Code", "Year"]
    ).sort_values()
    pd.testing.assert_index_equal(out.set_index(["Code", "Year"]).index, expected)
    if len(roster) != 158 or out.isna().any().any():
        raise ValueError("Incomplete country-year match or uncertainty interval.")
    np.testing.assert_array_equal(out.WHO_percent, out.panel_percent)
    observed = out.groupby("Code").WHO_percent.mean()
    np.testing.assert_allclose(
        observed, roster.loc[observed.index, "Diabetes1992-2014"], rtol=0, atol=1e-10
    )
    if not (
        out.WHO_lower_percent.ge(0)
        & out.WHO_lower_percent.le(out.WHO_percent)
        & out.WHO_upper_percent.ge(out.WHO_percent)
        & out.WHO_upper_percent.le(100)
    ).all():
        raise ValueError("Invalid source interval.")
    out["WHO_interval_width_pp"] = (
        out.WHO_upper_percent - out.WHO_lower_percent
    ).round(1)
    dest = Path("source_checks")
    dest.mkdir(exist_ok=True)
    out.to_csv(dest / "WHO_diabetes_country_year_check.csv", index=False)
    print(
        f"Verified {len(out)} annual values and {len(roster)} country means against WHO NCD_GLUC_04."
    )


if __name__ == "__main__":
    main()
