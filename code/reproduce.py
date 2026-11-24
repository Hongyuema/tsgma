"""Rebuild the manuscript's analysis outputs from the frozen inputs."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image

INPUTS = (
    "inputs",
    "annual_sources",
    "historical_sources",
    "historical_plot_inputs",
    "artwork_sources",
    "publication_artwork",
)
RESULTS = (
    "results",
    "annual_results",
    "outcome_audit",
    "corrected_results",
    "diagnostics",
    "food_results",
    "comparison",
    "coverage",
    "contrasts",
    "historical_plot_audit",
    "food_release_comparison",
    "grading_1990_2014",
    "period_search",
    "fixed_lags",
    "gdp_screen",
    "region_comparison",
    "simulation",
    "panel_benchmark",
    "food_definitions",
    "source_checks",
)


def compare(root, output):
    """Check complete output coverage, table contents and decoded figure pixels."""
    checks = []
    for directory in RESULTS:
        expected = {p.name for p in (root / directory).glob("*.csv")}
        actual = {p.name for p in (output / directory).glob("*.csv")}
        if not expected or expected != actual:
            raise ValueError(
                f"CSV file list differs for {directory}: {expected^actual}"
            )
        for name in sorted(expected):
            key = Path(directory) / name
            left, right = root / key, output / key
            exact = left.read_bytes() == right.read_bytes()
            if not exact:
                # Preserve literal labels (including the full-sample label "None").
                a = pd.read_csv(left, keep_default_na=False)
                b = pd.read_csv(right, keep_default_na=False)
                pd.testing.assert_frame_equal(
                    a, b, rtol=1e-9, atol=1e-12, check_dtype=False, obj=str(key)
                )
                for column in a:
                    if column.lower().startswith(
                        ("p", "q")
                    ) and pd.api.types.is_numeric_dtype(a[column]):
                        np.testing.assert_array_equal(
                            a[column] < 0.05, b[column] < 0.05, err_msg=str(key)
                        )
            checks.append(
                {
                    "file": str(key),
                    "comparison": (
                        "byte-identical" if exact else "numerically equivalent"
                    ),
                }
            )
        if directory == "source_checks":
            continue  # The JSON files here are frozen source records.
        for archived in sorted((root / directory).glob("*.json")):
            key = Path(directory) / archived.name
            a = json.loads(archived.read_text())
            b = json.loads((output / key).read_text())
            # Different Python/NumPy/pandas versions are recorded separately from analysis settings.
            runtime_keys = (
                ("python", "numpy", "pandas")
                if archived.name == "analysis_metadata.json"
                else ()
            )
            runtime_matches = all(a.get(k) == b.get(k) for k in runtime_keys)
            for k in runtime_keys:
                a.pop(k, None)
                b.pop(k, None)
            if a != b:
                raise ValueError(f"Analysis metadata differ: {key}")
            checks.append(
                {
                    "file": str(key),
                    "comparison": (
                        "identical metadata"
                        if runtime_matches
                        else "same settings; runtime differs"
                    ),
                }
            )
    expected = {
        p.name for p in (root / "figures").iterdir() if p.suffix in (".pdf", ".png")
    }
    actual = {
        p.name for p in (output / "figures").iterdir() if p.suffix in (".pdf", ".png")
    }
    if not expected or expected != actual:
        raise ValueError(f"Figure file list differs: {expected^actual}")
    for name in sorted(expected):
        key = Path("figures") / name
        if name.endswith(".png"):
            with Image.open(root / key) as a, Image.open(output / key) as b:
                same = np.array_equal(
                    np.asarray(a.convert("RGBA")), np.asarray(b.convert("RGBA"))
                )
            result = "identical pixels" if same else "rendering differs"
        else:
            content = (output / key).read_bytes()
            if not content.startswith(b"%PDF-") or not content.rstrip().endswith(
                b"%%EOF"
            ):
                raise ValueError(f"Incomplete PDF: {name}")
            dates = rb"/(CreationDate|ModDate)\s*\(D:[^)]*\)"
            same = re.sub(dates, b"/Date()", content) == re.sub(
                dates, b"/Date()", (root / key).read_bytes()
            )
            result = (
                "identical PDF except dates"
                if same
                else "PDF bytes differ beyond dates"
            )
        checks.append({"file": str(key), "comparison": result})
    pd.DataFrame(checks).to_csv(output / "reproduction_report.csv", index=False)
    for value, n in (
        pd.Series(row["comparison"] for row in checks).value_counts().items()
    ):
        print(f"{value}: {n}", flush=True)


def run(root, output):
    if output.exists():
        raise SystemExit(f"Choose a new output directory. Already exists: {output}")
    if (
        output == root
        or root in output.parents
        and output.relative_to(root).parts[0]
        in (*INPUTS, *RESULTS, "code", "figures", "manuscripts")
    ):
        raise SystemExit(
            "Choose a separate output directory outside the source and result folders."
        )
    manifest = json.loads((root / "MANIFEST_SHA256.json").read_text())
    for key, record in manifest.items():
        if Path(key).parts[0] not in (
            *INPUTS,
            *RESULTS,
            "code",
            "figures",
            "requirements.txt",
        ):
            continue
        if hashlib.sha256((root / key).read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Archive file failed SHA-256 verification: {key}")
    output.mkdir(parents=True)
    for directory in INPUTS:
        shutil.copytree(root / directory, output / directory)
    (output / "comparison").mkdir()
    env = os.environ.copy()
    env.update(
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        MPLCONFIGDIR=str(output / ".matplotlib"),
    )

    def step(script, *args):
        print("Running", script, *args, flush=True)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parent / (script + ".py")),
                *args,
            ],
            cwd=output,
            env=env,
            check=True,
        )

    def audit(source, dest):
        step(
            "audit_analysis",
            "--input",
            source,
            "--regions",
            "inputs/worldbank_regions.json",
            "--output",
            dest,
        )

    audit("inputs/Foods-1992-2017.xlsx", "results")
    step(
        "reconstruct_annual",
        "--sources",
        "annual_sources",
        "--snapshot",
        "results/analysis_snapshot.csv",
        "--output",
        "annual_results",
    )
    step("recover_outcomes")
    step("roster_history")
    audit("outcome_audit/corrected_snapshot.csv", "corrected_results")
    step("compare_outcome_versions")
    step(
        "current_model_diagnostics",
        "--annual-results",
        "annual_results",
        "--countries",
        "results/countries_and_missingness.csv",
        "--output",
        "diagnostics",
    )
    for script, dest in [
        ("current_food_models", "food_results"),
        ("coverage_sensitivity", "coverage"),
        ("meat_contrast", "contrasts"),
    ]:
        step(script, "--annual-results", "annual_results", "--output", dest)
    step("compare_scales", "--output", "comparison/scale_comparison.csv")
    for script in (
        "historical_plots",
        "compare_food_releases",
        "fixed_lags",
        "third_star",
        "grade_extended_period",
        "search_periods",
        "third_star_benchmark",
        "gdp_screen_audit",
        "region_comparison",
        "paired_slopes",
        "within_meat_contrast",
        "regional_years",
        "transition",
        "simulate_screen",
        "panel_benchmark",
        "food_definitions",
        "source_comparisons",
        "verify_who_diabetes",
        "style_figures",
        "method_artwork",
        "assemble_prism_figures",
    ):
        step(script)
    step(
        "verify_annual",
        "--results",
        "annual_results",
        "--dependence-dir",
        "diagnostics",
    )
    step(
        "verify_diagnostics",
        "--annual-results",
        "annual_results",
        "--countries",
        "results/countries_and_missingness.csv",
        "--diagnostics",
        "diagnostics",
    )
    for script in (
        "verify_food_releases",
        "verify_fixed_lags",
        "verify_gdp_screen",
        "verify_benchmarks",
        "verify_food_definitions",
        "verify_transition",
        "verify_regional_years",
        "verify_third_star",
    ):
        step(script)
    compare(root, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New directory for the rebuilt inputs, outputs and comparison report.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Extracted study archive containing frozen inputs and reference outputs.",
    )
    args = parser.parse_args()
    run(args.archive.resolve(), args.output.resolve())
