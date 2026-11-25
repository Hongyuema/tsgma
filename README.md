# TSGMA

Research code for Three-star Global Macro Association analysis. TSGMA combines country-mean correlations, covariate-adjusted correlations and within-country temporal models to grade candidate macro associations. The dietary application illustrates the method; the grades describe statistical support under specified models and do not establish causality.

This repository accompanies *Three-star Global Macro Association analysis for exploratory screening and the construction of macro association networks* by Hongyue Ma. The manuscript is under revision; no publication DOI has been assigned.

## Installation

The study environment uses Python 3.12. The package versions in `requirements.txt` are pinned to the environment used for the numerical reproduction check.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Run the example and tests

The synthetic example requires no study data. It fits a balanced panel with country and year effects, country-specific linear trends, and a five-year exposure average.

```bash
python examples/synthetic_panel.py
python -m unittest discover -s tests -v
```

The tests compare the transformed panel estimator with an explicit dummy-variable regression, including its country-clustered covariance. They also check the conjunction rule, multiplicity correction and the four possible grades.

## Reproduce the study

Extract `data/tsgma-data-v1.2.0.zip`. It contains the frozen numerical inputs, eligibility records, all reference tables, a data dictionary, source attributions and a SHA-256 manifest. Version 1.2.0 implements cumulative grading and accompanies the revised manuscript. The version-specific data and code deposit is doi:10.5281/zenodo.24527413. Version 1.1.0 remains available as a historical release; its grading rule and reference results differ from this version.

```bash
python code/reproduce.py --analysis-only \
  --archive /path/to/tsgma-data-v1.2.0 \
  --output /path/to/tsgma-rebuild
```

The output directory must not already exist. The command verifies the archive checksums, reconstructs the country means and annual panels, runs the analyses and simulations, and compares all 116 numerical tables and four metadata records with the archived results. The analysis runs offline after its dependencies and inputs have been obtained. Original figure artwork is excluded from the public data package. Full figure assembly remains available for users with the separately licensed submission archive: omit `--analysis-only` and point `--archive` to that archive. This stage uses archived Prism exports and Illustrator artboards; it does not rerun Prism itself.

`reproduction_report.csv` records numerical and graphical comparisons. Numerical tables use absolute tolerance `1e-12` and relative tolerance `1e-9`, with separate checks of decisions at 0.05. Figure rendering can vary with fonts and graphics libraries; such differences are reported separately from numerical agreement.

## Primary specification

The conceptual workflow advances pairs sequentially to reduce the number requiring later analyses. This repository implements the complete evaluation used in the paper: every eligible pair is fitted at every stage, and grades are assigned afterwards using the cumulative criteria. It is not an optimized early-stopping implementation. The complete calculation exposes findings that a strictly sequential screen could miss. A future staged implementation requires an explicit multiplicity strategy; correcting only within the retained subset is not assumed to reproduce these full-family results.

- Outcome period: 1990–2014.
- Exposure history: 1985–2014; delayed exposure is the preceding five-year mean.
- Eligibility: complete records assessed separately for each of 48 dietary–outcome pairs, yielding 119–122 countries or regions.
- Country-mean adjustment: log GDP per capita and total energy supply.
- Temporal adjustment: current exposure, current and preceding five-year means of log GDP and total energy, country effects, year effects, and country-specific linear trends.
- Inference: country-clustered standard errors with finite-sample correction and `G − 1` degrees of freedom.

Stars accumulate across the three stages. One star requires crude BH-adjusted `q < 0.05`. Two stars require both crude and adjusted `q < 0.05`, with matching signs. A third star additionally requires the delayed estimate to have that sign and the largest of all three unadjusted two-sided P values to pass Holm correction across all 48 pairs. Sign disagreement sets the conjunction P to 1. Testing families retain every pair before screening. A pair supported only after adjustment stays at zero stars but remains in the results. Stage-specific BH control does not establish an FDR bound for the combined grade classification.

The primary analysis yields 29 zero-star, 12 one-star and seven two-star pairs, with no three-star pair. Six pairs pass only the adjusted screen. The complete primary estimates are in `reference_results/primary_grades.csv`.

The 1994–2004 cereals–diabetes result is exploratory. It passes the third-star rule within that setting but not the correction over all 952 period–lag settings. It is not a replacement for the full-period analysis.

## Repository contents

| Path | Purpose |
| --- | --- |
| `code/` | Analysis, source reconciliation, simulation, figure assembly and verification scripts |
| `examples/` | Reproducible synthetic panel example |
| `tests/` | Independent numerical checks of the core estimator and grading rule |
| `reference_results/` | Primary estimates reported with the study |
| `data/` | Frozen numerical archive, data dictionary, attribution and component licences |
| `docs/ANALYSIS.md` | Script map and interpretation of analytical stages |
| `docs/DATA.md` | Input structure, provenance and third-party materials |
| `docs/VALIDATION.md` | Scope and results of verification |
| `CITATION.cff` | Software citation metadata |

## Citation and licence

Use the repository's **Cite this repository** entry to cite the software, recording the release or commit used. The code is distributed under the MIT licence. That licence does not apply to the source datasets, manuscript, or third-party artwork. Development provenance is recorded in `docs/PROVENANCE.md`.

For questions about the method, report an issue with the script name, package versions and a minimal reproducible example. Do not include credentials or unpublished personal data.
