# Verification record

The publication preparation check was run on 24 November 2026 with Python 3.12 and the pinned packages in `requirements.txt`.

## Statistical tests

Six independent tests passed. They cover opposite-sign exclusion, the maximum-P conjunction rule, all four star grades, correction over the complete 48-pair family, removal of the absorbed panel terms, and agreement with an explicit dummy-variable regression and its cluster-robust covariance.

The synthetic example produced a delayed coefficient of 0.664 with a 95% interval of 0.564 to 0.763; its generating coefficient was 0.600. It is a demonstration, not an empirical validation of the method.

## Full study reconstruction

The complete pipeline was run against the frozen study archive in a new output directory. All 172 archived outputs were reproduced:

| Comparison | Files |
| --- | ---: |
| Byte-identical numerical tables | 108 |
| Numerically equivalent tables | 8 |
| Identical analysis metadata | 4 |
| Identical decoded PNG pixels | 26 |
| Identical PDFs after removing creation/modification timestamps | 26 |

The eight numerically equivalent tables passed tolerances of `rtol=1e-9` and `atol=1e-12`, together with the applicable decision checks. The file-level report is supplied as `docs/reproduction_report.csv`.

The pipeline also executed its model-specific verification scripts. The temporal verification covered 48 grades, 144 lag fits, and 240 unrestricted lag coefficients and confidence intervals.

## Changes made for distribution

All 43 analysis scripts were formatted, and their abstract syntax trees were compared with the archived scripts before changing the runner. Formatting did not change executable statements. The runner was then given an explicit `--archive` argument and made to execute the repository's scripts against the selected archive. This modified runner was used for the full reconstruction reported above.

These checks establish agreement with the frozen computational results and selected independent numerical calculations. They do not establish the validity of causal interpretation, general performance outside the evaluated simulations, or independence from the original data sources. GitHub-hosted continuous integration is a separate check of the synthetic tests, not a substitute for the full-data reproduction.
