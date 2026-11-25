# Cumulative grading verification

Version 1.2.0 retains the frozen inputs, analytical samples and model specifications. Eight independent tests passed, including checks that adjustment-only support cannot bypass the crude stage, direction reversals prevent promotion, all three P values enter the conjunction, and multiplicity correction uses the full family. Independent explicit-design calculations verified the primary and sensitivity panel estimates and grades.

The primary analysis, four timing comparisons, the earlier harmonized-window grading, all 952 exploratory settings and the 200-replicate-per-scenario grading benchmark were rerun. The primary counts are 29, 12, 7 and 0 for grades zero through three. The exploratory search still contains 17 within-setting third-star findings, all for cereals and diabetes, with none surviving correction across the full search. The current-and-delayed simulation assigns three stars in 550 of 600 designated-pair evaluations and in none of the other 9,000 evaluations.

A fresh full numerical reconstruction passed all 116 table comparisons (108 byte-identical and eight numerically equivalent within rtol=1e-9 and atol=1e-12) and all four metadata comparisons. The file-level report is `docs/cumulative_reproduction_report.csv`. These checks establish computational agreement, not general statistical validity.

The following sections document earlier verification of version 1.1.0, not the cumulative grading update.

# Verification record

## Public data release, 25 November 2026

Version 1.1.0 adds a numerical-only reproduction mode and the public data archive. A fresh reconstruction using that archive reproduced all 116 numerical tables: 108 were byte-identical and eight agreed within the stated tolerances. All four analysis-metadata comparisons passed. The public reference metadata record the distributed files: two non-computational source documents are excluded, and machine-specific workbook save-location hints are removed. All worksheet contents, numerical inputs and estimates were unchanged. All observations exported from the eight historical Prism projects were checked for exact equality with their original series. The six independent statistical tests also passed.

The file-level numerical report is `docs/public_data_reproduction_report.csv`. The earlier full-artwork verification below remains a separate check of the licensed submission archive; no public-artwork reproduction is claimed.

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
