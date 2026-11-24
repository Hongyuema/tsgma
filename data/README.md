# TSGMA research data, version 1.1.0

Hongyue Ma. 2026. TSGMA: data and code for macro association analysis.

This archive contains the frozen numerical inputs and reference results for the dietary application of Three-star Global Macro Association analysis. It supports reproduction of all numerical analyses, including eligibility assessment, country-mean associations, temporal models, sensitivity analyses, simulations and the complete period search. No participant-level records are included.

## Reproduction

Use the code in Hongyuema/tsgma, version 1.1.0, and the pinned Python environment described in its README. From the code repository:

```bash
python code/reproduce.py --analysis-only \
  --archive /path/to/tsgma-data-v1.1.0 \
  --output /path/to/new-reconstruction
```

The output directory must be new. The runner first checks SHA-256 hashes, then rebuilds the analyses in that directory and compares all 116 numerical tables and four analysis-metadata records. It does not download replacement data. The public archive excludes publication artwork and published source documents. The numerical observations in eight historical Prism projects are supplied as lossless CSV exports; the original project formatting is not needed for the statistical calculations.

## Contents

| Directory | Contents |
| --- | --- |
| `inputs` | Historical country-mean workbook, frozen WHO diabetes query, World Bank regional classifications and supplementary covariate series |
| `annual_sources` | Frozen dietary, total-energy, cereal, GDP and health series, with source metadata |
| `historical_sources` | Historical workbooks and dietary export used to reconstruct country means and source definitions |
| `historical_plot_inputs` | Numerical observations from eight descriptive time-series examples and the alcohol-source workbook |
| `grading_1990_2014` | Primary annual panel, pair-specific eligibility, grades and timing checks |
| `period_search` | All 952 period–lag settings and their estimates, including unsuccessful settings |
| Other analysis directories | Country-mean results, sensitivity analyses, diagnostics, simulations and source comparisons |

The reference results are comparison targets, not substitutes for executing the analysis. Historical and harmonized inputs are kept separate because their food definitions, source releases and coverage differ. The original release underlying some historical food means remains unresolved; the frozen working files reproduce those means without claiming otherwise. Machine-specific save-location hints have been removed from workbook metadata; all worksheet contents are unchanged. The public manifest and reference metadata record the distributed files and omit two non-computational source documents that are not included.

## Interpretation

The primary outcome period is 1990–2014, with exposure history from 1985. Pair-specific completeness yields 119–122 countries or regions. Of 48 pairs, 23 receive no star, 12 receive one star and 13 receive two stars; none receives three stars. The cereals–diabetes result for 1994–2004 is an exploratory illustration of observation-period dependence. It satisfies the within-setting rule but not correction over the full search.

Dietary measures represent food availability, not measured individual intake. The health series are population estimates. Associations and grades do not establish individual effects, causal direction or the validity of an intervention.

## Attribution and reuse

Read `SOURCES_AND_LICENCES.md` before reuse. The code's MIT licence does not cover these data. Source-specific rights and conditions are retained, including the non-commercial, share-alike terms recorded in the frozen metadata. No ownership of third-party data is claimed. The data dictionary identifies key variables and units; the manifest identifies exact file versions.
