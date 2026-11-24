# Data and source materials

The empirical analysis uses national or regional aggregates. Dietary availability is measured in kcal/person/day; it is not individual dietary intake. Health outcomes are age-standardized population estimates. The analysis conditions on the released estimates and does not propagate all uncertainty in their original estimation models.

## Frozen inputs

The accompanying research-data archive has the following structure:

| Directory | Contents |
| --- | --- |
| `inputs/` | Historical country-mean workbook, region classifications and frozen source queries |
| `annual_sources/` | Dietary, GDP and health series and source metadata |
| `historical_sources/` | Source tables used to reconcile the historical country means |
| `historical_plot_inputs/` | Annual series underlying the descriptive illustrations |
| `artwork_sources/` | Original schematic and figure materials |
| `publication_artwork/` | Native Prism exports, Prism projects and original Illustrator artboards |

The archive also contains reference results and a SHA-256 manifest. `code/reproduce.py --archive ...` verifies these records before analysis. Paths are resolved locally; the repository does not fetch new releases in place of the frozen inputs.

## Source versions

The annual dietary reconstruction uses the OWID FAOSTAT food-balance release dated 25 February 2026. GDP per capita uses the Maddison Project Database 2023. The health series were obtained through WHO/OWID and frozen for the analysis. The metadata in the study archive identify the indicators and source records used for each variable.

FAO food balance sheets change methodology around 2009–2010. The reconstruction retains this distinction, and the sensitivity analyses examine its implications. Population denominators, category definitions and source-release substitutions can alter country means; a newly downloaded series is not necessarily equivalent to the study input.

## Access and reuse

The numerical input archive accompanies the submission and is distributed separately from this source-code repository. A permanent public data deposit has not yet been assigned. Public source data can be obtained from their providers, but replacing the frozen inputs with current releases does not guarantee reproduction of the reported results.

The MIT licence covers this repository's code. Data-provider terms and third-party artwork licences remain applicable to the separate archive. BioRender materials, published source documents and the original artwork are not relicensed by this repository. Do not upload those materials merely because they are needed for local figure reproduction.

The full figure-reproduction stage uses archived Prism exports. The computational archive also contains historical comparison panels and an earlier graphical-abstract variant. Final publication figures are supplied separately with the manuscript.
