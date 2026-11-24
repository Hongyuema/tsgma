# Data dictionary

## Identifiers and missing observations

`Code` identifies a country or region using the source's three-letter code. `Entity`, where present, gives its name. `Year` is a calendar year. A blank numerical field denotes a missing observation, not zero. Zero food supply is retained where reported. Country aggregates in the source files may include entities outside the analytical samples; eligibility files identify the actual samples. Do not infer a common sample from the number of rows in a source file.

## Analysis variables

| Variable | Definition and unit |
| --- | --- |
| Sixteen dietary components | Food energy available for consumption, kcal/person/day; item codes and aggregation rules are in `annual_results/source_item_mapping.csv` and the food-definition tables |
| `Total_energy` | Total dietary energy availability, kcal/person/day |
| `Non-HDL` | Age-standardized mean non-HDL cholesterol, mmol/L |
| `Diabetes1992-2014` | Age-standardized prevalence of raised fasting blood glucose/diabetes among adults aged 18 years or older, percent; the historical column name is retained and is not the year restriction for every analysis |
| `Hypertension` | Age-standardized hypertension prevalence among adults aged 30–79 years, percent |
| `GDP` | GDP per capita; the harmonized annual series uses Maddison Project Database 2023, international dollars at 2011 prices |
| `logGDP` | Natural logarithm of GDP per capita |
| `Median age` | Population median age, years |
| `Gender ratio` | Male-to-female population ratio, calculated as (100 − female percentage)/female percentage |
| `Mean years schooling` | Average years of schooling for ages 15–64; the historical mean uses available observations and is not a complete annual series |
| `Smoke` | Historical tobacco-use prevalence covariate, percent; source vintage is not fully documented in the original workbook |
| `BMI>25` | Historical overweight-prevalence covariate, percent; source vintage is not fully documented in the original workbook |
| `Calories` | Recorded historical calorie covariate; its original construction is unresolved and it is not substituted for harmonized `Total_energy` |

## Key files

`grading_1990_2014/panel_1985_2014.csv` is the primary harmonized annual panel. Eligibility is evaluated separately for each food–outcome pair. The preceding five-year exposure mean is constructed from years t−5 through t−1; it excludes year t. Current and delayed covariates, country and year effects, and country-specific linear trends are defined in the code.

`grading_1990_2014/third_star_grades.csv` contains the primary estimates and grades. Confidence limits are on the scale specified by the corresponding estimator. P values and adjusted probabilities are distinct quantities; `q` denotes false-discovery-rate adjustment and Holm-adjusted P values control family-wise error. A grade is an ordered support category under the stated model, not an effect-size measure.

`results/analysis_snapshot.csv` retains the historical 158-country working table. `outcome_audit/corrected_snapshot.csv` contains the health-mean reconstruction. These are distinct from the pair-specific primary samples and should not be pooled into a single table of independent observations.

Each `historical_plot_inputs/*.csv` contains `Code`, `variable`, `Year` and `value`. Values are exported directly from the numerical columns of the historical Prism projects. Their role is descriptive; the plotted examples do not determine eligibility for the primary temporal analysis.

## Reading the complete outputs

The directory and variable names are retained to keep the public archive compatible with the validated analysis. The code and `docs/ANALYSIS.md` in the repository map each script to its inputs and outputs. Source metadata describe definitions and original processing; historical metadata may describe the evidence available at an earlier analytical stage. The primary specification and its interpretation are stated in the archive README.
