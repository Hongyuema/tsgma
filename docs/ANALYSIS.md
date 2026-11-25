# Analysis map

The conceptual workflow advances pairs sequentially to reduce the number requiring later analyses. This repository implements the complete evaluation used in the paper: every eligible pair is fitted at every stage, and grades are assigned afterwards using the cumulative criteria. It is not an optimized early-stopping implementation. The complete calculation exposes findings that a strictly sequential screen could miss. A future staged implementation requires an explicit multiplicity strategy; correcting only within the retained subset is not assumed to reproduce these full-family results.


The pipeline retains separate outputs for the submitted country means, source-reconciled means, common-window models and primary temporal grading. They answer different questions and should not be substituted for one another.

| Stage | Main scripts | Output |
| --- | --- | --- |
| Country means | `audit_analysis.py`, `recover_outcomes.py` | Crude and partial correlations, uncertainty, covariate sensitivity and source reconciliation |
| Annual reconstruction | `reconstruct_annual.py` | Harmonized annual panel, variable definitions, country rosters and contemporaneous models |
| Primary grading | `grade_extended_period.py`, `third_star.py` | All 48 grades for 1990–2014, eligibility records and lag profiles |
| Observation periods | `search_periods.py`, `fixed_lags.py` | Period and lag sensitivity, including the complete exploratory search |
| Covariates and dependence | `current_food_models.py`, `current_model_diagnostics.py`, `regional_years.py` | Joint food models, residual diagnostics and regional year effects |
| Scale and composition | `paired_slopes.py`, `compare_scales.py`, `meat_contrast.py`, `within_meat_contrast.py` | Between/within comparisons and specified energy contrasts |
| Simulation | `simulate_screen.py`, `panel_benchmark.py`, `third_star_benchmark.py` | Performance under explicitly defined generating mechanisms |
| Verification | `verify_*.py` | Independent reconstructions of selected estimates and model diagnostics |
| Graphics | `style_figures.py`, `method_artwork.py`, `assemble_prism_figures.py` | Computational figure exports and assembly of archived Prism panels |

Run the pipeline through `code/reproduce.py`. Individual scripts use paths relative to the working analysis directory and have dependencies on earlier outputs.

`third_star.fit` removes country effects, year effects and optional country-specific linear trends from a complete balanced panel. Its covariance correction includes the absorbed terms in the model rank. It should not be used on an unbalanced panel by silently dropping rows or filling missing values. The main pipeline establishes complete pair-specific samples before calling it.

`third_star.grades` implements cumulative promotion. The first two stages use separate full-family BH corrections and require matching signs for promotion. The third stage uses the maximum of the three unadjusted P values when all signs agree, or 1 otherwise, followed by full-family Holm correction. Strong evidence for one component cannot compensate for weak evidence for another. In this complete reference implementation, a crude-screen failure prevents promotion but does not exclude a pair from calculation or reporting.

A window-specific grade depends on the years, lag definition and country sample used. The exploratory search evaluates all 48 pairs in every setting and reports the full search family. The historical 18-country trajectories are descriptive illustrations and do not determine eligibility or the third-star decision.

The main Figure 2–4 estimates use the historical country-mean periods with reconciled outcome values. Their 158-country crude sample and 143-country GDP-adjusted sample differ from the 119–122 countries eligible for the primary temporal analysis. These denominators are intentional and are recorded with the estimates.
