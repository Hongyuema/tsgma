"""Independent numerical checks of the panel estimator and grading rule."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from third_star import directional_p, fit, grades, transform


class GradingTests(unittest.TestCase):
    def test_opposite_signs_cannot_receive_a_third_star(self):
        np.testing.assert_array_equal(
            directional_p([1, -1], [1e-12, 1e-12], [-1, 1], [1e-12, 1e-12]),
            [1, 1],
        )

    def test_conjunction_uses_the_less_precise_component(self):
        np.testing.assert_allclose(
            directional_p([1, -1], [0.001, 0.2], [1, -1], [0.3, 0.002]),
            [0.3, 0.2],
        )

    def test_cumulative_grades_keep_adjustment_only_pairs_at_zero(self):
        crude = np.ones(48)
        adjusted = np.ones(48)
        delayed = np.ones(48)
        crude[[0, 1, 2]] = 1e-9
        adjusted[[0, 1, 3]] = 1e-9
        delayed[[0, 3]] = 1e-9
        *_, stars = grades(crude, adjusted, np.ones(48), delayed,
                           np.ones(48), np.ones(48))
        np.testing.assert_array_equal(stars[:5], [3, 2, 1, 0, 0])

    def test_direction_reversal_prevents_promotion(self):
        p = np.full(48, 1e-9)
        crude_sign = np.ones(48)
        adjusted_sign = np.ones(48)
        delayed_sign = np.ones(48)
        adjusted_sign[0] = -1
        delayed_sign[1] = -1
        _, _, joint, _, stars = grades(p, p, adjusted_sign, p,
                                       delayed_sign, crude_sign)
        np.testing.assert_array_equal(stars[:3], [1, 2, 3])
        np.testing.assert_array_equal(joint[:2], [1, 1])

    def test_conjunction_includes_crude_evidence(self):
        p1 = np.full(48, 0.03)
        p2 = np.full(48, 1e-9)
        p3 = np.full(48, 1e-9)
        _, _, joint, holm, stars = grades(p1, p2, np.ones(48), p3,
                                         np.ones(48), np.ones(48))
        np.testing.assert_allclose(joint, 0.03)
        np.testing.assert_allclose(holm, 1)
        np.testing.assert_array_equal(stars, np.full(48, 2))

    def test_holm_controls_the_whole_family_before_screening(self):
        crude = np.ones(48)
        adjusted = np.ones(48)
        delayed = np.ones(48)
        crude[0] = adjusted[0] = 1e-6
        delayed[0] = 0.002
        _, _, _, holm, stars = grades(crude, adjusted, np.ones(48),
                                      delayed, np.ones(48), np.ones(48))
        self.assertAlmostEqual(holm[0], 0.096)
        self.assertEqual(stars[0], 2)

    def test_transformation_removes_absorbed_terms(self):
        rng = np.random.default_rng(184)
        countries, years = 9, 12
        country = rng.normal(size=(countries, 1, 1))
        year = rng.normal(size=(1, years, 1))
        trend = rng.normal(size=(countries, 1, 1)) * np.arange(years)[None, :, None]
        np.testing.assert_allclose(transform(country + year + trend), 0, atol=2e-14)

    def test_fit_matches_explicit_fixed_effects_and_cluster_covariance(self):
        rng = np.random.default_rng(721)
        countries, years = 18, 14
        x = rng.normal(size=(countries, years))
        z = rng.normal(size=(countries, years))
        y = (
            0.4 * x
            - 0.2 * z
            + rng.normal(size=(countries, 1))
            + rng.normal(size=(1, years))
            + rng.normal(size=(countries, 1)) * np.arange(years)
            + rng.normal(size=(countries, years))
        )
        fitted, _, _ = fit(y, [x, z], np.array([1.0, 0.0]))
        group = np.repeat(np.arange(countries), years)
        time = np.tile(np.arange(years), countries)
        country_dummies = pd.get_dummies(group, drop_first=True, dtype=float).to_numpy()
        year_dummies = pd.get_dummies(time, drop_first=True, dtype=float).to_numpy()
        design = np.column_stack(
            [
                np.ones(len(group)),
                country_dummies,
                year_dummies,
                country_dummies * time[:, None],
                x.ravel(),
                z.ravel(),
            ]
        )
        model = sm.OLS(y.ravel(), design).fit()
        covariance = cov_cluster(model, group, use_correction=True)
        np.testing.assert_allclose(
            [fitted["beta"], fitted["se"]],
            [model.params[-2], np.sqrt(covariance[-2, -2])],
            rtol=1e-9,
            atol=1e-11,
        )


if __name__ == "__main__":
    unittest.main()
