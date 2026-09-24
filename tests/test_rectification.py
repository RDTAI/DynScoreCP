import unittest

import numpy as np

from dynscore.rectification import ConditionalCDFRectifier, LocationScaleRectifier


class RectificationTests(unittest.TestCase):
    def test_conditional_cdf_compares_within_difficulty_bin(self) -> None:
        scores = np.array([0.1, 0.2, 0.7, 0.9])
        difficulty = np.array([0.0, 0.1, 1.0, 1.1])
        model = ConditionalCDFRectifier.fit(scores, difficulty, bins=2)
        transformed = model.transform(np.array([0.15, 0.8]), np.array([0.05, 1.05]))
        np.testing.assert_allclose(transformed, [2 / 3, 2 / 3])

    def test_location_scale_supports_candidate_score_matrix(self) -> None:
        model = LocationScaleRectifier.fit(
            np.array([0.0, 2.0, 10.0, 14.0]),
            np.array([0.0, 0.1, 1.0, 1.1]),
            bins=2,
        )
        result = model.transform(np.array([[1.0, 2.0], [12.0, 14.0]]), [0.0, 1.0])
        np.testing.assert_allclose(result, [[0.0, 1.0], [0.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
